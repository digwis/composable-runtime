use crate::capability::Capability;
use crate::message::{Message, MessageError, MessageResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 消息总线（Message Bus）— 能力间的消息路由中心
///
/// 消息总线负责将消息从来源路由到目标能力，
/// 并跟踪消息流转历史，支持审计和调试。
pub struct MessageBus {
    capabilities: Arc<RwLock<HashMap<String, Arc<dyn Capability>>>>,
    history: Arc<RwLock<Vec<MessageLog>>>,
}

/// 消息流转日志
#[derive(Debug, Clone)]
pub struct MessageLog {
    pub message: Message,
    pub result: String,
    pub timestamp: chrono_like::Iso8601,
}

mod chrono_like {
    pub type Iso8601 = String;

    pub fn now() -> String {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{}ms", d.as_millis()))
            .unwrap_or_default()
    }
}

impl MessageBus {
    pub fn new() -> Self {
        Self {
            capabilities: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 注册能力到消息总线
    ///
    /// 如果同名能力已注册，则跳过（保留旧实例）。
    /// 这是为了保护运行时累积的适应度数据（runtime_fitness）：
    /// 重复 register 会用新的 ScriptedCapability 实例覆盖旧实例，
    /// 而 ScriptedCapability::from_genome 会从 genome.fitness 克隆 runtime_fitness，
    /// 导致旧实例在内存中累积的适应度数据丢失。
    ///
    /// 变异体和交叉后代使用不同的名字（如 `xxx-v2`），不会与父代冲突。
    pub async fn register(&self, capability: Arc<dyn Capability>) {
        let name = capability.name().to_string();
        let mut caps = self.capabilities.write().await;
        if caps.contains_key(&name) {
            tracing::debug!("能力 '{}' 已注册，跳过重复注册（保留运行时适应度）", name);
            return;
        }
        tracing::info!("注册能力: {} v{}", name, capability.version());
        caps.insert(name, capability);
    }

    /// 强制覆盖注册（用于需要更新实现的场景，如变异后替换父代）
    ///
    /// 注意：这会丢失旧实例的运行时适应度，调用方应先通过 `__fitness__` 动作
    /// 获取并持久化旧适应度。
    pub async fn register_force(&self, capability: Arc<dyn Capability>) {
        let name = capability.name().to_string();
        tracing::info!("强制覆盖能力: {} v{}", name, capability.version());
        self.capabilities.write().await.insert(name, capability);
    }

    /// 发送消息到目标能力并获取响应
    pub async fn send(&self, msg: Message) -> MessageResult {
        let caps = self.capabilities.read().await;
        let capability = caps.get(&msg.to).ok_or(MessageError::NotRegistered {
            capability: msg.to.clone(),
        })?;

        tracing::info!(
            "消息路由: {} -> {} (action: {})",
            msg.from.as_deref().unwrap_or("orchestrator"),
            msg.to,
            msg.action
        );

        let result = capability.handle(&msg).await;

        // 记录消息日志
        let log = MessageLog {
            message: msg.clone(),
            result: match &result {
                Ok(r) => format!("ok -> {}", r.to),
                Err(e) => format!("error: {e}"),
            },
            timestamp: chrono_like::now(),
        };
        self.history.write().await.push(log);

        result
    }

    /// 获取消息流转历史
    pub async fn history(&self) -> Vec<MessageLog> {
        self.history.read().await.clone()
    }

    /// 列出所有已注册能力
    pub async fn list_capabilities(&self) -> Vec<String> {
        self.capabilities.read().await.keys().cloned().collect()
    }

    /// 获取能力引用（用于查询 is_native 等类型信息）
    pub async fn get_capability(&self, name: &str) -> Option<Arc<dyn Capability>> {
        self.capabilities.read().await.get(name).cloned()
    }

    /// 能力自省 — 返回所有能力的详细信息
    pub async fn introspect(&self) -> Vec<crate::orchestrator::CapabilityInfo> {
        let caps = self.capabilities.read().await;
        caps.values()
            .map(|cap| crate::orchestrator::CapabilityInfo {
                name: cap.name().to_string(),
                version: cap.version().to_string(),
                actions: cap.actions().iter().map(|s| s.to_string()).collect(),
                description: cap.describe(),
            })
            .collect()
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    struct MockCapability {
        name: String,
    }

    #[async_trait::async_trait]
    impl Capability for MockCapability {
        fn name(&self) -> &str {
            &self.name
        }
        fn actions(&self) -> Vec<&str> {
            vec!["ping"]
        }
        async fn handle(&self, msg: &Message) -> MessageResult {
            Ok(Message::builder()
                .from(&self.name)
                .to(msg.from.as_deref().unwrap_or("caller"))
                .action("pong")
                .payload(serde_json::json!({"echo": msg.payload}))
                .build())
        }
    }

    #[tokio::test]
    async fn test_register_and_list() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability {
            name: "cap_a".into(),
        }))
        .await;
        let list = bus.list_capabilities().await;
        assert_eq!(list, vec!["cap_a"]);
    }

    #[tokio::test]
    async fn test_register_duplicate_skipped() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability { name: "dup".into() }))
            .await;
        bus.register(Arc::new(MockCapability { name: "dup".into() }))
            .await;
        let list = bus.list_capabilities().await;
        assert_eq!(list.len(), 1);
    }

    #[tokio::test]
    async fn test_register_force_overwrites() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability {
            name: "force_cap".into(),
        }))
        .await;
        bus.register_force(Arc::new(MockCapability {
            name: "force_cap".into(),
        }))
        .await;
        assert_eq!(bus.list_capabilities().await.len(), 1);
    }

    #[tokio::test]
    async fn test_send_to_unregistered() {
        let bus = MessageBus::new();
        let msg = Message::builder().to("no_such").action("test").build();
        let result = bus.send(msg).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_success() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability {
            name: "echo".into(),
        }))
        .await;
        let msg = Message::builder()
            .from("test")
            .to("echo")
            .action("ping")
            .payload(serde_json::json!({"data": 42}))
            .build();
        let result = bus.send(msg).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.action, "pong");
        assert_eq!(resp.payload["echo"]["data"], 42);
    }

    #[tokio::test]
    async fn test_history_recorded() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability {
            name: "h_cap".into(),
        }))
        .await;
        let msg = Message::builder().to("h_cap").action("ping").build();
        let _ = bus.send(msg).await;
        let history = bus.history().await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].message.to, "h_cap");
    }

    #[tokio::test]
    async fn test_get_capability() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability {
            name: "get_cap".into(),
        }))
        .await;
        let cap = bus.get_capability("get_cap").await;
        assert!(cap.is_some());
        assert_eq!(cap.unwrap().name(), "get_cap");
    }

    #[tokio::test]
    async fn test_introspect() {
        let bus = MessageBus::new();
        bus.register(Arc::new(MockCapability {
            name: "intro_cap".into(),
        }))
        .await;
        let info = bus.introspect().await;
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].name, "intro_cap");
        assert!(info[0].actions.contains(&"ping".to_string()));
    }

    #[tokio::test]
    async fn test_default() {
        let bus = MessageBus::default();
        assert!(bus.list_capabilities().await.is_empty());
    }
}
