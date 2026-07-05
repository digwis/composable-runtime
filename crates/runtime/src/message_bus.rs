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
    pub async fn register(&self, capability: Arc<dyn Capability>) {
        let name = capability.name().to_string();
        tracing::info!("注册能力: {} v{}", name, capability.version());
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
