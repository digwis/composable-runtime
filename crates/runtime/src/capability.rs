use crate::message::{Message, MessageResult};

/// 能力（Capability）— 可组合的软件单元
///
/// 每个能力声明自己的名称、版本和支持的动作列表，
/// 并通过 `handle` 方法处理传入的消息。
///
/// 能力之间不直接调用，而是通过消息协作，
/// 由运行时负责路由和编排。
#[async_trait::async_trait]
pub trait Capability: Send + Sync {
    /// 能力名称（全局唯一标识）
    fn name(&self) -> &str;

    /// 能力版本
    fn version(&self) -> &str {
        "0.1.0"
    }

    /// 该能力支持的动作列表
    fn actions(&self) -> Vec<&str> {
        vec![]
    }

    /// 处理消息
    ///
    /// 接收一条消息，返回处理结果消息。
    /// 能力内部不应直接调用其他能力，
    /// 而是通过返回消息让运行时进行路由。
    async fn handle(&self, msg: &Message) -> MessageResult;

    /// 能力描述（用于自省和文档）
    fn describe(&self) -> String {
        format!(
            "Capability '{}' v{} — actions: {:?}",
            self.name(),
            self.version(),
            self.actions()
        )
    }

    /// 是否为原生能力（Rust 编译实现，不可变异）
    ///
    /// 原生能力没有基因组，不参与进化引擎的变异/淘汰/适应度同步。
    /// ScriptedCapability 返回 false，原生实现返回 true。
    ///
    /// 用类型方法替代硬编码的 `["greet", "compute", ...]` 字符串列表——
    /// 后者每新增原生能力都要记得改列表，是典型的不变量被防御性代码掩盖。
    fn is_native(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCap {
        cap_name: String,
        cap_version: String,
        cap_actions: Vec<String>,
        native: bool,
    }

    #[async_trait::async_trait]
    impl Capability for TestCap {
        fn name(&self) -> &str {
            &self.cap_name
        }
        fn version(&self) -> &str {
            &self.cap_version
        }
        fn actions(&self) -> Vec<&str> {
            self.cap_actions.iter().map(|s| s.as_str()).collect()
        }
        fn is_native(&self) -> bool {
            self.native
        }
        async fn handle(&self, msg: &Message) -> MessageResult {
            Ok(Message::builder()
                .from(&self.cap_name)
                .to(msg.from.as_deref().unwrap_or("caller"))
                .action("resp")
                .payload(msg.payload.clone())
                .build())
        }
    }

    #[tokio::test]
    async fn test_default_version() {
        struct BareCap;
        #[async_trait::async_trait]
        impl Capability for BareCap {
            fn name(&self) -> &str {
                "bare"
            }
            async fn handle(&self, _: &Message) -> MessageResult {
                Ok(Message::builder().from("bare").to("x").action("r").build())
            }
        }
        let cap = BareCap;
        assert_eq!(cap.version(), "0.1.0");
    }

    #[tokio::test]
    async fn test_default_actions() {
        struct BareCap;
        #[async_trait::async_trait]
        impl Capability for BareCap {
            fn name(&self) -> &str {
                "bare"
            }
            async fn handle(&self, _: &Message) -> MessageResult {
                Ok(Message::builder().from("bare").to("x").action("r").build())
            }
        }
        let cap = BareCap;
        assert!(cap.actions().is_empty());
    }

    #[tokio::test]
    async fn test_default_is_native() {
        struct BareCap;
        #[async_trait::async_trait]
        impl Capability for BareCap {
            fn name(&self) -> &str {
                "bare"
            }
            async fn handle(&self, _: &Message) -> MessageResult {
                Ok(Message::builder().from("bare").to("x").action("r").build())
            }
        }
        let cap = BareCap;
        assert!(!cap.is_native());
    }

    #[tokio::test]
    async fn test_describe() {
        let cap = TestCap {
            cap_name: "test_cap".into(),
            cap_version: "1.0.0".into(),
            cap_actions: vec!["act1".into(), "act2".into()],
            native: true,
        };
        let desc = cap.describe();
        assert!(desc.contains("test_cap"));
        assert!(desc.contains("1.0.0"));
        assert!(desc.contains("act1"));
    }

    #[tokio::test]
    async fn test_handle_echo() {
        let cap = TestCap {
            cap_name: "echo".into(),
            cap_version: "0.1.0".into(),
            cap_actions: vec!["echo".into()],
            native: false,
        };
        let msg = Message::builder()
            .from("test")
            .to("echo")
            .action("echo")
            .payload(serde_json::json!({"x": 1}))
            .build();
        let resp = cap.handle(&msg).await.unwrap();
        assert_eq!(resp.action, "resp");
        assert_eq!(resp.payload["x"], 1);
    }

    #[tokio::test]
    async fn test_custom_version_and_actions() {
        let cap = TestCap {
            cap_name: "custom".into(),
            cap_version: "2.0.0".into(),
            cap_actions: vec!["a".into(), "b".into(), "c".into()],
            native: true,
        };
        assert_eq!(cap.version(), "2.0.0");
        assert_eq!(cap.actions(), vec!["a", "b", "c"]);
        assert!(cap.is_native());
    }
}
