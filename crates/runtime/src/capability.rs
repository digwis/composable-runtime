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
}
