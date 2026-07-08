use crate::capability::Capability;
use crate::message_bus::MessageBus;
use std::sync::Arc;

/// 能力注册中心（Registry）— 管理所有能力的生命周期
///
/// Registry 是 MessageBus 的构建器封装，
/// 提供链式注册 API，并在构建完成后交出 MessageBus。
pub struct RegistryBuilder {
    bus: MessageBus,
}

impl RegistryBuilder {
    pub fn new() -> Self {
        Self {
            bus: MessageBus::new(),
        }
    }

    /// 注册一个能力
    pub fn with(self, capability: impl Capability + 'static) -> Self {
        let arc: Arc<dyn Capability> = Arc::new(capability);
        // 使用 blocking 方式注册（在 builder 阶段是同步的）
        // 这里通过 try_write 避免死锁
        futures_lite::block_on(self.bus.register(arc));
        self
    }

    /// 注册一个已 Arc 包装的能力
    pub fn with_arc(self, capability: Arc<dyn Capability>) -> Self {
        futures_lite::block_on(self.bus.register(capability));
        self
    }

    /// 构建完成，返回 MessageBus
    pub fn build(self) -> MessageBus {
        self.bus
    }
}

impl Default for RegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 简化版 Registry — 对 MessageBus 的别名
pub type Registry = MessageBus;

mod futures_lite {
    pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, MessageResult};

    struct DummyCap;
    #[async_trait::async_trait]
    impl Capability for DummyCap {
        fn name(&self) -> &str {
            "dummy"
        }
        async fn handle(&self, _: &Message) -> MessageResult {
            Ok(Message::builder().from("dummy").to("x").action("r").build())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_registry_builder_with() {
        let bus = RegistryBuilder::new().with(DummyCap).build();
        let caps = bus.list_capabilities().await;
        assert_eq!(caps, vec!["dummy"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_registry_builder_chained() {
        let bus = RegistryBuilder::new()
            .with(DummyCap)
            .with_arc(Arc::new(DummyCap))
            .build();
        assert_eq!(bus.list_capabilities().await.len(), 1);
    }

    #[tokio::test]
    async fn test_registry_builder_empty() {
        let bus = RegistryBuilder::new().build();
        assert!(bus.list_capabilities().await.is_empty());
    }

    #[tokio::test]
    async fn test_registry_builder_default() {
        let bus = RegistryBuilder::default().build();
        assert!(bus.list_capabilities().await.is_empty());
    }

    #[tokio::test]
    async fn test_registry_type_alias() {
        let _: Registry = MessageBus::new();
    }
}
