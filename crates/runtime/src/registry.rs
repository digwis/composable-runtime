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
        // 在同步上下文中执行 future
        // 使用 tokio 的 current_thread runtime
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(f)
        })
    }
}
