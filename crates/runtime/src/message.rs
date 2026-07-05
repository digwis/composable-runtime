use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// 消息 — 能力间协作的统一通信单元
///
/// 每条消息包含：唯一 ID、来源能力、目标能力、动作名、
/// JSON 负载和可选元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub from: Option<String>,
    pub to: String,
    pub action: String,
    pub payload: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

/// 消息处理结果
pub type MessageResult = Result<Message, MessageError>;

/// 消息处理错误
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum MessageError {
    #[error("能力 '{capability}' 不支持动作 '{action}'")]
    UnsupportedAction { capability: String, action: String },

    #[error("负载校验失败: {detail}")]
    InvalidPayload { detail: String },

    #[error("能力 '{capability}' 内部错误: {detail}")]
    Internal { capability: String, detail: String },

    #[error("能力 '{capability}' 未注册")]
    NotRegistered { capability: String },
}

impl Message {
    /// 创建消息构建器
    pub fn builder() -> MessageBuilder {
        MessageBuilder::default()
    }

    /// 从 JSON 负载中提取类型化数据
    pub fn payload_as<T: for<'de> Deserialize<'de>>(&self) -> Result<T, MessageError> {
        serde_json::from_value(self.payload.clone())
            .map_err(|e| MessageError::InvalidPayload { detail: e.to_string() })
    }
}

/// 消息构建器
#[derive(Default)]
pub struct MessageBuilder {
    from: Option<String>,
    to: Option<String>,
    action: Option<String>,
    payload: Option<serde_json::Value>,
    metadata: HashMap<String, String>,
}

impl MessageBuilder {
    pub fn from(mut self, cap: impl Into<String>) -> Self {
        self.from = Some(cap.into());
        self
    }

    pub fn to(mut self, cap: impl Into<String>) -> Self {
        self.to = Some(cap.into());
        self
    }

    pub fn action(mut self, act: impl Into<String>) -> Self {
        self.action = Some(act.into());
        self
    }

    pub fn payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn build(self) -> Message {
        Message {
            id: Uuid::new_v4().to_string(),
            from: self.from,
            to: self.to.unwrap_or_default(),
            action: self.action.unwrap_or_default(),
            payload: self.payload.unwrap_or(serde_json::Value::Null),
            metadata: self.metadata,
        }
    }
}
