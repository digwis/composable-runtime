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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_builder_basic() {
        let msg = Message::builder()
            .from("cap_a")
            .to("cap_b")
            .action("hello")
            .payload(serde_json::json!({"name": "test"}))
            .build();

        assert_eq!(msg.from.as_deref(), Some("cap_a"));
        assert_eq!(msg.to, "cap_b");
        assert_eq!(msg.action, "hello");
        assert_eq!(msg.payload["name"], "test");
        assert!(!msg.id.is_empty());
    }

    #[test]
    fn test_message_builder_defaults() {
        let msg = Message::builder().build();
        assert!(msg.from.is_none());
        assert_eq!(msg.to, "");
        assert_eq!(msg.action, "");
        assert_eq!(msg.payload, serde_json::Value::Null);
    }

    #[test]
    fn test_message_metadata() {
        let msg = Message::builder()
            .to("cap")
            .action("act")
            .metadata("key1", "val1")
            .metadata("key2", "val2")
            .build();
        assert_eq!(msg.metadata.get("key1"), Some(&"val1".to_string()));
        assert_eq!(msg.metadata.get("key2"), Some(&"val2".to_string()));
    }

    #[test]
    fn test_message_payload_as() {
        #[derive(Deserialize)]
        struct Data { name: String, age: u32 }
        let msg = Message::builder()
            .payload(serde_json::json!({"name": "alice", "age": 30}))
            .build();
        let data: Data = msg.payload_as().unwrap();
        assert_eq!(data.name, "alice");
        assert_eq!(data.age, 30);
    }

    #[test]
    fn test_message_payload_as_invalid() {
        let msg = Message::builder()
            .payload(serde_json::json!({"name": "alice"}))
            .build();
        #[derive(Deserialize)]
        struct NeedAge { #[allow(dead_code)] age: u32 }
        let result: Result<NeedAge, _> = msg.payload_as();
        assert!(result.is_err());
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::builder()
            .from("a")
            .to("b")
            .action("act")
            .payload(serde_json::json!({"x": 1}))
            .build();
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.to, "b");
        assert_eq!(decoded.action, "act");
        assert_eq!(decoded.payload["x"], 1);
    }

    #[test]
    fn test_message_error_display() {
        let err = MessageError::UnsupportedAction {
            capability: "greet".into(),
            action: "dance".into(),
        };
        assert!(err.to_string().contains("greet"));
        assert!(err.to_string().contains("dance"));
    }
}
