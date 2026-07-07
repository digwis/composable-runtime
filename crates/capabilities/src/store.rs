use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 存储能力 — 内存键值存储
pub struct StoreCapability {
    data: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl StoreCapability {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for StoreCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct StoreSetInput {
    key: String,
    value: serde_json::Value,
}

#[derive(Deserialize)]
struct StoreGetInput {
    key: String,
}

#[derive(Serialize)]
struct StoreGetOutput {
    key: String,
    value: Option<serde_json::Value>,
    found: bool,
}

#[derive(Serialize)]
struct StoreSetOutput {
    key: String,
    success: bool,
}

#[derive(Serialize)]
struct StoreDeleteOutput {
    key: String,
    deleted: bool,
}

#[async_trait::async_trait]
impl Capability for StoreCapability {
    fn name(&self) -> &str {
        "store"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["set", "get", "delete", "list"]
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "set" => {
                let input: StoreSetInput = msg.payload_as()?;
                self.data.write().await.insert(input.key.clone(), input.value);
                let output = StoreSetOutput {
                    key: input.key,
                    success: true,
                };
                Ok(Message::builder()
                    .from("store")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("set.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }
            "get" => {
                let input: StoreGetInput = msg.payload_as()?;
                let value = self.data.read().await.get(&input.key).cloned();
                let found = value.is_some();
                let output = StoreGetOutput {
                    key: input.key,
                    value,
                    found,
                };
                Ok(Message::builder()
                    .from("store")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("get.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }
            "delete" => {
                let input: StoreGetInput = msg.payload_as()?;
                let deleted = self.data.write().await.remove(&input.key).is_some();
                let output = StoreDeleteOutput {
                    key: input.key,
                    deleted,
                };
                Ok(Message::builder()
                    .from("store")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("delete.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }
            "list" => {
                let keys: Vec<String> = self.data.read().await.keys().cloned().collect();
                Ok(Message::builder()
                    .from("store")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("list.response")
                    .payload(serde_json::json!({ "keys": keys }))
                    .build())
            }
            _ => Err(MessageError::UnsupportedAction {
                capability: "store".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
