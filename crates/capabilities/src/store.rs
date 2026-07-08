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
                self.data
                    .write()
                    .await
                    .insert(input.key.clone(), input.value);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(action: &str, payload: serde_json::Value) -> Message {
        Message::builder()
            .from("test")
            .to("store")
            .action(action)
            .payload(payload)
            .build()
    }

    #[tokio::test]
    async fn test_store_set_and_get() {
        let cap = StoreCapability::new();
        let set_msg = make_msg("set", serde_json::json!({"key": "k1", "value": 42}));
        cap.handle(&set_msg).await.unwrap();
        let get_msg = make_msg("get", serde_json::json!({"key": "k1"}));
        let resp = cap.handle(&get_msg).await.unwrap();
        assert_eq!(resp.payload["found"], true);
        assert_eq!(resp.payload["value"], 42);
    }

    #[tokio::test]
    async fn test_store_get_missing() {
        let cap = StoreCapability::new();
        let get_msg = make_msg("get", serde_json::json!({"key": "no_such"}));
        let resp = cap.handle(&get_msg).await.unwrap();
        assert_eq!(resp.payload["found"], false);
    }

    #[tokio::test]
    async fn test_store_delete() {
        let cap = StoreCapability::new();
        cap.handle(&make_msg(
            "set",
            serde_json::json!({"key": "del", "value": "x"}),
        ))
        .await
        .unwrap();
        let resp = cap
            .handle(&make_msg("delete", serde_json::json!({"key": "del"})))
            .await
            .unwrap();
        assert_eq!(resp.payload["deleted"], true);
        let get_resp = cap
            .handle(&make_msg("get", serde_json::json!({"key": "del"})))
            .await
            .unwrap();
        assert_eq!(get_resp.payload["found"], false);
    }

    #[tokio::test]
    async fn test_store_delete_missing() {
        let cap = StoreCapability::new();
        let resp = cap
            .handle(&make_msg("delete", serde_json::json!({"key": "ghost"})))
            .await
            .unwrap();
        assert_eq!(resp.payload["deleted"], false);
    }

    #[tokio::test]
    async fn test_store_list() {
        let cap = StoreCapability::new();
        cap.handle(&make_msg(
            "set",
            serde_json::json!({"key": "a", "value": 1}),
        ))
        .await
        .unwrap();
        cap.handle(&make_msg(
            "set",
            serde_json::json!({"key": "b", "value": 2}),
        ))
        .await
        .unwrap();
        let resp = cap
            .handle(&make_msg("list", serde_json::json!({})))
            .await
            .unwrap();
        let keys = resp.payload["keys"].as_array().unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_store_unsupported_action() {
        let cap = StoreCapability::new();
        let msg = make_msg("clear", serde_json::json!({}));
        assert!(cap.handle(&msg).await.is_err());
    }

    #[test]
    fn test_store_metadata() {
        let cap = StoreCapability::new();
        assert_eq!(cap.name(), "store");
        assert!(cap.is_native());
        assert_eq!(cap.actions(), vec!["set", "get", "delete", "list"]);
    }

    #[test]
    fn test_store_default() {
        let cap = StoreCapability::default();
        assert_eq!(cap.name(), "store");
    }
}
