use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};

/// 问候能力 — 根据名字生成问候语
pub struct GreetCapability;

#[derive(Deserialize)]
struct GreetInput {
    name: String,
    #[serde(default = "default_greeting")]
    greeting: String,
}

fn default_greeting() -> String {
    "你好".to_string()
}

#[derive(Serialize)]
struct GreetOutput {
    message: String,
    original_name: String,
}

#[async_trait::async_trait]
impl Capability for GreetCapability {
    fn name(&self) -> &str {
        "greet"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["hello", "goodbye"]
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "hello" => {
                let input: GreetInput = msg.payload_as()?;
                let output = GreetOutput {
                    message: format!("{}, {}!", input.greeting, input.name),
                    original_name: input.name,
                };
                Ok(Message::builder()
                    .from("greet")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("hello.response")
                    .payload(serde_json::to_value(&output).unwrap_or_default())
                    .build())
            }
            "goodbye" => {
                let input: GreetInput = msg.payload_as()?;
                let output = GreetOutput {
                    message: format!("再见, {}! 期待下次相见。", input.name),
                    original_name: input.name,
                };
                Ok(Message::builder()
                    .from("greet")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("goodbye.response")
                    .payload(serde_json::to_value(&output).unwrap_or_default())
                    .build())
            }
            _ => Err(MessageError::UnsupportedAction {
                capability: "greet".into(),
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
            .to("greet")
            .action(action)
            .payload(payload)
            .build()
    }

    #[tokio::test]
    async fn test_greet_hello() {
        let cap = GreetCapability;
        let msg = make_msg("hello", serde_json::json!({"name": "世界"}));
        let resp = cap.handle(&msg).await.unwrap();
        assert_eq!(resp.action, "hello.response");
        assert_eq!(resp.payload["message"], "你好, 世界!");
    }

    #[tokio::test]
    async fn test_greet_hello_custom_greeting() {
        let cap = GreetCapability;
        let msg = make_msg(
            "hello",
            serde_json::json!({"name": "Bob", "greeting": "Hello"}),
        );
        let resp = cap.handle(&msg).await.unwrap();
        assert_eq!(resp.payload["message"], "Hello, Bob!");
    }

    #[tokio::test]
    async fn test_greet_goodbye() {
        let cap = GreetCapability;
        let msg = make_msg("goodbye", serde_json::json!({"name": "Alice"}));
        let resp = cap.handle(&msg).await.unwrap();
        assert_eq!(resp.action, "goodbye.response");
        assert!(resp.payload["message"].as_str().unwrap().contains("Alice"));
    }

    #[tokio::test]
    async fn test_greet_unsupported_action() {
        let cap = GreetCapability;
        let msg = make_msg("unknown", serde_json::json!({"name": "test"}));
        assert!(cap.handle(&msg).await.is_err());
    }

    #[tokio::test]
    async fn test_greet_missing_name() {
        let cap = GreetCapability;
        let msg = make_msg("hello", serde_json::json!({}));
        assert!(cap.handle(&msg).await.is_err());
    }

    #[test]
    fn test_greet_metadata() {
        let cap = GreetCapability;
        assert_eq!(cap.name(), "greet");
        assert_eq!(cap.version(), "0.1.0");
        assert!(cap.is_native());
        assert_eq!(cap.actions(), vec!["hello", "goodbye"]);
    }
}
