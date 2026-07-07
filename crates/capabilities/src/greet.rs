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
                    .payload(serde_json::to_value(&output).unwrap())
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
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }
            _ => Err(MessageError::UnsupportedAction {
                capability: "greet".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
