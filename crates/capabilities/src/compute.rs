use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};

/// 计算能力 — 提供基本数学运算
pub struct ComputeCapability;

#[derive(Deserialize)]
struct ComputeInput {
    a: f64,
    b: f64,
}

#[derive(Serialize)]
struct ComputeOutput {
    result: f64,
    operation: String,
    operands: [f64; 2],
}

#[async_trait::async_trait]
impl Capability for ComputeCapability {
    fn name(&self) -> &str {
        "compute"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["add", "subtract", "multiply", "divide"]
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        let input: ComputeInput = msg.payload_as()?;

        let (result, op_name) = match msg.action.as_str() {
            "add" => (input.a + input.b, "add"),
            "subtract" => (input.a - input.b, "subtract"),
            "multiply" => (input.a * input.b, "multiply"),
            "divide" => {
                if input.b == 0.0 {
                    return Err(MessageError::Internal {
                        capability: "compute".into(),
                        detail: "除数不能为零".into(),
                    });
                }
                (input.a / input.b, "divide")
            }
            _ => {
                return Err(MessageError::UnsupportedAction {
                    capability: "compute".into(),
                    action: msg.action.clone(),
                })
            }
        };

        let output = ComputeOutput {
            result,
            operation: op_name.to_string(),
            operands: [input.a, input.b],
        };

        Ok(Message::builder()
            .from("compute")
            .to(msg.from.as_deref().unwrap_or("orchestrator"))
            .action(format!("{}.response", msg.action))
            .payload(serde_json::to_value(&output).unwrap())
            .build())
    }
}
