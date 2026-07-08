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
            .payload(serde_json::to_value(&output).unwrap_or_default())
            .build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(action: &str, a: f64, b: f64) -> Message {
        Message::builder()
            .from("test")
            .to("compute")
            .action(action)
            .payload(serde_json::json!({"a": a, "b": b}))
            .build()
    }

    #[tokio::test]
    async fn test_compute_add() {
        let cap = ComputeCapability;
        let resp = cap.handle(&make_msg("add", 3.0, 4.0)).await.unwrap();
        assert_eq!(resp.payload["result"], 7.0);
        assert_eq!(resp.payload["operation"], "add");
    }

    #[tokio::test]
    async fn test_compute_subtract() {
        let cap = ComputeCapability;
        let resp = cap.handle(&make_msg("subtract", 10.0, 3.0)).await.unwrap();
        assert_eq!(resp.payload["result"], 7.0);
    }

    #[tokio::test]
    async fn test_compute_multiply() {
        let cap = ComputeCapability;
        let resp = cap.handle(&make_msg("multiply", 6.0, 7.0)).await.unwrap();
        assert_eq!(resp.payload["result"], 42.0);
    }

    #[tokio::test]
    async fn test_compute_divide() {
        let cap = ComputeCapability;
        let resp = cap.handle(&make_msg("divide", 20.0, 4.0)).await.unwrap();
        assert_eq!(resp.payload["result"], 5.0);
    }

    #[tokio::test]
    async fn test_compute_divide_by_zero() {
        let cap = ComputeCapability;
        let result = cap.handle(&make_msg("divide", 10.0, 0.0)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_compute_unsupported_action() {
        let cap = ComputeCapability;
        let msg = Message::builder()
            .from("test")
            .to("compute")
            .action("power")
            .payload(serde_json::json!({"a": 2.0, "b": 3.0}))
            .build();
        assert!(cap.handle(&msg).await.is_err());
    }

    #[tokio::test]
    async fn test_compute_missing_fields() {
        let cap = ComputeCapability;
        let msg = Message::builder()
            .from("test")
            .to("compute")
            .action("add")
            .payload(serde_json::json!({"a": 1.0}))
            .build();
        assert!(cap.handle(&msg).await.is_err());
    }

    #[test]
    fn test_compute_metadata() {
        let cap = ComputeCapability;
        assert_eq!(cap.name(), "compute");
        assert!(cap.is_native());
        assert_eq!(cap.actions(), vec!["add", "subtract", "multiply", "divide"]);
    }
}
