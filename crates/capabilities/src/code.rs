use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// 代码执行能力 — 在系统运行时中执行代码
///
/// 原生能力种子：让 AI 能执行 Python/Node 代码，
/// 是"数据分析""生成代码并验证""自动化脚本"等能力的基础。
pub struct CodeCapability;

#[derive(Deserialize)]
struct CodeRunInput {
    code: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    env: std::collections::HashMap<String, String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
struct CodeFileInput {
    path: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Serialize)]
struct CodeRunOutput {
    language: String,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    success: bool,
}

#[async_trait::async_trait]
impl Capability for CodeCapability {
    fn name(&self) -> &str {
        "code"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["run_python", "run_node", "run_python_file", "run_node_file"]
    }

    fn describe(&self) -> String {
        "代码执行能力 — 运行 Python/Node 代码或脚本文件".to_string()
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "run_python" => {
                let input: CodeRunInput = msg.payload_as()?;
                let tmp = std::env::temp_dir().join(format!("code_{}.py", uuid::Uuid::new_v4()));

                tokio::fs::write(&tmp, &input.code)
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "code".into(),
                        detail: format!("写入临时文件失败: {}", e),
                    })?;

                let result = run_command("python3", &[tmp.to_string_lossy().to_string()], &input.cwd, &input.env, input.timeout_secs).await?;

                let _ = tokio::fs::remove_file(&tmp).await;

                let output = CodeRunOutput {
                    language: "python".into(),
                    stdout: result.0,
                    stderr: result.1,
                    exit_code: result.2,
                    success: result.3,
                };

                Ok(Message::builder()
                    .from("code")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("run_python.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "run_node" => {
                let input: CodeRunInput = msg.payload_as()?;
                let tmp = std::env::temp_dir().join(format!("code_{}.js", uuid::Uuid::new_v4()));

                tokio::fs::write(&tmp, &input.code)
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "code".into(),
                        detail: format!("写入临时文件失败: {}", e),
                    })?;

                let result = run_command("node", &[tmp.to_string_lossy().to_string()], &input.cwd, &input.env, input.timeout_secs).await?;

                let _ = tokio::fs::remove_file(&tmp).await;

                let output = CodeRunOutput {
                    language: "node".into(),
                    stdout: result.0,
                    stderr: result.1,
                    exit_code: result.2,
                    success: result.3,
                };

                Ok(Message::builder()
                    .from("code")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("run_node.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "run_python_file" => {
                let input: CodeFileInput = msg.payload_as()?;
                let mut args = vec![input.path.clone()];
                args.extend(input.args.iter().cloned());

                let result = run_command("python3", &args, &input.cwd, &std::collections::HashMap::new(), input.timeout_secs).await?;

                let output = CodeRunOutput {
                    language: "python".into(),
                    stdout: result.0,
                    stderr: result.1,
                    exit_code: result.2,
                    success: result.3,
                };

                Ok(Message::builder()
                    .from("code")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("run_python_file.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "run_node_file" => {
                let input: CodeFileInput = msg.payload_as()?;
                let mut args = vec![input.path.clone()];
                args.extend(input.args.iter().cloned());

                let result = run_command("node", &args, &input.cwd, &std::collections::HashMap::new(), input.timeout_secs).await?;

                let output = CodeRunOutput {
                    language: "node".into(),
                    stdout: result.0,
                    stderr: result.1,
                    exit_code: result.2,
                    success: result.3,
                };

                Ok(Message::builder()
                    .from("code")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("run_node_file.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            _ => Err(MessageError::UnsupportedAction {
                capability: "code".into(),
                action: msg.action.clone(),
            }),
        }
    }
}

async fn run_command(
    program: &str,
    args: &[String],
    cwd: &Option<String>,
    env: &std::collections::HashMap<String, String>,
    timeout_secs: Option<u64>,
) -> Result<(String, String, Option<i32>, bool), MessageError> {
    let mut cmd = Command::new(program);
    cmd.args(args);

    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }

    for (k, v) in env {
        cmd.env(k, v);
    }

    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let child = cmd.spawn().map_err(|e| MessageError::Internal {
        capability: "code".into(),
        detail: format!("启动 {} 失败: {}", program, e),
    })?;

    let result = if let Some(timeout) = timeout_secs {
        tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| MessageError::Internal {
            capability: "code".into(),
            detail: format!("执行超时 ({}s)", timeout),
        })?
        .map_err(|e| MessageError::Internal {
            capability: "code".into(),
            detail: format!("等待失败: {}", e),
        })?
    } else {
        child.wait_with_output().await.map_err(|e| {
            MessageError::Internal {
                capability: "code".into(),
                detail: format!("等待失败: {}", e),
            }
        })?
    };

    Ok((
        String::from_utf8_lossy(&result.stdout).to_string(),
        String::from_utf8_lossy(&result.stderr).to_string(),
        result.status.code(),
        result.status.success(),
    ))
}
