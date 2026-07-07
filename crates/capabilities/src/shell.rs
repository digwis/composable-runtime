use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::process::Command;

/// Shell 能力 — 执行系统命令
///
/// 原生能力种子：让 AI 能执行任意 shell 命令，
/// 是"安装依赖""运行构建""操作 git"等能力的基础。
pub struct ShellCapability;

#[derive(Deserialize)]
struct ShellExecInput {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Serialize)]
struct ShellExecOutput {
    command: String,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    success: bool,
}

#[derive(Deserialize)]
struct ShellEnvInput {
    #[serde(default)]
    names: Vec<String>,
}

#[derive(Deserialize)]
struct ShellPathInput {
    name: String,
}

#[async_trait::async_trait]
impl Capability for ShellCapability {
    fn name(&self) -> &str {
        "shell"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["exec", "exec_bg", "env", "which"]
    }

    fn describe(&self) -> String {
        "Shell 能力 — 执行系统命令、管理环境变量".to_string()
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "exec" => {
                let input: ShellExecInput = msg.payload_as()?;

                let mut cmd = Command::new(&input.command);
                cmd.args(&input.args);

                if let Some(cwd) = &input.cwd {
                    cmd.current_dir(cwd);
                }

                for (k, v) in &input.env {
                    cmd.env(k, v);
                }

                cmd.stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());

                let child = cmd.spawn().map_err(|e| MessageError::Internal {
                    capability: "shell".into(),
                    detail: format!("启动命令失败: {}", e),
                })?;

                let result = if let Some(timeout) = input.timeout_secs {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout),
                        child.wait_with_output(),
                    )
                    .await
                    .map_err(|_| MessageError::Internal {
                        capability: "shell".into(),
                        detail: format!("命令超时 ({}s)", timeout),
                    })?
                    .map_err(|e| MessageError::Internal {
                        capability: "shell".into(),
                        detail: format!("等待命令失败: {}", e),
                    })?
                } else {
                    child.wait_with_output().await.map_err(|e| {
                        MessageError::Internal {
                            capability: "shell".into(),
                            detail: format!("等待命令失败: {}", e),
                        }
                    })?
                };

                let output = ShellExecOutput {
                    command: format!("{} {}", input.command, input.args.join(" ")),
                    stdout: String::from_utf8_lossy(&result.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&result.stderr).to_string(),
                    exit_code: result.status.code(),
                    success: result.status.success(),
                };

                Ok(Message::builder()
                    .from("shell")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("exec.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "exec_bg" => {
                let input: ShellExecInput = msg.payload_as()?;

                let mut cmd = Command::new(&input.command);
                cmd.args(&input.args);

                if let Some(cwd) = &input.cwd {
                    cmd.current_dir(cwd);
                }

                for (k, v) in &input.env {
                    cmd.env(k, v);
                }

                cmd.stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());

                let child = cmd.spawn().map_err(|e| MessageError::Internal {
                    capability: "shell".into(),
                    detail: format!("启动后台命令失败: {}", e),
                })?;

                let pid = child.id();

                Ok(Message::builder()
                    .from("shell")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("exec_bg.response")
                    .payload(serde_json::json!({
                        "command": input.command,
                        "pid": pid,
                        "running": true,
                    }))
                    .build())
            }

            "env" => {
                let input: ShellEnvInput = msg.payload_as()?;

                let env_vars: HashMap<String, String> = if input.names.is_empty() {
                    std::env::vars().collect()
                } else {
                    input
                        .names
                        .iter()
                        .filter_map(|n| std::env::var(n).ok().map(|v| (n.clone(), v)))
                        .collect()
                };

                Ok(Message::builder()
                    .from("shell")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("env.response")
                    .payload(serde_json::json!({
                        "env": env_vars,
                        "count": env_vars.len(),
                    }))
                    .build())
            }

            "which" => {
                let input: ShellPathInput = msg.payload_as()?;
                let name = &input.name;

                let result = Command::new("which")
                    .arg(name)
                    .output()
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "shell".into(),
                        detail: format!("查找命令失败: {}", e),
                    })?;

                let path = if result.status.success() {
                    String::from_utf8_lossy(&result.stdout).trim().to_string()
                } else {
                    String::new()
                };

                Ok(Message::builder()
                    .from("shell")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("which.response")
                    .payload(serde_json::json!({
                        "name": name,
                        "path": path,
                        "found": !path.is_empty(),
                    }))
                    .build())
            }

            _ => Err(MessageError::UnsupportedAction {
                capability: "shell".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
