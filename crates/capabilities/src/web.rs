use runtime::{Capability, Message, MessageError, MessageResult};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::process::Command;

/// Web 服务器能力 — 启动/停止静态文件 HTTP 服务
///
/// 原生能力种子：让 AI 能启动 Web 服务，
/// 是"部署网站""提供 API""预览页面"等能力的基础。
pub struct WebCapability {
    servers: Arc<RwLock<std::collections::HashMap<String, u32>>>,
}

impl WebCapability {
    pub fn new() -> Self {
        Self {
            servers: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for WebCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct WebServeInput {
    root: String,
    #[serde(default = "default_port")]
    port: u16,
}

fn default_port() -> u16 {
    8080
}

#[derive(Deserialize)]
struct WebStopInput {
    port: u16,
}

#[async_trait::async_trait]
impl Capability for WebCapability {
    fn name(&self) -> &str {
        "web"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["serve", "stop", "list"]
    }

    fn describe(&self) -> String {
        "Web 服务器能力 — 启动/停止静态文件 HTTP 服务".to_string()
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "serve" => {
                let input: WebServeInput = msg.payload_as()?;

                let root = std::path::Path::new(&input.root);
                if !root.exists() {
                    return Err(MessageError::Internal {
                        capability: "web".into(),
                        detail: format!("根目录不存在: {}", input.root),
                    });
                }

                let port = input.port;

                let mut cmd = Command::new("python3");
                cmd.arg("-m").arg("http.server").arg(port.to_string());
                cmd.current_dir(&input.root);
                cmd.stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());

                let child = cmd.spawn().map_err(|e| MessageError::Internal {
                    capability: "web".into(),
                    detail: format!("启动 Web 服务器失败: {}", e),
                })?;

                let pid = child.id().unwrap_or(0);

                self.servers.write().await.insert(port.to_string(), pid);

                let url = format!("http://localhost:{}", port);

                Ok(Message::builder()
                    .from("web")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("serve.response")
                    .payload(serde_json::json!({
                        "root": input.root,
                        "port": port,
                        "url": url,
                        "pid": pid,
                        "running": true,
                    }))
                    .build())
            }

            "stop" => {
                let input: WebStopInput = msg.payload_as()?;

                let pid = self.servers.write().await.remove(&input.port.to_string());

                if let Some(pid) = pid {
                    #[cfg(unix)]
                    {
                        let _ = Command::new("kill")
                            .arg(pid.to_string())
                            .output()
                            .await;
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = Command::new("taskkill")
                            .args(["/PID", &pid.to_string(), "/F"])
                            .output()
                            .await;
                    }

                    Ok(Message::builder()
                        .from("web")
                        .to(msg.from.as_deref().unwrap_or("orchestrator"))
                        .action("stop.response")
                        .payload(serde_json::json!({
                            "port": input.port,
                            "stopped": true,
                        }))
                        .build())
                } else {
                    Ok(Message::builder()
                        .from("web")
                        .to(msg.from.as_deref().unwrap_or("orchestrator"))
                        .action("stop.response")
                        .payload(serde_json::json!({
                            "port": input.port,
                            "stopped": false,
                            "error": "该端口没有运行中的服务器",
                        }))
                        .build())
                }
            }

            "list" => {
                let servers = self.servers.read().await;
                let list: Vec<serde_json::Value> = servers
                    .iter()
                    .map(|(port, pid)| {
                        serde_json::json!({
                            "port": port.parse::<u16>().unwrap_or(0),
                            "pid": pid,
                            "url": format!("http://localhost:{}", port),
                        })
                    })
                    .collect();

                Ok(Message::builder()
                    .from("web")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("list.response")
                    .payload(serde_json::json!({
                        "servers": list,
                        "count": list.len(),
                    }))
                    .build())
            }

            _ => Err(MessageError::UnsupportedAction {
                capability: "web".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
