use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// HTTP 能力 — 发起 HTTP 请求
///
/// 原生能力种子：让 AI 能调用外部 API、下载资源，
/// 是"搜索信息""调用第三方服务""获取数据"等能力的基础。
pub struct HttpCapability {
    client: reqwest::Client,
}

impl HttpCapability {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("构建 HTTP 客户端失败"),
        }
    }
}

impl Default for HttpCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct HttpRequestInput {
    url: String,
    #[serde(default)]
    headers: std::collections::HashMap<String, String>,
    #[serde(default)]
    body: Option<serde_json::Value>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
struct HttpDownloadInput {
    url: String,
    path: String,
    #[serde(default)]
    headers: std::collections::HashMap<String, String>,
}

#[derive(Serialize)]
struct HttpResponseOutput {
    url: String,
    status: u16,
    headers: std::collections::HashMap<String, String>,
    body: String,
    success: bool,
}

#[derive(Serialize)]
struct HttpDownloadOutput {
    url: String,
    path: String,
    bytes: u64,
    success: bool,
}

#[async_trait::async_trait]
impl Capability for HttpCapability {
    fn name(&self) -> &str {
        "http"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["get", "post", "put", "delete", "download"]
    }

    fn describe(&self) -> String {
        "HTTP 能力 — 发起 GET/POST/PUT/DELETE 请求、下载文件".to_string()
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "get" | "post" | "put" | "delete" => {
                let input: HttpRequestInput = msg.payload_as()?;
                let method = match msg.action.as_str() {
                    "get" => reqwest::Method::GET,
                    "post" => reqwest::Method::POST,
                    "put" => reqwest::Method::PUT,
                    "delete" => reqwest::Method::DELETE,
                    _ => unreachable!(),
                };

                let mut req = self.client.request(method, &input.url);

                for (k, v) in &input.headers {
                    req = req.header(k, v);
                }

                if let Some(body) = &input.body {
                    req = req.json(body);
                }

                if let Some(timeout) = input.timeout_secs {
                    req = req.timeout(Duration::from_secs(timeout));
                }

                let resp = req.send().await.map_err(|e| MessageError::Internal {
                    capability: "http".into(),
                    detail: format!("请求失败: {}", e),
                })?;

                let status = resp.status().as_u16();
                let mut resp_headers = std::collections::HashMap::new();
                for (k, v) in resp.headers().iter() {
                    if let Ok(v_str) = v.to_str() {
                        resp_headers.insert(k.as_str().to_string(), v_str.to_string());
                    }
                }

                let body = resp.text().await.map_err(|e| MessageError::Internal {
                    capability: "http".into(),
                    detail: format!("读取响应体失败: {}", e),
                })?;

                let output = HttpResponseOutput {
                    url: input.url,
                    status,
                    headers: resp_headers,
                    success: status >= 200 && status < 300,
                    body,
                };

                Ok(Message::builder()
                    .from("http")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action(format!("{}.response", msg.action))
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "download" => {
                let input: HttpDownloadInput = msg.payload_as()?;

                let mut req = self.client.get(&input.url);
                for (k, v) in &input.headers {
                    req = req.header(k, v);
                }

                let resp = req.send().await.map_err(|e| MessageError::Internal {
                    capability: "http".into(),
                    detail: format!("下载请求失败: {}", e),
                })?;

                let bytes = resp.bytes().await.map_err(|e| MessageError::Internal {
                    capability: "http".into(),
                    detail: format!("读取下载数据失败: {}", e),
                })?;

                if let Some(parent) = std::path::Path::new(&input.path).parent() {
                    if !parent.exists() {
                        tokio::fs::create_dir_all(parent)
                            .await
                            .map_err(|e| MessageError::Internal {
                                capability: "http".into(),
                                detail: format!("创建目录失败: {}", e),
                            })?;
                    }
                }

                tokio::fs::write(&input.path, &bytes)
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "http".into(),
                        detail: format!("写入文件失败: {}", e),
                    })?;

                let output = HttpDownloadOutput {
                    url: input.url,
                    path: input.path,
                    bytes: bytes.len() as u64,
                    success: true,
                };

                Ok(Message::builder()
                    .from("http")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("download.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            _ => Err(MessageError::UnsupportedAction {
                capability: "http".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
