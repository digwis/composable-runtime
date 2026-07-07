use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};

/// 文件系统能力 — 操作系统级文件读写
///
/// 原生能力种子：让 AI 能操作真实文件系统，
/// 是构建"写网站""写代码""管理项目"等高阶能力的基础。
pub struct FsCapability;

#[derive(Deserialize)]
struct FsReadInput {
    path: String,
    #[serde(default)]
    encoding: Option<String>,
}

#[derive(Deserialize)]
struct FsWriteInput {
    path: String,
    content: String,
    #[serde(default)]
    append: bool,
}

#[derive(Deserialize)]
struct FsPathInput {
    path: String,
}

#[derive(Deserialize)]
struct FsMoveInput {
    from: String,
    to: String,
}

#[derive(Serialize)]
struct FsReadOutput {
    path: String,
    content: String,
    size: u64,
}

#[derive(Serialize)]
struct FsWriteOutput {
    path: String,
    bytes_written: u64,
    appended: bool,
}

#[derive(Serialize)]
struct FsListOutput {
    path: String,
    entries: Vec<FsEntry>,
}

#[derive(Serialize)]
struct FsEntry {
    name: String,
    is_dir: bool,
    size: u64,
}

#[derive(Serialize)]
struct FsDeleteOutput {
    path: String,
    deleted: bool,
}

#[derive(Serialize)]
struct FsMkdirOutput {
    path: String,
    created: bool,
}

#[derive(Serialize)]
struct FsMoveOutput {
    from: String,
    to: String,
    moved: bool,
}

#[async_trait::async_trait]
impl Capability for FsCapability {
    fn name(&self) -> &str {
        "fs"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["read", "write", "mkdir", "list", "delete", "move", "exists"]
    }

    fn describe(&self) -> String {
        "文件系统能力 — 读写文件、创建目录、列出内容、删除、移动".to_string()
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "read" => {
                let input: FsReadInput = msg.payload_as()?;
                let path = std::path::Path::new(&input.path);

                if !path.exists() {
                    return Err(MessageError::Internal {
                        capability: "fs".into(),
                        detail: format!("文件不存在: {}", input.path),
                    });
                }

                let bytes = tokio::fs::read(&input.path)
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "fs".into(),
                        detail: format!("读取失败: {}", e),
                    })?;

                let content = match input.encoding.as_deref() {
                    Some("base64") => {
                        use base64::{engine::general_purpose, Engine};
                        general_purpose::STANDARD.encode(&bytes)
                    }
                    _ => String::from_utf8_lossy(&bytes).to_string(),
                };

                let output = FsReadOutput {
                    path: input.path,
                    content,
                    size: bytes.len() as u64,
                };

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("read.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "write" => {
                let input: FsWriteInput = msg.payload_as()?;
                let path = std::path::Path::new(&input.path);

                if let Some(parent) = path.parent() {
                    if !parent.exists() {
                        tokio::fs::create_dir_all(parent)
                            .await
                            .map_err(|e| MessageError::Internal {
                                capability: "fs".into(),
                                detail: format!("创建目录失败: {}", e),
                            })?;
                    }
                }

                let bytes_written = if input.append {
                    use tokio::io::AsyncWriteExt;
                    let mut f = tokio::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&input.path)
                        .await
                        .map_err(|e| MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("打开文件失败: {}", e),
                        })?;
                    f.write_all(input.content.as_bytes()).await.map_err(|e| {
                        MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("追加写入失败: {}", e),
                        }
                    })?;
                    input.content.len()
                } else {
                    tokio::fs::write(&input.path, &input.content)
                        .await
                        .map_err(|e| MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("写入失败: {}", e),
                        })?;
                    input.content.len()
                };

                let output = FsWriteOutput {
                    path: input.path,
                    bytes_written: bytes_written as u64,
                    appended: input.append,
                };

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("write.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "mkdir" => {
                let input: FsPathInput = msg.payload_as()?;
                let path = std::path::Path::new(&input.path);

                let created = if path.exists() {
                    false
                } else {
                    tokio::fs::create_dir_all(&input.path)
                        .await
                        .map_err(|e| MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("创建目录失败: {}", e),
                        })?;
                    true
                };

                let output = FsMkdirOutput {
                    path: input.path,
                    created,
                };

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("mkdir.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "list" => {
                let input: FsPathInput = msg.payload_as()?;
                let path = std::path::Path::new(&input.path);

                if !path.exists() {
                    return Err(MessageError::Internal {
                        capability: "fs".into(),
                        detail: format!("路径不存在: {}", input.path),
                    });
                }

                let mut entries = Vec::new();
                let mut dir = tokio::fs::read_dir(&input.path)
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "fs".into(),
                        detail: format!("读取目录失败: {}", e),
                    })?;

                while let Some(entry) = dir.next_entry().await.map_err(|e| {
                    MessageError::Internal {
                        capability: "fs".into(),
                        detail: format!("遍历目录失败: {}", e),
                    }
                })? {
                    let metadata = entry.metadata().await.map_err(|e| {
                        MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("读取元数据失败: {}", e),
                        }
                    })?;
                    entries.push(FsEntry {
                        name: entry.file_name().to_string_lossy().to_string(),
                        is_dir: metadata.is_dir(),
                        size: metadata.len(),
                    });
                }

                entries.sort_by(|a, b| {
                    b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name))
                });

                let output = FsListOutput { path: input.path, entries };

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("list.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "delete" => {
                let input: FsPathInput = msg.payload_as()?;
                let path = std::path::Path::new(&input.path);

                let deleted = if path.is_dir() {
                    tokio::fs::remove_dir_all(&input.path)
                        .await
                        .map_err(|e| MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("删除目录失败: {}", e),
                        })?;
                    true
                } else if path.exists() {
                    tokio::fs::remove_file(&input.path)
                        .await
                        .map_err(|e| MessageError::Internal {
                            capability: "fs".into(),
                            detail: format!("删除文件失败: {}", e),
                        })?;
                    true
                } else {
                    false
                };

                let output = FsDeleteOutput { path: input.path, deleted };

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("delete.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "move" => {
                let input: FsMoveInput = msg.payload_as()?;

                if let Some(parent) = std::path::Path::new(&input.to).parent() {
                    if !parent.exists() {
                        tokio::fs::create_dir_all(parent)
                            .await
                            .map_err(|e| MessageError::Internal {
                                capability: "fs".into(),
                                detail: format!("创建目标目录失败: {}", e),
                            })?;
                    }
                }

                tokio::fs::rename(&input.from, &input.to)
                    .await
                    .map_err(|e| MessageError::Internal {
                        capability: "fs".into(),
                        detail: format!("移动失败: {}", e),
                    })?;

                let output = FsMoveOutput {
                    from: input.from,
                    to: input.to,
                    moved: true,
                };

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("move.response")
                    .payload(serde_json::to_value(&output).unwrap())
                    .build())
            }

            "exists" => {
                let input: FsPathInput = msg.payload_as()?;
                let exists = std::path::Path::new(&input.path).exists();

                Ok(Message::builder()
                    .from("fs")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("exists.response")
                    .payload(serde_json::json!({
                        "path": input.path,
                        "exists": exists,
                        "is_dir": std::path::Path::new(&input.path).is_dir(),
                        "is_file": std::path::Path::new(&input.path).is_file(),
                    }))
                    .build())
            }

            _ => Err(MessageError::UnsupportedAction {
                capability: "fs".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
