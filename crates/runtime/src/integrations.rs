//! Local cloud-service integration detection.
//!
//! Detection only asks each CLI for account metadata. Credential material is
//! never read from keychains, config files, environment variables, or output.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConnection {
    pub id: String,
    pub role: String,
    pub status: String,
    pub installed: bool,
    pub authenticated: bool,
    pub healthy: bool,
    pub cli_path: Option<String>,
    pub version: Option<String>,
    pub account: Option<String>,
    pub account_name: Option<String>,
    pub workspace: Option<String>,
    pub workspace_id: Option<String>,
    pub scopes: Vec<String>,
    pub warnings: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStatus {
    pub checked_at: u64,
    pub knowledge_backend: String,
    pub capability_backend: String,
    pub notion: ServiceConnection,
    pub github: ServiceConnection,
    pub git: ServiceConnection,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudResourceState {
    pub initialized_at: Option<u64>,
    pub notion_root: Option<CloudResource>,
    pub notion_children: Vec<CloudResource>,
    pub github_capability_repository: Option<CloudResource>,
    #[serde(default)]
    pub last_github_sync_at: Option<u64>,
    #[serde(default)]
    pub last_notion_sync_at: Option<u64>,
    #[serde(default)]
    pub github_sync_changed: bool,
    #[serde(default)]
    pub last_sync_errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudResource {
    pub id: String,
    pub name: String,
    pub url: Option<String>,
    pub kind: String,
    pub private: Option<bool>,
}

pub async fn detect_integrations() -> IntegrationStatus {
    let (notion, github, git) = tokio::join!(detect_notion(), detect_github(), detect_git());
    IntegrationStatus {
        checked_at: unix_now(),
        knowledge_backend: "notion".into(),
        capability_backend: "github".into(),
        notion,
        github,
        git,
    }
}

pub fn load_cloud_resources(storage_dir: &Path) -> CloudResourceState {
    std::fs::read_to_string(storage_dir.join("cloud_resources.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub async fn ensure_cloud_resources(storage_dir: &Path) -> Result<CloudResourceState, String> {
    let _guard = acquire_bootstrap_lock(storage_dir).await?;
    let integrations = detect_integrations().await;
    let mut state = load_cloud_resources(storage_dir);
    state.warnings.clear();

    let mut github_repository = None;
    if integrations.github.healthy {
        match ensure_github_repository(&integrations.github).await {
            Ok(repository) => {
                state.github_capability_repository = Some(repository.clone());
                github_repository = Some(repository);
            }
            Err(error) => state.warnings.push(format!("GitHub 初始化失败: {}", error)),
        }
    } else {
        state
            .warnings
            .push("GitHub 尚未连接，跳过能力仓库初始化".into());
    }

    if integrations.notion.healthy {
        let github_anchor = github_repository
            .as_ref()
            .map(|repository| (&integrations.github, repository));
        match ensure_notion_memory(&integrations.notion, github_anchor, storage_dir, &mut state)
            .await
        {
            Ok((root, children)) => {
                state.notion_root = Some(root);
                state.notion_children = children;
            }
            Err(error) => state.warnings.push(format!("Notion 初始化失败: {}", error)),
        }
    } else {
        state
            .warnings
            .push("Notion 尚未连接，跳过知识空间初始化".into());
    }

    state.initialized_at = Some(unix_now());
    persist_cloud_resources(storage_dir, &state)?;
    Ok(state)
}

async fn ensure_notion_memory(
    notion: &ServiceConnection,
    github_anchor: Option<(&ServiceConnection, &CloudResource)>,
    storage_dir: &Path,
    state: &mut CloudResourceState,
) -> Result<(CloudResource, Vec<CloudResource>), String> {
    let binary = notion
        .cli_path
        .as_deref()
        .map(PathBuf::from)
        .ok_or_else(|| "ntn CLI 路径缺失".to_string())?;
    const ROOT_TITLE: &str = "Composable Runtime Agent Memory";
    let local_root = if let Some(root) = state.notion_root.as_ref() {
        if notion_page_is_active(
            &command_json(&binary, &["pages", "get", &root.id, "--json"]).await,
        ) {
            Some(root.clone())
        } else {
            None
        }
    } else {
        None
    };
    let anchored_root = if local_root.is_none() {
        match github_anchor {
            Some((github, repository)) => {
                find_anchored_notion_root(&binary, github, repository).await
            }
            None => None,
        }
    } else {
        None
    };
    let root = match local_root.or(anchored_root) {
        Some(root) => root,
        None => match find_notion_page(&binary, ROOT_TITLE, None).await? {
            Some(root) => root,
            None => create_notion_page(&binary, ROOT_TITLE, notion_root_content(), None).await?,
        },
    };

    // Persist the remote identifier before creating children. If the daemon is
    // stopped during bootstrap, the next run resumes this page instead of
    // creating another workspace-level root.
    state.notion_root = Some(root.clone());
    persist_cloud_resources(storage_dir, state)?;
    if let Some((github, repository)) = github_anchor {
        if let Err(error) = persist_notion_root_anchor(github, repository, &root.id).await {
            tracing::warn!("GitHub Notion 根页面锚点同步失败: {}", error);
        }
    }

    let definitions = [
        ("User Memory", "用户长期目标、偏好、约束与稳定事实。"),
        ("Projects", "项目愿景、上下文、优先级、进展与真实反馈。"),
        ("Capabilities", "能力目录、适用场景、版本、验证与人类评价。"),
        ("Learnings", "实验、失败、根因、经验和后续改进。"),
        ("Decisions", "架构与产品决策、替代方案、理由和影响。"),
    ];
    let mut remote_children = list_notion_child_pages(&binary, &root.id).await?;
    let mut children = Vec::new();
    for (title, description) in definitions {
        let child = match remote_children
            .iter()
            .find(|resource| resource.name == title)
            .cloned()
        {
            Some(child) => child,
            None => {
                create_notion_page(
                    &binary,
                    title,
                    &format!(
                        "# {}\n\n{}\n\n由 composable-runtime 管理。",
                        title, description
                    ),
                    Some(&root.id),
                )
                .await?
            }
        };
        remote_children.push(child.clone());
        children.push(child.clone());
        if let Some(indexed) = state
            .notion_children
            .iter_mut()
            .find(|resource| resource.name == title)
        {
            *indexed = child;
        } else {
            state.notion_children.push(child);
        }
        persist_cloud_resources(storage_dir, state)?;
    }
    state.notion_children = children.clone();
    Ok((root, children))
}

async fn list_notion_child_pages(
    binary: &Path,
    root_id: &str,
) -> Result<Vec<CloudResource>, String> {
    let path = format!("v1/blocks/{}/children", root_id);
    let value = command_json(binary, &["api", &path, "page_size==100"]).await?;
    Ok(value
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|block| {
            block.get("type").and_then(Value::as_str) == Some("child_page")
                && block.get("in_trash").and_then(Value::as_bool) != Some(true)
        })
        .filter_map(|block| {
            let id = block.get("id").and_then(Value::as_str)?.to_string();
            let name = block
                .get("child_page")
                .and_then(|child| child.get("title"))
                .and_then(Value::as_str)?
                .to_string();
            Some(CloudResource {
                url: Some(format!("https://app.notion.com/p/{}", id.replace('-', ""))),
                id,
                name,
                kind: "notion_page".into(),
                private: None,
            })
        })
        .collect())
}

async fn find_notion_page(
    binary: &Path,
    title: &str,
    parent_page_id: Option<&str>,
) -> Result<Option<CloudResource>, String> {
    let data = serde_json::to_string(&serde_json::json!({
        "query": title,
        "filter": {"property": "object", "value": "page"},
        "page_size": 100
    }))
    .map_err(|error| error.to_string())?;
    let value = command_json(binary, &["api", "v1/search", "-d", &data]).await?;
    let result = value
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|page| {
            notion_page_title(page).as_deref() == Some(title)
                && page.get("in_trash").and_then(Value::as_bool) != Some(true)
                && page.get("is_archived").and_then(Value::as_bool) != Some(true)
                && match parent_page_id {
                    Some(parent) => {
                        page.get("parent")
                            .and_then(|value| value.get("page_id"))
                            .and_then(Value::as_str)
                            == Some(parent)
                    }
                    None => {
                        page.get("parent")
                            .and_then(|value| value.get("type"))
                            .and_then(Value::as_str)
                            == Some("workspace")
                    }
                }
        })
        // Notion search order is not stable. Choosing the oldest exact match
        // makes recovery deterministic if a previous process stopped after a
        // successful remote create but before persisting its local index.
        .min_by_key(|page| {
            page.get("created_time")
                .and_then(Value::as_str)
                .unwrap_or("9999-12-31T23:59:59.999Z")
        });
    Ok(result.and_then(notion_resource))
}

async fn create_notion_page(
    binary: &Path,
    title: &str,
    content: &str,
    parent_page_id: Option<&str>,
) -> Result<CloudResource, String> {
    let mut args = vec!["pages", "create", "--content", content, "--json"];
    let parent;
    if let Some(parent_page_id) = parent_page_id {
        parent = format!("page:{}", parent_page_id);
        args.push("--parent");
        args.push(&parent);
    }
    let value = command_json(binary, &args).await?;
    notion_resource(&value).ok_or_else(|| format!("Notion 创建页面后未返回页面 ID: {}", title))
}

async fn ensure_github_repository(github: &ServiceConnection) -> Result<CloudResource, String> {
    let binary = github
        .cli_path
        .as_deref()
        .map(PathBuf::from)
        .ok_or_else(|| "gh CLI 路径缺失".to_string())?;
    let owner = github
        .account
        .as_deref()
        .ok_or_else(|| "GitHub 活动账号缺失".to_string())?;
    let repository = format!("{}/composable-runtime-capabilities", owner);
    let fields = "nameWithOwner,isPrivate,url,description";
    let value = match command_json(&binary, &["repo", "view", &repository, "--json", fields]).await
    {
        Ok(value) => value,
        Err(_) => {
            command_text(
                &binary,
                &[
                    "repo",
                    "create",
                    &repository,
                    "--private",
                    "--add-readme",
                    "--disable-wiki",
                    "--description",
                    "Versioned capability packages for composable-runtime",
                ],
            )
            .await?;
            command_json(&binary, &["repo", "view", &repository, "--json", fields]).await?
        }
    };
    Ok(CloudResource {
        id: value
            .get("nameWithOwner")
            .and_then(Value::as_str)
            .unwrap_or(&repository)
            .to_string(),
        name: repository,
        url: value.get("url").and_then(Value::as_str).map(str::to_string),
        kind: "github_capability_repository".into(),
        private: value.get("isPrivate").and_then(Value::as_bool),
    })
}

const NOTION_ROOT_VARIABLE: &str = "COMPOSABLE_RUNTIME_NOTION_ROOT_ID";

async fn find_anchored_notion_root(
    notion_binary: &Path,
    github: &ServiceConnection,
    repository: &CloudResource,
) -> Option<CloudResource> {
    let github_binary = github.cli_path.as_deref().map(PathBuf::from)?;
    let root_id = command_text(
        &github_binary,
        &[
            "variable",
            "get",
            NOTION_ROOT_VARIABLE,
            "--repo",
            &repository.id,
        ],
    )
    .await
    .ok()?;
    let value = command_json(notion_binary, &["pages", "get", root_id.trim(), "--json"])
        .await
        .ok()?;
    if !notion_page_is_active(&Ok(value.clone())) {
        return None;
    }
    notion_resource(value.get("page").unwrap_or(&value))
}

async fn persist_notion_root_anchor(
    github: &ServiceConnection,
    repository: &CloudResource,
    root_id: &str,
) -> Result<(), String> {
    let binary = github
        .cli_path
        .as_deref()
        .map(PathBuf::from)
        .ok_or_else(|| "gh CLI 路径缺失".to_string())?;
    command_text(
        &binary,
        &[
            "variable",
            "set",
            NOTION_ROOT_VARIABLE,
            "--repo",
            &repository.id,
            "--body",
            root_id,
        ],
    )
    .await
    .map(|_| ())
}

fn notion_root_content() -> &'static str {
    "# Composable Runtime Agent Memory\n\n跨设备共享的长期知识入口。运行任务和敏感上下文仍保留在本地。\n\n## Knowledge Areas\n\n- User Memory\n- Projects\n- Capabilities\n- Learnings\n- Decisions\n\n## Storage Boundary\n\nNotion 保存提炼后的知识；GitHub 保存可执行能力包；SQLite 保存本地任务事件。"
}

fn notion_page_title(page: &Value) -> Option<String> {
    page.get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| {
            properties.values().find_map(|property| {
                (property.get("type").and_then(Value::as_str) == Some("title")).then(|| {
                    property
                        .get("title")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|item| item.get("plain_text").and_then(Value::as_str))
                        .collect::<String>()
                })
            })
        })
}

fn notion_resource(page: &Value) -> Option<CloudResource> {
    let id = page.get("id").and_then(Value::as_str)?.to_string();
    Some(CloudResource {
        name: notion_page_title(page).unwrap_or_else(|| "Notion Page".into()),
        id,
        url: page.get("url").and_then(Value::as_str).map(str::to_string),
        kind: "notion_page".into(),
        private: None,
    })
}

fn notion_page_is_active(result: &Result<Value, String>) -> bool {
    result.as_ref().is_ok_and(|page| {
        let page = page.get("page").unwrap_or(page);
        page.get("in_trash").and_then(Value::as_bool) != Some(true)
            && page.get("is_archived").and_then(Value::as_bool) != Some(true)
    })
}

pub(crate) fn persist_cloud_resources(
    storage_dir: &Path,
    state: &CloudResourceState,
) -> Result<(), String> {
    let path = storage_dir.join("cloud_resources.json");
    let temporary = path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(state).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    std::fs::rename(temporary, path).map_err(|error| error.to_string())
}

async fn detect_notion() -> ServiceConnection {
    let Some(binary) = find_binary("ntn") else {
        return unavailable("notion", "knowledge", "未安装 ntn CLI");
    };
    let version = command_text(&binary, &["--version"])
        .await
        .ok()
        .map(|value| value.trim().to_string());
    match command_json(&binary, &["whoami", "--json"]).await {
        Ok(value) => {
            let bot = value.get("bot").unwrap_or(&Value::Null);
            let owner = bot
                .get("owner")
                .and_then(|owner| owner.get("user"))
                .unwrap_or(&Value::Null);
            let account = owner
                .get("person")
                .and_then(|person| person.get("email"))
                .and_then(Value::as_str)
                .map(str::to_string);
            ServiceConnection {
                id: "notion".into(),
                role: "knowledge".into(),
                status: "connected".into(),
                installed: true,
                authenticated: true,
                healthy: true,
                cli_path: Some(binary.display().to_string()),
                version,
                account,
                account_name: owner
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                workspace: bot
                    .get("workspace_name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                workspace_id: bot
                    .get("workspace_id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                scopes: vec!["public_api".into()],
                warnings: Vec::new(),
                error: None,
            }
        }
        Err(error) => ServiceConnection {
            id: "notion".into(),
            role: "knowledge".into(),
            status: "not_authenticated".into(),
            installed: true,
            authenticated: false,
            healthy: false,
            cli_path: Some(binary.display().to_string()),
            version,
            account: None,
            account_name: None,
            workspace: None,
            workspace_id: None,
            scopes: Vec::new(),
            warnings: Vec::new(),
            error: Some(redact_error(&error)),
        },
    }
}

async fn detect_github() -> ServiceConnection {
    let Some(binary) = find_binary("gh") else {
        return unavailable("github", "capability_registry", "未安装 GitHub CLI");
    };
    let version = command_text(&binary, &["--version"])
        .await
        .ok()
        .and_then(|value| value.lines().next().map(str::to_string));
    match command_json(&binary, &["auth", "status", "--json", "hosts"]).await {
        Ok(value) => {
            let active = value
                .get("hosts")
                .and_then(|hosts| hosts.get("github.com"))
                .and_then(Value::as_array)
                .and_then(|accounts| {
                    accounts
                        .iter()
                        .find(|account| {
                            account.get("active").and_then(Value::as_bool) == Some(true)
                        })
                        .or_else(|| accounts.first())
                });
            let Some(active) = active else {
                return ServiceConnection {
                    id: "github".into(),
                    role: "capability_registry".into(),
                    status: "not_authenticated".into(),
                    installed: true,
                    authenticated: false,
                    healthy: false,
                    cli_path: Some(binary.display().to_string()),
                    version,
                    account: None,
                    account_name: None,
                    workspace: None,
                    workspace_id: None,
                    scopes: Vec::new(),
                    warnings: Vec::new(),
                    error: Some("GitHub CLI 没有活动账号".into()),
                };
            };
            let state = active
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let scopes = active
                .get("scopes")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|scope| !scope.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            let mut warnings = Vec::new();
            if !scopes.iter().any(|scope| scope == "repo") {
                warnings.push("缺少 repo 权限，无法同步私有能力仓库".into());
            }
            ServiceConnection {
                id: "github".into(),
                role: "capability_registry".into(),
                status: if state == "success" {
                    "connected"
                } else {
                    "error"
                }
                .into(),
                installed: true,
                authenticated: state == "success",
                healthy: state == "success",
                cli_path: Some(binary.display().to_string()),
                version,
                account: active
                    .get("login")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                account_name: None,
                workspace: Some("github.com".into()),
                workspace_id: None,
                scopes,
                warnings,
                error: (state != "success").then(|| "GitHub 认证状态异常".into()),
            }
        }
        Err(error) => ServiceConnection {
            id: "github".into(),
            role: "capability_registry".into(),
            status: "not_authenticated".into(),
            installed: true,
            authenticated: false,
            healthy: false,
            cli_path: Some(binary.display().to_string()),
            version,
            account: None,
            account_name: None,
            workspace: Some("github.com".into()),
            workspace_id: None,
            scopes: Vec::new(),
            warnings: Vec::new(),
            error: Some(redact_error(&error)),
        },
    }
}

async fn detect_git() -> ServiceConnection {
    let Some(binary) = find_binary("git") else {
        return unavailable("git", "local_version_control", "未安装 Git");
    };
    let version = command_text(&binary, &["--version"])
        .await
        .ok()
        .map(|value| value.trim().to_string());
    ServiceConnection {
        id: "git".into(),
        role: "local_version_control".into(),
        status: "available".into(),
        installed: true,
        authenticated: true,
        healthy: version.is_some(),
        cli_path: Some(binary.display().to_string()),
        version,
        account: None,
        account_name: None,
        workspace: None,
        workspace_id: None,
        scopes: Vec::new(),
        warnings: Vec::new(),
        error: None,
    }
}

fn unavailable(id: &str, role: &str, error: &str) -> ServiceConnection {
    ServiceConnection {
        id: id.into(),
        role: role.into(),
        status: "not_installed".into(),
        installed: false,
        authenticated: false,
        healthy: false,
        cli_path: None,
        version: None,
        account: None,
        account_name: None,
        workspace: None,
        workspace_id: None,
        scopes: Vec::new(),
        warnings: Vec::new(),
        error: Some(error.into()),
    }
}

async fn command_json(binary: &Path, args: &[&str]) -> Result<Value, String> {
    let output = command_output(binary, args).await?;
    serde_json::from_slice(&output).map_err(|error| format!("CLI JSON 解析失败: {}", error))
}

async fn command_text(binary: &Path, args: &[&str]) -> Result<String, String> {
    let output = command_output(binary, args).await?;
    String::from_utf8(output).map_err(|error| error.to_string())
}

async fn command_output(binary: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let mut child = Command::new(binary)
        .args(args)
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| error.to_string())?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let read_stdout = async move {
        let mut bytes = Vec::new();
        if let Some(mut stdout) = stdout {
            stdout
                .read_to_end(&mut bytes)
                .await
                .map_err(|error| error.to_string())?;
        }
        Ok::<_, String>(bytes)
    };
    let read_stderr = async move {
        let mut bytes = Vec::new();
        if let Some(mut stderr) = stderr {
            stderr
                .read_to_end(&mut bytes)
                .await
                .map_err(|error| error.to_string())?;
        }
        Ok::<_, String>(bytes)
    };
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (status, stdout, stderr) = tokio::join!(child.wait(), read_stdout, read_stderr);
        let status = status.map_err(|error| error.to_string())?;
        let stdout = stdout?;
        let stderr = stderr?;
        if status.success() {
            Ok(stdout)
        } else {
            Err(String::from_utf8_lossy(&stderr).trim().to_string())
        }
    })
    .await
    .map_err(|_| "CLI 状态检查超时".to_string())?;
    result
}

struct BootstrapLock {
    path: PathBuf,
}

impl Drop for BootstrapLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn acquire_bootstrap_lock(storage_dir: &Path) -> Result<BootstrapLock, String> {
    let path = storage_dir.join("cloud_bootstrap.lock");
    for _ in 0..120 {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => return Ok(BootstrapLock { path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let stale = std::fs::metadata(&path)
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
                    .and_then(|modified| modified.elapsed().ok())
                    .is_some_and(|elapsed| elapsed > std::time::Duration::from_secs(300));
                if stale {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            Err(error) => return Err(format!("创建云端初始化锁失败: {}", error)),
        }
    }
    Err("云端资源正在由另一个任务初始化".into())
}

fn find_binary(name: &str) -> Option<PathBuf> {
    let mut candidates = std::env::var_os("PATH")
        .map(|path| {
            std::env::split_paths(&path)
                .map(|dir| dir.join(name))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".local/bin").join(name));
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin").join(name));
    candidates.push(PathBuf::from("/usr/local/bin").join(name));
    candidates.push(PathBuf::from("/usr/bin").join(name));
    candidates.into_iter().find(|path| path.is_file())
}

fn redact_error(error: &str) -> String {
    let compact = error.lines().take(4).collect::<Vec<_>>().join(" ");
    compact
        .split_whitespace()
        .map(|part| {
            if part.starts_with("gho_")
                || part.starts_with("ghp_")
                || part.starts_with("secret_")
                || part.starts_with("ntn_")
            {
                "[redacted]"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(500)
        .collect()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_redact_common_tokens() {
        assert!(!redact_error("failed gho_abc123 secret_xyz").contains("abc123"));
    }

    #[test]
    fn binary_lookup_accepts_absolute_common_locations() {
        assert!(find_binary("git").is_some());
    }

    #[test]
    fn page_activity_reads_ntn_get_wrapper() {
        let active = Ok(serde_json::json!({
            "page": {"in_trash": false, "is_archived": false}
        }));
        let trashed = Ok(serde_json::json!({
            "page": {"in_trash": true, "is_archived": false}
        }));
        assert!(notion_page_is_active(&active));
        assert!(!notion_page_is_active(&trashed));
    }
}
