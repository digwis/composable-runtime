//! Workspace observer and graph snapshot.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const DIRECTORY_SCAN_LIMIT: usize = 1_000;
const RECENT_ITEM_LIMIT: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceGraph {
    pub generated_at: u64,
    pub roots: Vec<String>,
    pub projects: Vec<WorkspaceProjectNode>,
    pub totals: WorkspaceTotals,
    #[serde(default)]
    pub sources: Vec<WorkspaceSourceNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSourceNode {
    pub id: String,
    pub label: String,
    pub status: String,
    pub item_count: usize,
    #[serde(default = "default_true")]
    pub item_count_exact: bool,
    #[serde(default)]
    pub scan_limit_reached: bool,
    #[serde(default)]
    pub items: Vec<WorkspaceSourceItem>,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSourceItem {
    pub kind: String,
    pub title: String,
    pub location: String,
    #[serde(default)]
    pub modified_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceProjectNode {
    pub name: String,
    pub path: String,
    pub kind: Vec<String>,
    pub branch: String,
    pub head: String,
    pub dirty: bool,
    pub changed_files: usize,
    pub todo_count: usize,
    pub branches: usize,
    pub remotes: Vec<String>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceTotals {
    pub projects: usize,
    pub changed_files: usize,
    pub todos: usize,
    pub branches: usize,
    pub remotes: usize,
    #[serde(default)]
    pub sources: usize,
    #[serde(default)]
    pub available_sources: usize,
    #[serde(default)]
    pub source_items: usize,
}

pub async fn observe(roots: &[PathBuf], storage_dir: &Path) -> WorkspaceGraph {
    let mut projects = Vec::new();
    for root in roots {
        let candidates = if root.join(".git").exists() {
            vec![root.clone()]
        } else {
            std::fs::read_dir(root)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path())
                .collect()
        };
        for path in candidates {
            if !path.is_dir() || !path.join(".git").exists() {
                continue;
            }
            if let Some(node) = inspect_project(&path).await {
                projects.push(node);
            }
        }
    }
    projects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    let mut totals = projects
        .iter()
        .fold(WorkspaceTotals::default(), |mut total, project| {
            total.projects += 1;
            total.changed_files += project.changed_files;
            total.todos += project.todo_count;
            total.branches += project.branches;
            total.remotes += project.remotes.len();
            total
        });
    let sources = observe_local_sources();
    totals.sources = sources.len();
    totals.available_sources = sources
        .iter()
        .filter(|source| matches!(source.status.as_str(), "available" | "partial"))
        .count();
    totals.source_items = sources.iter().map(|source| source.item_count).sum();
    let graph = WorkspaceGraph {
        generated_at: unix_now(),
        roots: roots
            .iter()
            .map(|root| root.display().to_string())
            .collect(),
        projects,
        totals,
        sources,
    };
    let path = storage_dir.join("workspace_graph.json");
    let _ = std::fs::write(path, serde_json::to_vec_pretty(&graph).unwrap_or_default());
    graph
}

fn observe_local_sources() -> Vec<WorkspaceSourceNode> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Vec::new();
    };
    [
        ("downloads", "Downloads", home.join("Downloads")),
        ("desktop", "Desktop", home.join("Desktop")),
        ("documents", "Documents", home.join("Documents")),
    ]
    .into_iter()
    .map(|(id, label, path)| inspect_directory_source(id, label, &path))
    .collect()
}

fn inspect_directory_source(id: &str, label: &str, path: &Path) -> WorkspaceSourceNode {
    inspect_directory_source_with_limits(id, label, path, DIRECTORY_SCAN_LIMIT, RECENT_ITEM_LIMIT)
}

fn inspect_directory_source_with_limits(
    id: &str,
    label: &str,
    path: &Path,
    scan_limit: usize,
    recent_limit: usize,
) -> WorkspaceSourceNode {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) => {
            return WorkspaceSourceNode {
                id: id.into(),
                label: label.into(),
                status: directory_error_status(&error).into(),
                item_count: 0,
                item_count_exact: true,
                scan_limit_reached: false,
                items: Vec::new(),
                evidence: vec![format!("无法读取 {}", path.display())],
                error: Some(error.to_string()),
            };
        }
    };
    let mut item_count = 0usize;
    let mut items = Vec::new();
    let mut entry_errors = Vec::new();
    let mut scan_limit_reached = false;
    for (attempt, entry) in entries.enumerate() {
        if attempt >= scan_limit {
            scan_limit_reached = true;
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                if entry_errors.len() < 10 {
                    entry_errors.push(error.to_string());
                }
                continue;
            }
        };
        item_count += 1;
        let path = entry.path();
        let file_type = entry.file_type().ok();
        let metadata = entry.metadata().ok();
        let modified_at = metadata
            .as_ref()
            .and_then(|metadata| metadata.modified().ok())
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs());
        items.push(WorkspaceSourceItem {
            kind: if file_type.as_ref().is_some_and(|kind| kind.is_symlink()) {
                "symlink".into()
            } else if file_type.as_ref().is_some_and(|kind| kind.is_dir()) {
                "directory".into()
            } else {
                "file".into()
            },
            title: entry.file_name().to_string_lossy().to_string(),
            location: path.display().to_string(),
            modified_at,
        });
    }
    items.sort_by(|left, right| {
        right
            .modified_at
            .cmp(&left.modified_at)
            .then_with(|| left.title.cmp(&right.title))
    });
    items.truncate(recent_limit);
    let partial = scan_limit_reached || !entry_errors.is_empty();
    let mut evidence = vec![format!("已完成 {} 的浅层元数据观察", path.display())];
    if scan_limit_reached {
        evidence.push(format!("扫描达到 {} 项预算，结果已截断", scan_limit));
    }
    if !entry_errors.is_empty() {
        evidence.push(format!("有 {} 个目录项读取失败", entry_errors.len()));
    }
    WorkspaceSourceNode {
        id: id.into(),
        label: label.into(),
        status: if partial { "partial" } else { "available" }.into(),
        item_count,
        item_count_exact: !partial,
        scan_limit_reached,
        items,
        evidence,
        error: (!entry_errors.is_empty()).then(|| entry_errors.join("; ")),
    }
}

fn directory_error_status(error: &std::io::Error) -> &'static str {
    match error.kind() {
        std::io::ErrorKind::PermissionDenied => "permission_required",
        std::io::ErrorKind::NotFound => "unavailable",
        _ => "error",
    }
}

fn default_true() -> bool {
    true
}

pub fn load(storage_dir: &Path) -> Option<WorkspaceGraph> {
    std::fs::read_to_string(storage_dir.join("workspace_graph.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
}

async fn inspect_project(path: &Path) -> Option<WorkspaceProjectNode> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let branch = git(path, &["branch", "--show-current"]).await?;
    let head = git(path, &["log", "-1", "--format=%h %s"]).await?;
    let status = git(path, &["status", "--short"]).await?;
    let branches = git(path, &["branch", "--list"])
        .await
        .map(|s| s.lines().count())
        .unwrap_or_default();
    let remotes = git(path, &["remote", "-v"])
        .await
        .map(|s| {
            s.lines()
                .filter_map(|line| line.split_whitespace().nth(1).map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut kind = Vec::new();
    for (label, marker) in [
        ("rust", "Cargo.toml"),
        ("node", "package.json"),
        ("python", "pyproject.toml"),
        ("cmake", "CMakeLists.txt"),
    ] {
        if path.join(marker).exists() {
            kind.push(label.into());
        }
    }
    let todo_count = Command::new("rg")
        .args(["-n", "--hidden", "-g", "!.git", "TODO|FIXME|HACK"])
        .current_dir(path)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).lines().count())
        .unwrap_or_default();
    let mut evidence = Vec::new();
    if !status.trim().is_empty() {
        evidence.push(format!("工作区有 {} 个变更文件", status.lines().count()));
    }
    if todo_count > 0 {
        evidence.push(format!("发现 {} 个 TODO/FIXME/HACK 标记", todo_count));
    }
    Some(WorkspaceProjectNode {
        name,
        path: path.display().to_string(),
        kind,
        branch: branch.trim().into(),
        head: head.trim().into(),
        dirty: !status.trim().is_empty(),
        changed_files: status.lines().count(),
        todo_count,
        branches,
        remotes,
        evidence,
    })
}

async fn git(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_source_reports_recent_metadata_without_reading_contents() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        std::fs::write(dir.path().join("report.txt"), "secret-content").expect("写入测试文件");
        std::fs::create_dir(dir.path().join("project-folder")).expect("创建测试子目录");

        let source = inspect_directory_source("fixture", "Fixture", dir.path());
        assert_eq!(source.status, "available");
        assert_eq!(source.item_count, 2);
        assert!(source.item_count_exact);
        assert!(!source.scan_limit_reached);
        assert!(source.items.iter().any(|item| item.title == "report.txt"));
        assert!(source.items.iter().any(|item| item.kind == "directory"));
        assert!(!serde_json::to_string(&source)
            .unwrap()
            .contains("secret-content"));
    }

    #[test]
    fn directory_source_bounds_scan_and_recent_items() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        for index in 0..12 {
            std::fs::write(
                dir.path().join(format!("item-{index:02}.txt")),
                index.to_string(),
            )
            .expect("写入测试文件");
        }

        let source = inspect_directory_source_with_limits("fixture", "Fixture", dir.path(), 10, 3);
        assert_eq!(source.status, "partial");
        assert_eq!(source.item_count, 10);
        assert!(!source.item_count_exact);
        assert!(source.scan_limit_reached);
        assert_eq!(source.items.len(), 3);
        assert!(source
            .evidence
            .iter()
            .any(|item| item.contains("10 项预算")));
    }

    #[test]
    fn directory_source_reports_missing_directory() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        let missing = dir.path().join("missing");
        let source = inspect_directory_source("fixture", "Fixture", &missing);
        assert_eq!(source.status, "unavailable");
        assert_eq!(source.item_count, 0);
        assert!(source.error.is_some());
    }

    #[test]
    fn directory_error_status_preserves_error_class() {
        assert_eq!(
            directory_error_status(&std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
            "permission_required"
        );
        assert_eq!(
            directory_error_status(&std::io::Error::from(std::io::ErrorKind::NotFound)),
            "unavailable"
        );
        assert_eq!(
            directory_error_status(&std::io::Error::from(std::io::ErrorKind::Other)),
            "error"
        );
    }

    #[test]
    fn legacy_workspace_graph_defaults_new_source_fields() {
        let graph: WorkspaceGraph = serde_json::from_value(serde_json::json!({
            "generated_at": 1,
            "roots": [],
            "projects": [],
            "totals": {
                "projects": 0,
                "changed_files": 0,
                "todos": 0,
                "branches": 0,
                "remotes": 0
            }
        }))
        .expect("旧 Workspace Graph 应兼容");
        assert!(graph.sources.is_empty());
        assert_eq!(graph.totals.sources, 0);
        assert_eq!(graph.totals.source_items, 0);
    }
}
