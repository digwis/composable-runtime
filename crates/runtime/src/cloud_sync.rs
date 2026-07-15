//! Personal cloud synchronization.
//!
//! GitHub receives executable capability definitions only. Notion receives
//! bounded, human-readable summaries of personal knowledge. Raw task events,
//! credentials, thought chains, and project files remain local.

use crate::integrations::{
    detect_integrations, ensure_cloud_resources, persist_cloud_resources, CloudResource,
    CloudResourceState, ServiceConnection,
};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

pub async fn sync_personal_cloud(storage_dir: &Path) -> Result<CloudResourceState, String> {
    let mut state = ensure_cloud_resources(storage_dir).await?;
    let integrations = detect_integrations().await;
    let mut sync_errors = Vec::new();

    if integrations.github.healthy {
        match state.github_capability_repository.as_ref() {
            Some(repository) => {
                match sync_capabilities_to_github(storage_dir, &integrations.github, repository)
                    .await
                {
                    Ok(changed) => {
                        state.last_github_sync_at = Some(unix_now());
                        state.github_sync_changed = changed;
                    }
                    Err(error) => sync_errors.push(format!("GitHub 能力同步失败: {}", error)),
                }
            }
            None => sync_errors.push("GitHub 私有能力仓库尚未初始化".into()),
        }
    }

    if integrations.notion.healthy {
        match sync_knowledge_to_notion(storage_dir, &integrations.notion, &state).await {
            Ok(()) => state.last_notion_sync_at = Some(unix_now()),
            Err(error) => sync_errors.push(format!("Notion 知识同步失败: {}", error)),
        }
    }

    state.last_sync_errors = sync_errors.clone();
    for error in sync_errors {
        if !state.warnings.contains(&error) {
            state.warnings.push(error);
        }
    }
    persist_cloud_resources(storage_dir, &state)?;
    Ok(state)
}

async fn sync_capabilities_to_github(
    storage_dir: &Path,
    github: &ServiceConnection,
    repository: &CloudResource,
) -> Result<bool, String> {
    let gh = github
        .cli_path
        .as_deref()
        .map(PathBuf::from)
        .ok_or_else(|| "gh CLI 路径缺失".to_string())?;
    let git = find_binary("git").ok_or_else(|| "git CLI 路径缺失".to_string())?;
    let cloud_dir = storage_dir.join("cloud");
    let checkout = cloud_dir.join("capabilities");
    std::fs::create_dir_all(&cloud_dir).map_err(|error| error.to_string())?;

    if !checkout.join(".git").is_dir() {
        if checkout.exists() {
            std::fs::remove_dir_all(&checkout).map_err(|error| error.to_string())?;
        }
        command(
            &gh,
            &[
                "repo",
                "clone",
                &repository.id,
                &checkout.display().to_string(),
            ],
            None,
        )
        .await?;
    } else {
        command(
            &git,
            &["pull", "--ff-only", "origin", "main"],
            Some(&checkout),
        )
        .await?;
    }

    let source = storage_dir.join(".evolution");
    let target = checkout.join("capabilities");
    write_capability_snapshot(&source, &target)?;
    command(
        &git,
        &["config", "user.name", "composable-runtime"],
        Some(&checkout),
    )
    .await?;
    command(
        &git,
        &["config", "user.email", "runtime@localhost"],
        Some(&checkout),
    )
    .await?;
    command(
        &git,
        &["add", "--all", "--", "capabilities"],
        Some(&checkout),
    )
    .await?;
    let status = command(
        &git,
        &["status", "--porcelain", "--", "capabilities"],
        Some(&checkout),
    )
    .await?;
    if status.trim().is_empty() {
        return Ok(false);
    }
    command(
        &git,
        &["commit", "-m", "chore: sync personal capabilities"],
        Some(&checkout),
    )
    .await?;
    command(&git, &["push", "origin", "HEAD:main"], Some(&checkout)).await?;
    Ok(true)
}

fn write_capability_snapshot(source: &Path, target: &Path) -> Result<(), String> {
    if !source.is_dir() {
        return Err(format!("能力存储目录不存在: {}", source.display()));
    }
    let temporary = target.with_extension("tmp");
    if temporary.exists() {
        std::fs::remove_dir_all(&temporary).map_err(|error| error.to_string())?;
    }
    std::fs::create_dir_all(&temporary).map_err(|error| error.to_string())?;
    for entry in std::fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() && (name == "shared" || path.join("genome.yaml").is_file()) {
            copy_tree(&path, &temporary.join(&name))?;
        } else if path.is_file() && name == "manifest.yaml" {
            std::fs::copy(&path, temporary.join(&name)).map_err(|error| error.to_string())?;
        }
    }
    if target.exists() {
        std::fs::remove_dir_all(target).map_err(|error| error.to_string())?;
    }
    std::fs::rename(temporary, target).map_err(|error| error.to_string())
}

fn copy_tree(source: &Path, target: &Path) -> Result<(), String> {
    std::fs::create_dir_all(target).map_err(|error| error.to_string())?;
    for entry in std::fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &target_path)?;
        } else if source_path.is_file() {
            std::fs::copy(source_path, target_path).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

async fn sync_knowledge_to_notion(
    storage_dir: &Path,
    notion: &ServiceConnection,
    state: &CloudResourceState,
) -> Result<(), String> {
    let binary = notion
        .cli_path
        .as_deref()
        .map(PathBuf::from)
        .ok_or_else(|| "ntn CLI 路径缺失".to_string())?;
    let pages = [
        ("User Memory", render_user_memory(storage_dir)),
        ("Projects", render_projects(storage_dir)),
        ("Capabilities", render_capabilities(storage_dir)),
        ("Learnings", render_learnings(storage_dir)),
        ("Decisions", render_decisions(storage_dir)),
    ];
    for (title, content) in pages {
        let page = state
            .notion_children
            .iter()
            .find(|page| page.name == title)
            .ok_or_else(|| format!("Notion 页面尚未初始化: {}", title))?;
        command(
            &binary,
            &["pages", "update", &page.id, "--content", &content, "--json"],
            None,
        )
        .await?;
    }
    Ok(())
}

fn render_user_memory(storage_dir: &Path) -> String {
    let value = read_json(&storage_dir.join("memory.json"));
    let mut out = page_header("User Memory", "个人长期工作流、偏好与使用统计的提炼快照。");
    let Some(value) = value else {
        out.push_str("\n当前尚无长期记忆。\n");
        return out;
    };
    if let Some(stats) = value.get("stats") {
        out.push_str("\n## Statistics\n\n");
        for key in [
            "total_sessions",
            "total_tasks",
            "total_successes",
            "total_failures",
            "total_evolution_events",
            "total_capabilities_created",
        ] {
            if let Some(number) = stats.get(key).and_then(Value::as_u64) {
                out.push_str(&format!("- {}: {}\n", key, number));
            }
        }
    }
    out.push_str("\n## Successful Workflows\n\n");
    let workflows = value.get("workflow_templates").and_then(Value::as_array);
    append_object_summaries(
        &mut out,
        workflows,
        40,
        &["task", "success_count", "fitness", "last_used"],
    );
    out
}

fn render_projects(storage_dir: &Path) -> String {
    let mut out = page_header("Projects", "个人项目愿景、优先级和已验证进展。");
    let mut projects = Vec::new();
    collect_project_memories(storage_dir, storage_dir, 0, &mut projects);
    if projects.is_empty() {
        out.push_str("\n当前尚无项目记忆。\n");
        return out;
    }
    for project in projects.into_iter().take(50) {
        let name = project
            .get("project_path")
            .and_then(Value::as_str)
            .unwrap_or("Project");
        out.push_str(&format!("\n## {}\n\n", markdown_text(name)));
        if let Some(vision) = project.get("vision").and_then(Value::as_str) {
            if !vision.trim().is_empty() {
                out.push_str(&format!("{}\n\n", markdown_text(vision)));
            }
        }
        if let Some(priorities) = project.get("priorities").and_then(Value::as_array) {
            for priority in priorities.iter().filter_map(Value::as_str).take(20) {
                out.push_str(&format!("- Priority: {}\n", markdown_text(priority)));
            }
        }
        for (label, field) in [
            ("Completed", "completed_goals"),
            ("Rejected", "rejected_goals"),
            ("Feedback", "feedback"),
        ] {
            let count = project
                .get(field)
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            out.push_str(&format!("- {}: {}\n", label, count));
        }
    }
    out
}

fn render_capabilities(storage_dir: &Path) -> String {
    let mut out = page_header(
        "Capabilities",
        "当前个人能力种群及其真实适应度摘要。可执行源码保存在私有 GitHub 仓库。",
    );
    let path = storage_dir.join(".evolution/manifest.yaml");
    let manifest = std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_yaml::from_str::<crate::genome_yaml::Manifest>(&raw).ok());
    let Some(mut manifest) = manifest else {
        out.push_str("\n当前尚无能力清单。\n");
        return out;
    };
    manifest
        .capabilities
        .sort_by(|left, right| right.score.total_cmp(&left.score));
    out.push_str(&format!("\n能力总数：{}\n\n", manifest.total_capabilities));
    out.push_str("| Capability | Version | Score | Success | Calls | Actions |\n|---|---:|---:|---:|---:|---:|\n");
    for capability in manifest.capabilities.into_iter().take(200) {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.1}% | {} | {} |\n",
            markdown_cell(&capability.name),
            markdown_cell(&capability.version),
            capability.score,
            capability.success_rate * 100.0,
            capability.call_count,
            capability.action_count
        ));
    }
    out
}

fn render_learnings(storage_dir: &Path) -> String {
    let mut out = page_header("Learnings", "从真实反馈、测试和变异中提炼的近期教训。");
    let value = read_json(&storage_dir.join(".evolution/evolution_memory.json"));
    out.push_str("\n## Recent Lessons\n\n");
    append_object_summaries(
        &mut out,
        value
            .as_ref()
            .and_then(|value| value.get("lessons"))
            .and_then(Value::as_array),
        100,
        &[
            "lesson",
            "capability",
            "failure_type",
            "learned_at",
            "referenced_count",
        ],
    );
    out
}

fn render_decisions(storage_dir: &Path) -> String {
    let mut out = page_header("Decisions", "自主控制回路近期作出的、可审计的决策摘要。");
    let value = read_json(&storage_dir.join("autonomy/state.json"));
    out.push_str("\n## Recent Decisions\n\n");
    append_object_summaries(
        &mut out,
        value
            .as_ref()
            .and_then(|value| value.get("decisions"))
            .and_then(Value::as_array),
        80,
        &[
            "decision",
            "action",
            "reason",
            "project_path",
            "created_at",
            "timestamp",
        ],
    );
    out
}

fn page_header(title: &str, description: &str) -> String {
    format!(
        "# {}\n\n{}\n\n_Last synchronized: {}_\n",
        title,
        description,
        unix_now()
    )
}

fn append_object_summaries(
    out: &mut String,
    values: Option<&Vec<Value>>,
    limit: usize,
    fields: &[&str],
) {
    let Some(values) = values else {
        out.push_str("- 暂无记录\n");
        return;
    };
    if values.is_empty() {
        out.push_str("- 暂无记录\n");
    }
    for value in values.iter().rev().take(limit) {
        let details = fields
            .iter()
            .filter_map(|field| {
                value
                    .get(*field)
                    .map(|value| format!("{}={}", field, compact_value(value)))
            })
            .collect::<Vec<_>>();
        if !details.is_empty() {
            out.push_str(&format!("- {}\n", markdown_text(&details.join(" · "))));
        }
    }
}

fn compact_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.chars().take(500).collect(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => "null".into(),
        other => serde_json::to_string(other)
            .unwrap_or_default()
            .chars()
            .take(500)
            .collect(),
    }
}

fn collect_project_memories(root: &Path, path: &Path, depth: usize, output: &mut Vec<Value>) {
    if depth > 5 || output.len() >= 50 || path.ends_with(".evolution") || path.ends_with("cloud") {
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let child = entry.path();
        if child.is_dir() {
            collect_project_memories(root, &child, depth + 1, output);
        } else if child.extension().and_then(|value| value.to_str()) == Some("json")
            && child != root.join("memory.json")
        {
            if let Some(value) = read_json(&child) {
                if value.get("project_path").and_then(Value::as_str).is_some()
                    && (value.get("vision").is_some() || value.get("priorities").is_some())
                {
                    output.push(value);
                }
            }
        }
    }
}

fn read_json(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
}

fn markdown_text(value: &str) -> String {
    value
        .replace('\r', " ")
        .replace('\n', " ")
        .replace('|', "\\|")
}

fn markdown_cell(value: &str) -> String {
    markdown_text(value).chars().take(120).collect()
}

async fn command(binary: &Path, args: &[&str], cwd: Option<&Path>) -> Result<String, String> {
    let mut process = Command::new(binary);
    process.args(args).kill_on_drop(true);
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }
    let output = tokio::time::timeout(std::time::Duration::from_secs(180), process.output())
        .await
        .map_err(|_| format!("{} 命令超时", binary.display()))?
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(redact(&String::from_utf8_lossy(&output.stderr)));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
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
    candidates.extend([
        PathBuf::from("/opt/homebrew/bin").join(name),
        PathBuf::from("/usr/local/bin").join(name),
        PathBuf::from("/usr/bin").join(name),
    ]);
    candidates.into_iter().find(|path| path.is_file())
}

fn redact(value: &str) -> String {
    value
        .split_whitespace()
        .map(|part| {
            if part.starts_with("gho_")
                || part.starts_with("ghp_")
                || part.starts_with("ntn_")
                || part.starts_with("secret_")
            {
                "[redacted]"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(1000)
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
    fn capability_snapshot_excludes_private_evolution_memory() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        let target = root.path().join("target");
        std::fs::create_dir_all(source.join("cap/actions")).unwrap();
        std::fs::write(source.join("cap/genome.yaml"), "name: cap\n").unwrap();
        std::fs::write(source.join("cap/actions/run.py"), "print('{}')\n").unwrap();
        std::fs::write(source.join("manifest.yaml"), "total_capabilities: 1\n").unwrap();
        std::fs::write(source.join("evolution_memory.json"), "{\"secret\":true}").unwrap();
        write_capability_snapshot(&source, &target).unwrap();
        assert!(target.join("cap/actions/run.py").is_file());
        assert!(target.join("manifest.yaml").is_file());
        assert!(!target.join("evolution_memory.json").exists());
    }

    #[test]
    fn markdown_summary_is_bounded_and_single_line() {
        let value = format!("a\n|{}", "x".repeat(700));
        let rendered = markdown_text(&value);
        assert!(!rendered.contains('\n'));
        assert!(rendered.contains("\\|"));
    }
}
