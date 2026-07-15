//! Multi-variant exploration and isolated experiment execution.
//!
//! Exploration proposes alternatives; experiments execute each alternative in
//! its own temporary worktree. No experiment writes to the user's current
//! branch, and worktrees/branches are cleaned up after the result is recorded.

use crate::driver::EvolutionDriver;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerProposal {
    pub id: String,
    pub title: String,
    pub approach: String,
    pub task: String,
    pub expected_value: String,
    pub risk: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerResult {
    pub project_path: String,
    pub objective: String,
    pub proposals: Vec<ExplorerProposal>,
    pub generated_at: u64,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExperimentRequest {
    pub project_path: String,
    pub objective: String,
    pub variants: Vec<ExperimentVariant>,
    #[serde(default)]
    pub verify_command: Option<String>,
    #[serde(default)]
    pub benchmark_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentVariant {
    pub id: String,
    pub title: String,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    pub variant_id: String,
    pub title: String,
    pub status: String,
    pub branch: Option<String>,
    pub worktree: Option<String>,
    pub agent_output: String,
    pub verification: Option<ExperimentCommandResult>,
    pub benchmark: Option<ExperimentCommandResult>,
    pub diff_stat: String,
    pub elapsed_ms: u64,
    pub cleaned_up: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentBatch {
    pub batch_id: String,
    pub project_path: String,
    pub objective: String,
    pub status: String,
    pub variants: Vec<ExperimentVariant>,
    pub runs: Vec<ExperimentRun>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentCommandResult {
    pub command: String,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

pub struct ExperimentEngine {
    storage_dir: PathBuf,
    pi_binary: String,
    pi_model: String,
}

impl ExperimentEngine {
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            storage_dir: storage_dir.into(),
            pi_binary: std::env::var("ORCH_PI_BIN").unwrap_or_else(|_| "pi".into()),
            pi_model: std::env::var("ORCH_PI_MODEL")
                .unwrap_or_else(|_| "opengateway/tencent/hy3".into()),
        }
    }

    pub async fn explore(
        &self,
        project_path: &str,
        objective: &str,
        driver: Option<Arc<dyn EvolutionDriver>>,
        max_variants: usize,
    ) -> Result<ExplorerResult, String> {
        let project = canonical_project(project_path).await?;
        let max_variants = max_variants.clamp(2, 6);
        if let Some(driver) = driver.filter(|driver| driver.has_llm_backend()) {
            let prompt = format!(
                "你是软件架构探索者。针对真实 Git 项目目标，提出 {} 个互相独立、可在隔离 worktree 中验证的方案。不要修改文件。严格只输出 JSON 数组，每项字段 id,title,approach,task,expected_value,risk,confidence(0到1)。目标：{}\n项目：{}",
                max_variants, objective, project.display()
            );
            if let Ok(raw) = driver
                .execute(
                    &prompt,
                    "smart:explorer",
                    Some("只输出 JSON 数组，不要 markdown。方案必须可执行、可验证、不要凭空编造。"),
                )
                .await
            {
                let proposals = parse_explorer_output(&raw, max_variants);
                if proposals.len() >= 2 {
                    return Ok(ExplorerResult {
                        project_path: project.display().to_string(),
                        objective: objective.into(),
                        proposals,
                        generated_at: unix_now(),
                        source: "llm".into(),
                    });
                }
            }
        }
        Ok(ExplorerResult {
            project_path: project.display().to_string(),
            objective: objective.into(),
            proposals: fallback_proposals(objective, max_variants),
            generated_at: unix_now(),
            source: "policy_fallback".into(),
        })
    }

    pub async fn run_batch(
        &self,
        batch_id: &str,
        request: &ExperimentRequest,
    ) -> Result<ExperimentBatch, String> {
        let project = canonical_project(&request.project_path).await?;
        if request.variants.len() < 2 {
            return Err("至少需要两个实验方案".into());
        }
        let variants = request.variants.iter().take(6).cloned().collect::<Vec<_>>();
        let started = unix_now();
        let queued = ExperimentBatch {
            batch_id: batch_id.into(),
            project_path: project.display().to_string(),
            objective: request.objective.clone(),
            status: "queued".into(),
            variants: variants.clone(),
            runs: Vec::new(),
            created_at: started,
            completed_at: None,
        };
        persist_batch(&self.storage_dir, &queued)?;
        let running = ExperimentBatch {
            status: "running".into(),
            ..queued.clone()
        };
        persist_batch(&self.storage_dir, &running)?;
        let futures = variants.iter().cloned().map(|variant| {
            self.run_variant(
                batch_id,
                &project,
                &request.objective,
                variant,
                request.verify_command.as_deref(),
                request.benchmark_command.as_deref(),
            )
        });
        let runs = futures::future::join_all(futures).await;
        let batch = ExperimentBatch {
            batch_id: batch_id.into(),
            project_path: project.display().to_string(),
            objective: request.objective.clone(),
            status: "completed".into(),
            variants,
            runs,
            created_at: started,
            completed_at: Some(unix_now()),
        };
        persist_batch(&self.storage_dir, &batch)?;
        Ok(batch)
    }

    async fn run_variant(
        &self,
        batch_id: &str,
        project: &Path,
        objective: &str,
        variant: ExperimentVariant,
        verify: Option<&str>,
        benchmark: Option<&str>,
    ) -> ExperimentRun {
        let start = Instant::now();
        let safe_id = sanitize_id(&variant.id);
        let branch = format!("orch/experiment/{}-{}", sanitize_id(batch_id), safe_id);
        let worktree = self
            .storage_dir
            .join("experiments")
            .join(sanitize_id(batch_id))
            .join(&safe_id);
        let mut run = ExperimentRun {
            variant_id: variant.id.clone(),
            title: variant.title.clone(),
            status: "failed".into(),
            branch: Some(branch.clone()),
            worktree: Some(worktree.display().to_string()),
            agent_output: String::new(),
            verification: None,
            benchmark: None,
            diff_stat: String::new(),
            elapsed_ms: 0,
            cleaned_up: false,
            error: None,
        };
        if let Err(error) = std::fs::create_dir_all(worktree.parent().unwrap_or(&worktree))
            .map_err(|e| e.to_string())
            .and_then(|_| {
                git_sync(
                    project,
                    &[
                        "worktree",
                        "add",
                        "-b",
                        &branch,
                        worktree.to_string_lossy().as_ref(),
                        "HEAD",
                    ],
                )
            })
        {
            run.error = Some(error);
            run.elapsed_ms = start.elapsed().as_millis() as u64;
            return run;
        }
        let prompt = format!("你正在隔离实验 worktree 中。总目标：{}\n当前方案：{}\n任务：{}\n要求：只修改本方案需要的文件；不要 push、不要创建 PR；完成后报告修改、测试和风险。", objective, variant.title, variant.task);
        match tokio::time::timeout(
            std::time::Duration::from_secs(900),
            Command::new(&self.pi_binary)
                .args([
                    "--no-session",
                    "--approve",
                    "--mode",
                    "text",
                    "--model",
                    &self.pi_model,
                    "--tools",
                    "read,grep,find,ls,bash,edit,write",
                    "-p",
                    &prompt,
                ])
                .kill_on_drop(true)
                .current_dir(&worktree)
                .output(),
        )
        .await
        {
            Ok(Ok(output)) if output.status.success() => {
                run.agent_output = String::from_utf8_lossy(&output.stdout).to_string()
            }
            Ok(Ok(output)) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                let detail = if stderr.is_empty() {
                    "无错误输出".into()
                } else {
                    stderr
                };
                run.error = Some(format!(
                    "pi 执行失败（退出码 {:?}）：{}",
                    output.status.code(),
                    detail
                ));
            }
            Ok(Err(error)) => run.error = Some(format!("启动 pi 失败: {}", error)),
            Err(_) => run.error = Some("pi 执行超时 (900s)".into()),
        }
        if run.error.is_none() {
            run.diff_stat = git_sync(
                project,
                &["-C", worktree.to_string_lossy().as_ref(), "diff", "--stat"],
            )
            .unwrap_or_default();
            if let Some(command) = verify {
                run.verification = Some(run_command(&worktree, command).await);
            }
            if let Some(command) = benchmark {
                run.benchmark = Some(run_command(&worktree, command).await);
            }
            let verification_ok = run
                .verification
                .as_ref()
                .map(|result| result.success)
                .unwrap_or(true);
            let benchmark_ok = run
                .benchmark
                .as_ref()
                .map(|result| result.success)
                .unwrap_or(true);
            if verification_ok && benchmark_ok {
                run.status = "passed".into();
            }
        }
        let _ = git_sync(
            project,
            &[
                "worktree",
                "remove",
                "--force",
                worktree.to_string_lossy().as_ref(),
            ],
        );
        let _ = git_sync(project, &["branch", "-D", &branch]);
        run.cleaned_up = true;
        run.worktree = None;
        run.elapsed_ms = start.elapsed().as_millis() as u64;
        run
    }
}

pub fn load_batches(storage_dir: &Path) -> Vec<ExperimentBatch> {
    let dir = storage_dir.join("experiments");
    let mut batches = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|raw| serde_json::from_str(&raw).ok())
        .collect::<Vec<ExperimentBatch>>();
    batches.sort_by_key(|batch| batch.created_at);
    batches
}

fn persist_batch(storage_dir: &Path, batch: &ExperimentBatch) -> Result<(), String> {
    let dir = storage_dir.join("experiments");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join(format!("{}.json", sanitize_id(&batch.batch_id))),
        serde_json::to_vec_pretty(batch).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

async fn canonical_project(value: &str) -> Result<PathBuf, String> {
    let project = std::fs::canonicalize(value).map_err(|e| format!("项目路径不可用: {}", e))?;
    let root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(&project)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if !root.status.success()
        || String::from_utf8_lossy(&root.stdout).trim() != project.to_string_lossy()
    {
        return Err("项目路径必须是 Git 仓库根目录".into());
    }
    Ok(project)
}

fn parse_explorer_output(raw: &str, max: usize) -> Vec<ExplorerProposal> {
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<Vec<ExplorerProposal>>(cleaned)
        .ok()
        .unwrap_or_default()
        .into_iter()
        .filter(|proposal| !proposal.id.is_empty() && !proposal.task.is_empty())
        .take(max)
        .collect()
}

fn fallback_proposals(objective: &str, max: usize) -> Vec<ExplorerProposal> {
    let choices = [
        (
            "minimal",
            "最小改动并补充测试",
            "以最小修改实现目标，优先补测试和回滚路径。",
            0.68,
        ),
        (
            "refactor",
            "结构化重构",
            "重构相关模块边界，减少重复逻辑并保持现有行为。",
            0.55,
        ),
        (
            "performance",
            "性能与缓存实验",
            "增加必要的缓存或批处理，使用 benchmark 比较前后性能。",
            0.48,
        ),
        (
            "observability",
            "可观测性完善",
            "增加日志、指标和失败诊断，先验证问题是否真实存在。",
            0.42,
        ),
        (
            "docs",
            "文档与使用体验",
            "补充 README、示例和迁移说明，降低用户使用成本。",
            0.40,
        ),
    ];
    choices
        .into_iter()
        .take(max)
        .map(|(id, title, approach, confidence)| ExplorerProposal {
            id: id.into(),
            title: title.into(),
            approach: approach.into(),
            task: format!("针对目标‘{}’，执行{}", objective, approach),
            expected_value: "形成可验证的独立方案并保留回滚路径".into(),
            risk: "实验结果仅在隔离 worktree 中产生，不自动合并".into(),
            confidence,
        })
        .collect()
}

async fn run_command(cwd: &Path, command: &str) -> ExperimentCommandResult {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(900),
        Command::new("sh")
            .args(["-lc", command])
            .current_dir(cwd)
            .output(),
    )
    .await;
    match result {
        Ok(Ok(output)) => ExperimentCommandResult {
            command: command.into(),
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code(),
        },
        Ok(Err(error)) => ExperimentCommandResult {
            command: command.into(),
            success: false,
            stdout: String::new(),
            stderr: error.to_string(),
            exit_code: None,
        },
        Err(_) => ExperimentCommandResult {
            command: command.into(),
            success: false,
            stdout: String::new(),
            stderr: "命令超时 (900s)".into(),
            exit_code: None,
        },
    }
}

fn git_sync(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn sanitize_id(value: &str) -> String {
    let safe = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .chars()
        .take(80)
        .collect::<String>();
    if safe.is_empty() {
        "variant".into()
    } else {
        safe
    }
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
    fn fallback_explorer_returns_multiple_independent_options() {
        let proposals = fallback_proposals("改善稳定性", 4);
        assert_eq!(proposals.len(), 4);
        assert_ne!(proposals[0].id, proposals[1].id);
    }

    #[test]
    fn ids_are_safe_for_git_paths() {
        assert_eq!(sanitize_id("a/b c"), "a-b-c");
    }
}
