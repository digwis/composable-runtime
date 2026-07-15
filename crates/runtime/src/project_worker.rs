//! Human-supervised project worker.
//!
//! The worker turns a real repository task into an auditable change set:
//! inspect -> (optional approval) isolated worktree -> pi -> verification.

use crate::message::Message;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

const PI_CAPABILITY_EXTENSION_SOURCE: &str = include_str!("../assets/pi-capability-runtime.ts");

#[derive(Debug, Clone, Serialize)]
struct PiCapabilityToolSpec {
    capability: String,
    action: String,
    input: serde_json::Value,
}

#[derive(Debug, Clone)]
struct PiRuntimeBridge {
    extension: PathBuf,
    worktree: PathBuf,
    trace_path: PathBuf,
    capabilities: Vec<PiCapabilityToolSpec>,
}

#[derive(Debug, Clone)]
struct PiProcessSandbox {
    worktree: PathBuf,
    temp_dir: PathBuf,
    trace_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityInvocation {
    pub capability: String,
    pub action: String,
    pub phase: String,
    pub input: serde_json::Value,
    pub output_summary: String,
    pub success: bool,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTaskResult {
    pub task_id: String,
    pub project_path: String,
    pub task: String,
    #[serde(default)]
    pub proposal_id: Option<String>,
    pub approved: bool,
    pub worktree: Option<String>,
    pub branch: Option<String>,
    pub proposal: String,
    pub agent_output: String,
    /// The execution backend that performed the repository work.
    /// Project tasks currently delegate implementation to pi.
    #[serde(default = "default_project_executor")]
    pub executor: String,
    /// Isolation boundary used for the coding agent itself.
    #[serde(default = "default_project_sandbox")]
    pub sandbox_backend: String,
    /// Capabilities that were actually invoked through the runtime MessageBus.
    /// Project skill candidates are intentionally kept separate from this list.
    #[serde(default)]
    pub used_capabilities: Vec<String>,
    #[serde(default)]
    pub capability_trace: Vec<CapabilityInvocation>,
    /// Whether the verified worktree branch was applied to the project root.
    #[serde(default)]
    pub applied: bool,
    /// Why the branch was not applied, when application was intentionally skipped.
    #[serde(default)]
    pub apply_error: Option<String>,
    pub verification: Option<CommandResult>,
    /// Project skills that were selected as candidates for this task.
    #[serde(default)]
    pub skill_candidates: Vec<String>,
    /// Evidence-backed project validation, recorded by the daemon after execution.
    #[serde(default)]
    pub real_validation: Option<ProjectValidation>,
    /// Human review of the project contribution.
    #[serde(default)]
    pub feedback: Option<ProjectFeedback>,
    pub git_status: String,
    pub diff_stat: String,
}

fn default_project_executor() -> String {
    "pi".into()
}

fn default_project_sandbox() -> String {
    "git_worktree".into()
}

pub fn project_pi_sandbox_backend() -> String {
    if pi_os_sandbox_enabled() {
        "macos_sandbox_exec+git_worktree".into()
    } else {
        default_project_sandbox()
    }
}

/// Record verification evidence only for capabilities actually invoked by the
/// runtime. A pi-backed project task therefore remains a pi task instead of
/// being misattributed to marker-derived candidate skills.
pub async fn record_project_validation(
    shared: &Arc<tokio::sync::Mutex<crate::daemon::SharedState>>,
    result: &mut ProjectTaskResult,
) {
    let Some(verification) = result.verification.as_ref() else {
        return;
    };
    let verified_capabilities = result
        .capability_trace
        .iter()
        .filter(|invocation| invocation.phase == "post_change" && invocation.success)
        .map(|invocation| invocation.capability.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut state = shared.lock().await;
    let evidence = format_project_verification_evidence(verification);
    let signal = crate::validator::RealWorldSignal {
        success: verification.success,
        evidence: evidence.clone(),
        strength: crate::validator::SignalStrength::RealTask,
    };
    let mut recorded = Vec::new();
    for name in &verified_capabilities {
        if let Some(genome) = state.evolution.genomes_mut().get_mut(name) {
            crate::validator::record_validation(genome, &signal);
            genome.fitness.recompute_score();
            recorded.push(name.clone());
        }
    }
    if !recorded.is_empty() {
        let _ = state.evolution.save_fitness();
    }
    result.real_validation = Some(ProjectValidation {
        success: verification.success,
        command: verification.command.clone(),
        evidence,
        signal_strength: "RealTask".into(),
        recorded_capabilities: recorded,
    });
}

fn format_project_verification_evidence(verification: &CommandResult) -> String {
    let output = if verification.stderr.trim().is_empty() {
        verification.stdout.trim()
    } else {
        verification.stderr.trim()
    };
    let summary: String = output.chars().take(1000).collect();
    format!(
        "命令: {}\n成功: {}\n退出码: {:?}\n输出:\n{}",
        verification.command, verification.success, verification.exit_code, summary
    )
}

/// Persist a technically successful automatic task as project memory. Human
/// usefulness feedback remains a separate signal.
pub fn record_project_task_outcome(
    storage_dir: &Path,
    result: &ProjectTaskResult,
) -> Result<(), String> {
    let Some(verification) = result.verification.as_ref() else {
        return Ok(());
    };
    if !verification.success || !result.applied || result.diff_stat.trim().is_empty() {
        return Ok(());
    }
    let mut memory = load_project_memory(storage_dir, Path::new(&result.project_path));
    let proposal_id = result
        .proposal_id
        .clone()
        .unwrap_or_else(|| format!("task-{}", short_hash(&result.task)));
    if !memory
        .completed_goals
        .iter()
        .any(|event| event.proposal_id == proposal_id)
    {
        memory.completed_goals.push(ProjectMemoryEvent {
            proposal_id,
            title: result.task.chars().take(120).collect(),
            task: result.task.clone(),
            recorded_at: unix_now(),
        });
        memory.updated_at = Some(unix_now());
        save_project_memory(storage_dir, &memory)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectValidation {
    pub success: bool,
    pub command: String,
    pub evidence: String,
    pub signal_strength: String,
    #[serde(default)]
    pub recorded_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFeedback {
    pub useful: bool,
    pub note: String,
    pub rated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub command: String,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredProject {
    pub name: String,
    pub path: String,
    pub branch: String,
    pub head: String,
    pub dirty: bool,
    pub changed_files: usize,
    pub kind: Vec<String>,
    pub verify_command: Option<String>,
    /// Last bounded verification state for the current repository fingerprint.
    pub health_status: String,
    pub evidence: Vec<String>,
    pub last_checked_at: Option<u64>,
    pub proposals: Vec<ProjectProposal>,
    #[serde(default)]
    pub memory: ProjectMemorySummary,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectMemorySummary {
    pub vision: String,
    pub priorities: Vec<String>,
    pub completed_count: usize,
    pub rejected_count: usize,
    pub feedback_count: usize,
    pub updated_at: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectMemory {
    pub project_path: String,
    pub vision: String,
    pub priorities: Vec<String>,
    pub completed_goals: Vec<ProjectMemoryEvent>,
    pub rejected_goals: Vec<ProjectMemoryEvent>,
    pub feedback: Vec<ProjectMemoryFeedback>,
    pub updated_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMemoryEvent {
    pub proposal_id: String,
    pub title: String,
    pub task: String,
    pub recorded_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMemoryFeedback {
    pub task_id: String,
    pub useful: bool,
    pub note: String,
    pub recorded_at: u64,
}

impl ProjectMemory {
    fn summary(&self) -> ProjectMemorySummary {
        ProjectMemorySummary {
            vision: self.vision.clone(),
            priorities: self.priorities.clone(),
            completed_count: self.completed_goals.len(),
            rejected_count: self.rejected_goals.len(),
            feedback_count: self.feedback.len(),
            updated_at: self.updated_at,
        }
    }

    fn render(&self) -> String {
        let completed = self
            .completed_goals
            .iter()
            .rev()
            .take(8)
            .map(|e| format!("- {}: {}", e.title, e.task))
            .collect::<Vec<_>>()
            .join("\n");
        let rejected = self
            .rejected_goals
            .iter()
            .rev()
            .take(8)
            .map(|e| format!("- {}", e.title))
            .collect::<Vec<_>>()
            .join("\n");
        let feedback = self
            .feedback
            .iter()
            .rev()
            .take(8)
            .map(|e| format!("- useful={} {}", e.useful, e.note))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "项目愿景: {}\n项目优先级: {}\n已完成目标:\n{}\n已拒绝目标:\n{}\n近期反馈:\n{}",
            if self.vision.is_empty() {
                "未设置"
            } else {
                &self.vision
            },
            if self.priorities.is_empty() {
                "未设置".to_string()
            } else {
                self.priorities.join("、")
            },
            if completed.is_empty() {
                "(无)"
            } else {
                &completed
            },
            if rejected.is_empty() {
                "(无)"
            } else {
                &rejected
            },
            if feedback.is_empty() {
                "(无)"
            } else {
                &feedback
            }
        )
    }
}

pub fn record_project_memory_feedback(
    storage_dir: &Path,
    project_path: &str,
    task_id: &str,
    task: &str,
    useful: bool,
    note: &str,
) -> Result<(), String> {
    let path = Path::new(project_path);
    let mut memory = load_project_memory(storage_dir, path);
    memory.project_path = project_path.to_string();
    memory.feedback.push(ProjectMemoryFeedback {
        task_id: task_id.to_string(),
        useful,
        note: note.to_string(),
        recorded_at: unix_now(),
    });
    if memory.feedback.len() > 100 {
        memory.feedback.drain(0..memory.feedback.len() - 100);
    }
    if useful {
        let id = format!("task-{}", short_hash(task));
        if !memory
            .completed_goals
            .iter()
            .any(|event| event.proposal_id == id && event.task == task)
        {
            memory.completed_goals.push(ProjectMemoryEvent {
                proposal_id: id,
                title: task.chars().take(120).collect(),
                task: task.to_string(),
                recorded_at: unix_now(),
            });
        }
    }
    memory.updated_at = Some(unix_now());
    save_project_memory(storage_dir, &memory)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectProposal {
    pub id: String,
    /// Stable, model-supplied intent key. Unlike evidence and wording, this
    /// should stay unchanged while the underlying user goal is unchanged.
    #[serde(default)]
    pub goal_key: String,
    /// Questions whose answers could materially change the decision or plan.
    #[serde(default)]
    pub learning_questions: Vec<String>,
    /// Why this opportunity deserves disproportionate attention.
    #[serde(default)]
    pub impact_scope: String,
    #[serde(default)]
    pub leverage_score: f64,
    pub title: String,
    pub reason: String,
    pub task: String,
    /// Concrete observations that justify interrupting the user.
    /// An empty list means the item is not suitable for an active prompt.
    #[serde(default)]
    pub evidence: Vec<String>,
    pub verify_command: Option<String>,
    pub priority: String,
    pub status: String,
    /// Why this is worth the user's attention, separate from the implementation task.
    #[serde(default)]
    pub expected_value: String,
    /// What could go wrong if the proposal is accepted.
    #[serde(default)]
    pub risk: String,
    /// Broad intent category used for project memory and feedback analysis.
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub value_score: f64,
    #[serde(default)]
    pub risk_score: f64,
    #[serde(default)]
    pub attention_cost: f64,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub initiative: crate::initiative::InitiativeDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectGoalCache {
    fingerprint: String,
    generated_at: u64,
    proposals: Vec<ProjectProposal>,
}

fn project_memory_path(storage_dir: &Path, project_path: &Path) -> PathBuf {
    storage_dir.join("project_memory").join(format!(
        "{}.json",
        short_hash(&project_path.display().to_string())
    ))
}

pub fn project_memory_path_for(storage_dir: &Path, project_path: &str) -> PathBuf {
    project_memory_path(storage_dir, Path::new(project_path))
}

pub fn project_memory_key(project_path: &str) -> String {
    short_hash(project_path)
}

fn load_project_memory(storage_dir: &Path, project_path: &Path) -> ProjectMemory {
    let path = project_memory_path(storage_dir, project_path);
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_else(|| ProjectMemory {
            project_path: project_path.display().to_string(),
            ..Default::default()
        })
}

pub fn load_project_memory_for(storage_dir: &Path, project_path: &str) -> ProjectMemory {
    load_project_memory(storage_dir, Path::new(project_path))
}

fn save_project_memory(storage_dir: &Path, memory: &ProjectMemory) -> Result<(), String> {
    let path = project_memory_path(storage_dir, Path::new(&memory.project_path));
    std::fs::create_dir_all(path.parent().unwrap_or(storage_dir)).map_err(|e| e.to_string())?;
    std::fs::write(
        path,
        serde_json::to_vec_pretty(memory).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

pub fn save_project_memory_for(path: &Path, memory: &ProjectMemory) -> Result<(), String> {
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    std::fs::write(
        path,
        serde_json::to_vec_pretty(memory).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

pub fn now_for_memory() -> u64 {
    unix_now()
}

#[derive(Debug, Clone)]
struct ProjectContext {
    fingerprint: String,
    goal: String,
    readme: String,
    manifests: String,
    markers: Vec<String>,
    recent_commits: String,
    remote_url: Option<String>,
    research_summary: String,
    opportunity_signals: Vec<String>,
}

impl ProjectContext {
    fn read(path: &Path, fingerprint: &str) -> Result<Self, String> {
        let goal = read_bounded(
            path,
            &[".orch-goal.md", ".orch-goal", "PROJECT_GOAL.md"],
            5000,
        );
        let readme = read_bounded(path, &["README.md", "README"], 7000);
        let manifests = read_bounded(
            path,
            &[
                "Cargo.toml",
                "package.json",
                "pyproject.toml",
                "CMakeLists.txt",
            ],
            7000,
        );
        let recent_commits = std::process::Command::new("git")
            .args(["log", "-5", "--oneline"])
            .current_dir(path)
            .output()
            .map(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .chars()
                    .take(2500)
                    .collect()
            })
            .unwrap_or_default();
        let remote_url = std::process::Command::new("git")
            .args(["remote", "get-url", "origin"])
            .current_dir(path)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .filter(|value| value.starts_with("http://") || value.starts_with("https://"));
        let mut markers = Vec::new();
        for name in [
            "TODO",
            "FIXME",
            "HACK",
            "SECURITY",
            "CHANGELOG.md",
            ".github/workflows",
        ] {
            if name.contains('.') || name.contains('/') {
                if path.join(name).exists() {
                    markers.push(format!("存在 {}", name));
                }
            } else if let Ok(out) = std::process::Command::new("rg")
                .args([
                    "-n",
                    "--hidden",
                    "-g",
                    "!.git",
                    "-g",
                    "!node_modules",
                    "-g",
                    "!target",
                    "-g",
                    "!.venv",
                    "-g",
                    "!dist",
                    "-g",
                    "!build",
                    name,
                ])
                .current_dir(path)
                .output()
            {
                if !out.stdout.is_empty() {
                    markers.push(format!("发现 {} 标记", name));
                }
            }
        }
        let content_fingerprint = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            fingerprint,
            goal,
            readme,
            manifests,
            markers.join("\n"),
            recent_commits,
            remote_url.as_deref().unwrap_or("")
        );
        let opportunity_signals = collect_opportunity_signals(path, &goal, &readme, &manifests);
        Ok(Self {
            fingerprint: short_hash(&content_fingerprint),
            goal,
            readme,
            manifests,
            markers,
            recent_commits,
            remote_url,
            research_summary: String::new(),
            opportunity_signals,
        })
    }

    fn render(&self) -> String {
        format!("用户/项目目标:\n{}\n\nREADME:\n{}\n\n项目清单:\n{}\n\n标记:\n{}\n\n机会信号:\n{}\n\n最近提交:\n{}\n\n代码托管地址:\n{}\n\n外部研究证据:\n{}", self.goal_or_fallback(), self.readme, self.manifests, self.markers.join("\n"), if self.opportunity_signals.is_empty() { "未发现结构化机会信号".to_string() } else { self.opportunity_signals.join("\n") }, self.recent_commits, self.remote_url.as_deref().unwrap_or("未发现"), if self.research_summary.is_empty() { "未执行外部研究" } else { &self.research_summary })
    }

    fn goal_or_fallback(&self) -> &str {
        if self.goal.trim().is_empty() {
            "未提供明确目标，请只基于证据提出低风险、可验证的维护建议。"
        } else {
            &self.goal
        }
    }
}

fn read_bounded(path: &Path, names: &[&str], limit: usize) -> String {
    names
        .iter()
        .find_map(|name| std::fs::read_to_string(path.join(name)).ok())
        .map(|v| v.chars().take(limit).collect())
        .unwrap_or_default()
}

fn collect_opportunity_signals(
    path: &Path,
    goal: &str,
    readme: &str,
    manifests: &str,
) -> Vec<String> {
    let mut signals = Vec::new();
    let goal_lower = goal.to_lowercase();
    let readme_lower = readme.to_lowercase();
    let manifests_lower = manifests.to_lowercase();
    if goal.trim().is_empty() {
        signals.push("未发现明确的项目目标，需要用户补充愿景或优先级".into());
    }
    if readme.trim().is_empty() {
        signals.push("内容/文档机会：缺少 README 或项目入口说明".into());
    } else if readme.len() < 400 {
        signals.push("内容/文档机会：README 内容较短，可能缺少真实使用路径".into());
    }
    if path.join("package.json").exists() {
        if !readme_lower.contains("usage") && !readme_lower.contains("使用") {
            signals.push("内容运营机会：Node 项目缺少明确的使用场景或操作示例".into());
        }
        if !manifests_lower.contains("test")
            && !path.join("tests").exists()
            && !path.join("__tests__").exists()
        {
            signals.push("质量机会：未发现 Node 测试脚本或测试目录".into());
        }
    }
    if path.join("Cargo.toml").exists() && !path.join("tests").exists() {
        signals.push("质量机会：Rust 项目未发现集成测试目录，可评估补充关键路径测试".into());
    }
    if goal_lower.contains("内容")
        || goal_lower.contains("运营")
        || goal_lower.contains("用户")
        || goal_lower.contains("增长")
        || goal_lower.contains("seo")
    {
        signals.push(
            "用户目标信号：项目目标包含内容、运营、用户或增长方向，应优先提出可执行的内容/产品实验"
                .into(),
        );
    }
    if path.join("docs").is_dir() || path.join("content").is_dir() || path.join("posts").is_dir() {
        signals.push(
            "内容资产信号：项目包含 docs/content/posts 目录，可分析内容缺口、主题覆盖和更新计划"
                .into(),
        );
    }
    if path.join(".github/issues").exists() || path.join("CHANGELOG.md").exists() {
        signals
            .push("产品反馈信号：发现 issue 或 changelog 资产，可提取用户需求和未完成方向".into());
    }
    signals
}

fn parse_project_proposals(
    raw: &str,
    path: &Path,
    context: &ProjectContext,
    memory: &ProjectMemory,
) -> Vec<ProjectProposal> {
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let Ok(values) = serde_json::from_str::<Vec<serde_json::Value>>(cleaned) else {
        return Vec::new();
    };
    values
        .into_iter()
        .filter_map(|value| {
            let text = |key: &str| {
                value
                    .get(key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string()
            };
            let evidence = value
                .get("evidence")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .take(5)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if evidence.is_empty() || !evidence.iter().any(|item| context.render().contains(item)) {
                return None;
            }
            let task = text("task");
            let title = text("title");
            if title.is_empty() || task.is_empty() {
                return None;
            }
            let supplied_goal_key = text("goal_key");
            let learning_questions = value
                .get("learning_questions")
                .and_then(|item| item.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str())
                        .map(str::trim)
                        .filter(|item| !item.is_empty())
                        .take(3)
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let impact_scope = text("impact_scope");
            let leverage_score = value
                .get("leverage_score")
                .and_then(|item| item.as_f64())
                .unwrap_or_else(|| default_leverage_score(&text("category")))
                .clamp(0.0, 1.0);
            let goal_key = if supplied_goal_key.is_empty() {
                stable_goal_key(&title, &task)
            } else {
                normalize_goal_key(&supplied_goal_key)
            };
            let id = format!(
                "{}-goal-{}",
                path.file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("project"),
                short_hash(&goal_key)
            );
            if memory
                .rejected_goals
                .iter()
                .chain(memory.completed_goals.iter())
                .any(|event| {
                    event.proposal_id == id
                        || event.title == title
                        || project_goals_are_similar(&title, &task, &event.title, &event.task)
                })
            {
                return None;
            }
            let value_score = value
                .get("value_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let risk_score = value
                .get("risk_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let attention_cost = value
                .get("attention_cost")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let priority = if value_score >= 0.7 && risk_score <= 0.55 && attention_cost <= 0.7 {
                "high"
            } else if value_score >= 0.4 {
                "medium"
            } else {
                "low"
            };
            let evidence_strength = (evidence.len() as f64 / 5.0).clamp(0.0, 1.0);
            let confidence =
                (evidence_strength * 0.45 + value_score * 0.35 + (1.0 - risk_score) * 0.20)
                    .clamp(0.0, 1.0);
            let initiative =
                crate::initiative::decide(confidence, value_score, risk_score, attention_cost);
            Some(ProjectProposal {
                id,
                goal_key,
                learning_questions,
                impact_scope,
                leverage_score,
                title,
                reason: text("reason"),
                task,
                evidence,
                verify_command: value
                    .get("verify_command")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                priority: priority.into(),
                status: "proposed".into(),
                expected_value: text("expected_value"),
                risk: text("risk"),
                category: text("category"),
                value_score,
                risk_score,
                attention_cost,
                confidence,
                initiative,
            })
        })
        .take(3)
        .collect()
}

fn normalize_goal_key(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|character| character.is_alphanumeric())
        .collect()
}

fn stable_goal_key(title: &str, task: &str) -> String {
    normalize_goal_key(&format!(
        "{}:{}",
        title,
        task.chars().take(240).collect::<String>()
    ))
}

fn default_leverage_score(category: &str) -> f64 {
    match crate::value_energy::normalize_category(category).as_str() {
        "security" | "bug" => 0.85,
        "feature" | "growth" => 0.75,
        "test" | "dependency" => 0.65,
        "content" | "docs" => 0.50,
        _ => 0.35,
    }
}

pub(crate) fn effective_leverage_score(proposal: &ProjectProposal) -> f64 {
    if proposal.leverage_score > 0.0 {
        proposal.leverage_score.clamp(0.0, 1.0)
    } else {
        default_leverage_score(&proposal.category)
    }
}

/// Character bigrams work for both Chinese and Latin text and let persisted
/// human decisions suppress paraphrases generated by a later model call.
pub(crate) fn project_goals_are_similar(
    left_title: &str,
    left_task: &str,
    right_title: &str,
    right_task: &str,
) -> bool {
    fn goal_family(value: &str) -> Option<&'static str> {
        let value = value.to_lowercase();
        let has = |terms: &[&str]| terms.iter().any(|term| value.contains(term));
        if has(&["macos", "mac ", "intel mac"]) && has(&["intel", "x64", "arm64-only", "架构"]) {
            return Some("mac_intel_support");
        }
        if has(&["ssh"])
            && has(&[
                "凭据",
                "密码",
                "私钥",
                "credential",
                "password",
                "private key",
                "safestorage",
                "加密",
            ])
            && has(&["安全", "security", "明文", "加密", "存储", "storage"])
        {
            return Some("ssh_credential_security");
        }
        if has(&["ssh"])
            && has(&["better-sqlite3", "sqlite", "数据库"])
            && has(&["集成测试", "integration test", "mock ssh", "crud"])
        {
            return Some("ssh_sqlite_integration_tests");
        }
        if has(&["readme"]) && has(&["截图", "screenshot", "screenshots"]) {
            return Some("readme_screenshots");
        }
        if has(&[
            "快速上手",
            "quick start",
            "quickstart",
            "getting started",
            "使用场景",
            "操作教程",
        ]) {
            return Some("quick_start_docs");
        }
        if has(&["github actions", "ci", "工作流"])
            && has(&["typecheck", "类型检查", "测试门禁", "质量门禁"])
        {
            return Some("ci_quality_gate");
        }
        if has(&["better-sqlite3", "原生模块", "native module"])
            && has(&["多平台", "跨平台", "linux", "windows", "macos"])
            && has(&["ci", "构建", "build", "兼容"])
        {
            return Some("native_module_multiplatform_ci");
        }
        None
    }
    let left_text = format!("{} {}", left_title, left_task);
    let right_text = format!("{} {}", right_title, right_task);
    if goal_family(&left_text).is_some_and(|family| Some(family) == goal_family(&right_text)) {
        return true;
    }
    fn bigrams(value: &str) -> std::collections::HashSet<String> {
        let normalized = normalize_goal_key(value);
        let chars = normalized.chars().collect::<Vec<_>>();
        chars
            .windows(2)
            .map(|pair| pair.iter().collect::<String>())
            .collect()
    }
    let left = bigrams(&format!("{}{}", left_title, left_task));
    let right = bigrams(&format!("{}{}", right_title, right_task));
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let shared = left.intersection(&right).count() as f64;
    let dice = (2.0 * shared) / (left.len() + right.len()) as f64;
    dice >= 0.46
}

pub struct ProjectWorker {
    pi_binary: String,
    pi_model: String,
    pi_extension: Option<PathBuf>,
    storage_dir: PathBuf,
    bus: Option<Arc<crate::message_bus::MessageBus>>,
}

fn install_pi_capability_extension(storage_dir: &Path) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("ORCH_PI_CAPABILITY_EXTENSION") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!("Pi Runtime Bridge 不存在: {}", path.display()));
    }
    let dir = storage_dir.join("pi_extensions");
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join("capability-runtime.ts");
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    if current != PI_CAPABILITY_EXTENSION_SOURCE {
        std::fs::write(&path, PI_CAPABILITY_EXTENSION_SOURCE).map_err(|error| error.to_string())?;
    }
    Ok(path)
}

/// Read pi's selected provider/model without copying API keys into the
/// runtime. The returned value is accepted by pi as `provider/model`.
fn load_pi_default_model() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join(".pi/agent/settings.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let provider = value
        .get("defaultProvider")
        .and_then(|v| v.as_str())?
        .trim();
    let model = value.get("defaultModel").and_then(|v| v.as_str())?.trim();
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    if model.contains('/') {
        Some(model.to_string())
    } else {
        Some(format!("{}/{}", provider, model))
    }
}

async fn run_pi_with_fallback(
    binary: &str,
    preferred_model: &str,
    cwd: &Path,
    prompt: &str,
    timeout_secs: u64,
    runtime_bridge: Option<&PiRuntimeBridge>,
    process_sandbox: Option<&PiProcessSandbox>,
) -> Result<String, String> {
    let fallback = std::env::var("ORCH_PI_FALLBACK_MODEL")
        .unwrap_or_else(|_| "opengateway/tencent/hy3".into());
    let mut models = vec![preferred_model.to_string()];
    if !fallback.trim().is_empty() && fallback != preferred_model {
        models.push(fallback);
    }
    let preferred_timeout_secs = std::env::var("ORCH_PI_PREFERRED_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(180)
        .max(15)
        .min(timeout_secs.max(1));
    run_pi_with_models(
        binary,
        &models,
        cwd,
        prompt,
        timeout_secs.max(1),
        preferred_timeout_secs,
        runtime_bridge,
        process_sandbox,
    )
    .await
}

fn pi_error_allows_model_fallback(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    [
        "finish_reason",
        "stream ended",
        "stream",
        "model",
        "timeout",
        "connection",
        "network",
        "authorization failed",
        "429",
        "502",
        "503",
        "504",
        "signal",
        "无错误输出",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || error.contains("模型")
        || error.contains("超时")
        || error.contains("网络")
}

async fn run_pi_with_models(
    binary: &str,
    models: &[String],
    cwd: &Path,
    prompt: &str,
    timeout_secs: u64,
    preferred_timeout_secs: u64,
    runtime_bridge: Option<&PiRuntimeBridge>,
    process_sandbox: Option<&PiProcessSandbox>,
) -> Result<String, String> {
    let mut errors = Vec::new();
    for (index, model) in models.iter().enumerate() {
        let attempt_timeout = if index == 0 && models.len() > 1 {
            preferred_timeout_secs.min(timeout_secs).max(1)
        } else {
            timeout_secs.max(1)
        };
        match run_pi_once(
            binary,
            model,
            cwd,
            prompt,
            attempt_timeout,
            runtime_bridge,
            process_sandbox,
        )
        .await
        {
            Ok(output) => {
                if index > 0 {
                    tracing::warn!(preferred = %models[0], fallback = %model, "pi 首选模型不可用，已使用回退模型");
                }
                return Ok(output);
            }
            Err(error) => {
                let retryable = pi_error_allows_model_fallback(&error);
                errors.push(format!("{}: {}", model, error));
                if !retryable || index + 1 >= models.len() {
                    break;
                }
            }
        }
    }
    Err(format!("pi 执行失败；已尝试模型：{}", errors.join(" | ")))
}

async fn run_pi_once(
    binary: &str,
    model: &str,
    cwd: &Path,
    prompt: &str,
    timeout_secs: u64,
    runtime_bridge: Option<&PiRuntimeBridge>,
    process_sandbox: Option<&PiProcessSandbox>,
) -> Result<String, String> {
    let mut command = if let Some(sandbox) = process_sandbox.filter(|_| pi_os_sandbox_enabled()) {
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command
            .arg("-p")
            .arg(pi_macos_sandbox_profile(sandbox))
            .arg(binary)
            .env("TMPDIR", &sandbox.temp_dir);
        command
    } else {
        Command::new(binary)
    };
    command.args([
        "--no-session",
        "--approve",
        "--mode",
        "text",
        "--model",
        model,
        "--tools",
        if runtime_bridge.is_some() {
            "read,grep,find,ls,bash,edit,write,runtime_capabilities,runtime_capability_call"
        } else {
            "read,grep,find,ls,bash,edit,write"
        },
        "-p",
        prompt,
    ]);
    if let Some(bridge) = runtime_bridge {
        let capabilities = serde_json::to_string(&bridge.capabilities)
            .map_err(|error| format!("序列化 Pi 能力白名单失败: {}", error))?;
        command
            .args(["--extension", bridge.extension.to_string_lossy().as_ref()])
            .env("ORCH_PI_CAPABILITIES", capabilities)
            .env("ORCH_PI_WORKTREE", &bridge.worktree)
            .env("ORCH_PI_CAPABILITY_TRACE", &bridge.trace_path)
            .env(
                "ORCH_DAEMON_URL",
                std::env::var("ORCH_DAEMON_URL").unwrap_or_else(|_| "http://127.0.0.1:7331".into()),
            );
    }
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        command.kill_on_drop(true).current_dir(cwd).output(),
    )
    .await
    .map_err(|_| format!("pi 执行超时 ({}s)", timeout_secs))?
    .map_err(|error| format!("启动 pi 失败: {}", error))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if output.status.success() {
        return Ok(stdout);
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let detail = if stderr.is_empty() { stdout } else { stderr };
    let code = output
        .status
        .code()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "signal".into());
    Err(format!(
        "退出码 {}: {}",
        code,
        if detail.is_empty() {
            "无错误输出"
        } else {
            &detail
        }
    ))
}

fn pi_os_sandbox_enabled() -> bool {
    if !cfg!(target_os = "macos") || !Path::new("/usr/bin/sandbox-exec").is_file() {
        return false;
    }
    for key in ["ORCH_OS_SANDBOX", "ORCH_PI_OS_SANDBOX"] {
        if std::env::var(key).is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false"
            )
        }) {
            return false;
        }
    }
    true
}

fn pi_macos_sandbox_profile(sandbox: &PiProcessSandbox) -> String {
    let mut writable = vec![sandbox.worktree.clone(), sandbox.temp_dir.clone()];
    if let Some(trace_dir) = sandbox.trace_dir.as_ref() {
        writable.push(trace_dir.clone());
    }
    let mut profile = String::from("(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write* (literal \"/dev/null\")");
    for path in writable {
        let normalized = path.canonicalize().unwrap_or(path);
        profile.push_str(&format!(
            " (subpath \"{}\")",
            escape_pi_sandbox_literal(&normalized.to_string_lossy())
        ));
    }
    profile.push_str(")\n");
    profile
}

fn escape_pi_sandbox_literal(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn prepare_pi_process_sandbox(
    storage_dir: &Path,
    task_id: &str,
    worktree: &Path,
    runtime_bridge: Option<&PiRuntimeBridge>,
) -> Result<PiProcessSandbox, String> {
    let temp_dir = storage_dir.join("pi_tmp").join(task_id);
    std::fs::create_dir_all(&temp_dir).map_err(|error| error.to_string())?;
    Ok(PiProcessSandbox {
        worktree: worktree.to_path_buf(),
        temp_dir,
        trace_dir: runtime_bridge
            .and_then(|bridge| bridge.trace_path.parent())
            .map(Path::to_path_buf),
    })
}

impl ProjectWorker {
    fn prepare_pi_runtime_bridge(
        &self,
        task_id: &str,
        worktree: &Path,
        capability_plan: &[(String, String, serde_json::Value)],
    ) -> Result<Option<PiRuntimeBridge>, String> {
        let Some(extension) = self.pi_extension.as_ref() else {
            return Ok(None);
        };
        if capability_plan.is_empty() {
            return Ok(None);
        }
        let trace_dir = self.storage_dir.join("pi_capability_traces");
        std::fs::create_dir_all(&trace_dir).map_err(|error| error.to_string())?;
        let trace_path = trace_dir.join(format!("{}.jsonl", task_id));
        // A durable retry gets a fresh trace for this Pi attempt; baseline and
        // post-change evidence remain in the persisted task result.
        std::fs::write(&trace_path, b"").map_err(|error| error.to_string())?;
        let capabilities = capability_plan
            .iter()
            .map(|(capability, action, input)| PiCapabilityToolSpec {
                capability: capability.clone(),
                action: action.clone(),
                input: capability_input_for_path(input, worktree),
            })
            .collect();
        Ok(Some(PiRuntimeBridge {
            extension: extension.clone(),
            worktree: worktree.to_path_buf(),
            trace_path,
            capabilities,
        }))
    }

    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        let storage_dir = storage_dir.into();
        let pi_extension = match install_pi_capability_extension(&storage_dir) {
            Ok(path) => Some(path),
            Err(error) => {
                tracing::warn!("安装 Pi Runtime Bridge 失败: {}", error);
                None
            }
        };
        Self {
            pi_binary: std::env::var("ORCH_PI_BIN").unwrap_or_else(|_| "pi".into()),
            // Prefer an explicitly selected model, then the model selected in pi's
            // settings. A compatibility fallback is added at execution time.
            pi_model: std::env::var("ORCH_PI_MODEL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .or_else(load_pi_default_model)
                .unwrap_or_else(|| "opengateway/tencent/hy3".into()),
            pi_extension,
            storage_dir,
            bus: None,
        }
    }

    pub fn with_bus(mut self, bus: Arc<crate::message_bus::MessageBus>) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Discover immediate child repositories under configured project roots.
    /// This is read-only and intentionally does not recurse into every folder.
    pub async fn discover_projects(&self, roots: &[PathBuf]) -> Vec<DiscoveredProject> {
        self.discover_projects_with_driver(roots, None).await
    }

    /// Discover projects without any LLM, remote research, or proposal
    /// generation. This path is used by polling APIs and must return the
    /// current workspace state quickly. Static health/opportunity signals are
    /// still included, as are valid cached LLM proposals from a recent scan.
    pub async fn discover_projects_fast(&self, roots: &[PathBuf]) -> Vec<DiscoveredProject> {
        let mut paths = Vec::new();
        for root in roots {
            if root.is_dir() && root.join(".git").exists() {
                paths.push(root.clone());
                continue;
            }
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() || !path.join(".git").exists() {
                    continue;
                }
                paths.push(path);
            }
        }
        let mut jobs = tokio::task::JoinSet::new();
        for path in paths {
            jobs.spawn(async move { inspect_project_with_health(&path, None).await });
        }
        let mut projects = Vec::new();
        while let Some(result) = jobs.join_next().await {
            if let Ok(Ok(project)) = result {
                projects.push(self.enrich_project_fast(project).await);
            }
        }
        projects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        projects
    }

    /// Discover projects and, when an LLM is available, generate cached,
    /// evidence-backed development proposals from project context.
    pub async fn discover_projects_with_driver(
        &self,
        roots: &[PathBuf],
        driver: Option<Arc<dyn crate::driver::EvolutionDriver>>,
    ) -> Vec<DiscoveredProject> {
        let mut paths = Vec::new();
        for root in roots {
            if root.is_dir() && root.join(".git").exists() {
                paths.push(root.clone());
                continue;
            }
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            paths.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.is_dir() && path.join(".git").exists()),
            );
        }
        let mut jobs = tokio::task::JoinSet::new();
        let storage = self.storage_dir.clone();
        for path in paths {
            let storage = storage.clone();
            jobs.spawn(async move { inspect_project_with_health(&path, Some(&storage)).await });
        }
        let mut projects = Vec::new();
        while let Some(result) = jobs.join_next().await {
            if let Ok(Ok(project)) = result {
                projects.push(self.enrich_project_fast(project).await);
            }
        }
        if driver.is_none() {
            projects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
            return projects;
        }
        for project in &mut projects {
            let enriched = self.enrich_project(project.clone(), driver.clone()).await;
            *project = enriched;
        }
        projects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        projects
    }

    async fn enrich_project_fast(&self, mut project: DiscoveredProject) -> DiscoveredProject {
        let memory = load_project_memory(&self.storage_dir, Path::new(&project.path));
        project.memory = memory.summary();
        if let Some(cached) = self.load_cached_project_proposals(Path::new(&project.path), &memory)
        {
            let existing = project
                .proposals
                .iter()
                .map(|p| p.id.clone())
                .collect::<std::collections::HashSet<_>>();
            project.proposals.extend(
                cached
                    .into_iter()
                    .filter(|proposal| !existing.contains(&proposal.id)),
            );
        }
        project
    }

    fn load_cached_project_proposals(
        &self,
        path: &Path,
        memory: &ProjectMemory,
    ) -> Option<Vec<ProjectProposal>> {
        let cache_path = self
            .storage_dir
            .join("project_goals")
            .join(format!("{}.json", short_hash(&path.display().to_string())));
        let raw = std::fs::read_to_string(cache_path).ok()?;
        let cache = serde_json::from_str::<ProjectGoalCache>(&raw).ok()?;
        let _memory_fingerprint = short_hash(&serde_json::to_string(memory).unwrap_or_default());
        // Fast polling deliberately accepts a recent cache without rescanning
        // the repository. The background refresh revalidates the fingerprint
        // and replaces stale proposals after a project change.
        (unix_now().saturating_sub(cache.generated_at) < 900).then_some(cache.proposals)
    }

    async fn enrich_project(
        &self,
        mut project: DiscoveredProject,
        driver: Option<Arc<dyn crate::driver::EvolutionDriver>>,
    ) -> DiscoveredProject {
        let fingerprint = format!("{}\n{}", project.head, project.path);
        let memory = load_project_memory(&self.storage_dir, Path::new(&project.path));
        project.memory = memory.summary();
        if let Some(generated) = self
            .generate_project_proposals(Path::new(&project.path), &fingerprint, driver, &memory)
            .await
        {
            let existing = project
                .proposals
                .iter()
                .map(|p| p.id.clone())
                .collect::<std::collections::HashSet<_>>();
            for proposal in generated {
                if !existing.contains(&proposal.id) {
                    project.proposals.push(proposal);
                }
            }
        }
        project
    }

    async fn generate_project_proposals(
        &self,
        path: &Path,
        fingerprint: &str,
        driver: Option<Arc<dyn crate::driver::EvolutionDriver>>,
        memory: &ProjectMemory,
    ) -> Option<Vec<ProjectProposal>> {
        let context = ProjectContext::read(path, fingerprint).ok()?;
        let cache_dir = self.storage_dir.join("project_goals");
        let _ = std::fs::create_dir_all(&cache_dir);
        let cache_path =
            cache_dir.join(format!("{}.json", short_hash(&path.display().to_string())));
        let memory_fingerprint = short_hash(&serde_json::to_string(memory).unwrap_or_default());
        if let Ok(raw) = std::fs::read_to_string(&cache_path) {
            if let Ok(cache) = serde_json::from_str::<ProjectGoalCache>(&raw) {
                if cache.fingerprint == format!("{}:{}", context.fingerprint, memory_fingerprint)
                    && unix_now().saturating_sub(cache.generated_at) < 900
                {
                    return Some(cache.proposals);
                }
            }
        }

        let Some(driver) = driver else { return None };
        if !driver.has_llm_backend() {
            return None;
        }
        let mut context = context;
        if let Some(remote) = context.remote_url.clone() {
            let engine = crate::research::ResearchEngine::new(&self.storage_dir);
            let research = engine
                .research(crate::research::ResearchRequest {
                    urls: vec![remote],
                    query: None,
                    max_sources: Some(3),
                    force_refresh: false,
                })
                .await;
            context.research_summary = research
                .evidence
                .iter()
                .map(|item| {
                    format!(
                        "来源={} 可信度={:.2} 标题={} 摘要={}",
                        item.url,
                        item.confidence,
                        item.title,
                        item.excerpt.chars().take(500).collect::<String>()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
        let prompt = format!(
            "你是一个面向真实用户目标的项目发展 Agent。根据下面的仓库证据和长期记忆，提出最多 5 个高价值、可验证、需要人类批准的机会。你必须优先考虑：1) 可复现 Bug、回归和安全风险；2) 用户真正需要的新功能和产品实验；3) 内容运营、SEO、文档、教程和用户增长；4) 测试、CI、依赖和可维护性。不要只提出 git/cargo/npm 工具操作，也不要凭空编造需求。每个候选必须绑定证据，证据必须逐字引用下面上下文中的片段。不要再次提出长期记忆里已完成、已拒绝或语义相同的目标。若证据不足返回空数组。严格只输出 JSON 数组，每项字段：goal_key(简短稳定的英文 snake_case 意图键；同一目标即使措辞或证据变化也必须保持相同), learning_questions(0到3个问题；只填写答案可能改变是否执行、实施方向或验收标准的关键未知，不要填写可直接从上下文回答的问题), impact_scope(core_frequency|differentiator|user_interest|supporting；分别表示高频核心路径、产品差异化、与用户利益/安全直接相关、外围支持), leverage_score(0到1；只有具备高频使用、差异化卖点或用户利益证据时才能给高分), title, reason, task, evidence(字符串数组，必须引用下面证据), expected_value, risk, category( bug|feature|content|growth|test|docs|security|dependency|maintenance ), value_score(0到1，代表对用户目标的预期收益), risk_score(0到1), attention_cost(0到1，代表需要用户投入的注意力), priority(high|medium|low), verify_command(字符串或 null)。评分必须由证据、项目目标和长期反馈支持，不能仅凭主观判断。对于 feature/content/growth 类任务，task 必须描述一个可在隔离环境中完成并可由用户验收的最小实验或产物。\n\n仓库上下文:\n{}",
            format!("{}\n\n长期项目记忆:\n{}", context.render(), memory.render())
        );
        let response = match driver
            .execute(
                &prompt,
                "smart:project",
                Some("只返回 JSON，不要 markdown。重视用户目标、证据和可验证收益。"),
            )
            .await
        {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(project = %path.display(), "项目目标生成暂不可用: {}", error);
                // Avoid hammering an unavailable provider on every dashboard poll.
                let cache = ProjectGoalCache {
                    fingerprint: format!("{}:{}", context.fingerprint, memory_fingerprint),
                    generated_at: unix_now(),
                    proposals: Vec::new(),
                };
                let _ = std::fs::write(
                    &cache_path,
                    serde_json::to_vec_pretty(&cache).unwrap_or_default(),
                );
                return Some(Vec::new());
            }
        };
        let proposals = parse_project_proposals(&response, path, &context, memory);
        let cache = ProjectGoalCache {
            fingerprint: format!("{}:{}", context.fingerprint, memory_fingerprint),
            generated_at: unix_now(),
            proposals: proposals.clone(),
        };
        let _ = std::fs::write(&cache_path, serde_json::to_vec_pretty(&cache).ok()?);
        Some(proposals)
    }

    /// Run a supervised task. Without approval this only returns a proposal.
    pub async fn run(
        &self,
        project_path: impl AsRef<Path>,
        task: &str,
        verify_command: Option<&str>,
        approve: bool,
    ) -> Result<ProjectTaskResult, String> {
        self.run_with_task_id(project_path, task, verify_command, approve, None)
            .await
    }

    /// Run a task with a stable durable id. Reusing the id reattaches to an
    /// interrupted worktree, allowing daemon recovery to continue from files
    /// already produced by the previous attempt.
    pub async fn run_with_task_id(
        &self,
        project_path: impl AsRef<Path>,
        task: &str,
        verify_command: Option<&str>,
        approve: bool,
        durable_task_id: Option<&str>,
    ) -> Result<ProjectTaskResult, String> {
        let project = std::fs::canonicalize(project_path.as_ref())
            .map_err(|e| format!("项目路径不可用: {}", e))?;
        let root = git(&project, &["rev-parse", "--show-toplevel"]).await?;
        if root.trim() != project.to_string_lossy() {
            return Err("项目路径必须是 Git 仓库根目录".into());
        }
        let status = git(&project, &["status", "--short"]).await?;
        let log = git(&project, &["log", "-1", "--oneline"]).await?;
        let proposal = format!(
            "项目: {}\n当前提交: {}\n未提交变更:\n{}\n\n任务: {}",
            project.display(),
            log.trim(),
            if status.trim().is_empty() {
                "(无)"
            } else {
                status.trim()
            },
            task
        );
        let task_id = durable_task_id
            .map(str::to_string)
            .unwrap_or_else(|| format!("project-{}", uuid::Uuid::new_v4()));
        let skill_candidates = project_skill_candidates(&project);

        if !approve {
            return Ok(ProjectTaskResult {
                task_id,
                project_path: project.display().to_string(),
                task: task.into(),
                proposal_id: None,
                approved: false,
                worktree: None,
                branch: None,
                proposal,
                agent_output: String::new(),
                executor: default_project_executor(),
                sandbox_backend: project_pi_sandbox_backend(),
                used_capabilities: Vec::new(),
                capability_trace: Vec::new(),
                applied: false,
                apply_error: None,
                verification: None,
                skill_candidates,
                real_validation: None,
                feedback: None,
                git_status: status,
                diff_stat: String::new(),
            });
        }

        let branch = format!("orch/{}", task_id.trim_start_matches("project-"));
        let worktree = self.storage_dir.join("worktrees").join(&task_id);
        std::fs::create_dir_all(worktree.parent().unwrap()).map_err(|e| e.to_string())?;
        if worktree.join(".git").exists() {
            git(&worktree, &["rev-parse", "--show-toplevel"]).await?;
        } else {
            let branch_exists = !git(&project, &["branch", "--list", &branch])
                .await?
                .trim()
                .is_empty();
            if branch_exists {
                git(
                    &project,
                    &[
                        "worktree",
                        "add",
                        worktree.to_string_lossy().as_ref(),
                        &branch,
                    ],
                )
                .await?;
            } else {
                git(
                    &project,
                    &[
                        "worktree",
                        "add",
                        "-b",
                        &branch,
                        worktree.to_string_lossy().as_ref(),
                        "HEAD",
                    ],
                )
                .await?;
            }
        }

        let capability_plan = self
            .select_capability_plan(&project, task, verify_command)
            .await;
        let runtime_bridge =
            self.prepare_pi_runtime_bridge(&task_id, &worktree, &capability_plan)?;
        let process_sandbox = prepare_pi_process_sandbox(
            &self.storage_dir,
            &task_id,
            &worktree,
            runtime_bridge.as_ref(),
        )?;
        let mut capability_trace = Vec::new();
        let mut used_capabilities = Vec::new();
        for (capability, action, input) in &capability_plan {
            let baseline_input = capability_input_for_path(input, &worktree);
            let invocation = self
                .invoke_capability(capability, action, baseline_input, "baseline")
                .await;
            if !used_capabilities.contains(capability) {
                used_capabilities.push(capability.clone());
            }
            capability_trace.push(invocation);
        }

        let capability_guidance = if capability_trace.is_empty() {
            "系统没有找到与本任务语义匹配的已进化能力，请自行分析并完成任务。".to_string()
        } else {
            let evidence = capability_trace
                .iter()
                .map(|item| {
                    format!(
                        "- {}.{}: {}（{}，{}ms）",
                        item.capability,
                        item.action,
                        item.output_summary,
                        if item.success { "成功" } else { "失败" },
                        item.elapsed_ms
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "系统已在修改前实际执行以下可复用能力。请把成功结果作为项目证据使用；失败结果只作为风险信号，不要假装它成功。\n{}",
                evidence
            )
        };
        let runtime_guidance = if runtime_bridge.is_some() {
            "你还可以使用 runtime_capabilities 查看本任务获准的进化能力，并在分析或验证需要时使用 runtime_capability_call 动态调用。只调用白名单中的能力；必须根据返回的 success 判断结果，不得把失败调用描述为成功。"
        } else {
            "本任务没有可供动态调用的进化能力，请使用常规工具完成。"
        };
        let prompt = format!(
            "你正在一个隔离 Git worktree 中工作。任务：{}\n\n{}\n\n{}\n\n要求：只修改完成任务所需文件；运行相关测试；不要 push、不要创建 PR。优先利用系统已执行的能力证据，不要重复实现已有稳定检查。完成后简要报告修改、测试和剩余风险。",
            task, capability_guidance, runtime_guidance
        );
        let timeout_secs = std::env::var("ORCH_PI_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(900);
        let agent_result = run_pi_with_fallback(
            &self.pi_binary,
            &self.pi_model,
            &worktree,
            &prompt,
            timeout_secs,
            runtime_bridge.as_ref(),
            Some(&process_sandbox),
        )
        .await;
        let _ = std::fs::remove_dir_all(&process_sandbox.temp_dir);
        let agent_output = agent_result?;
        if let Some(bridge) = runtime_bridge.as_ref() {
            for invocation in read_pi_capability_trace(&bridge.trace_path) {
                if !used_capabilities.contains(&invocation.capability) {
                    used_capabilities.push(invocation.capability.clone());
                }
                capability_trace.push(invocation);
            }
        }
        let verification = match verify_command {
            Some(cmd) => Some(run_shell(&worktree, cmd).await?),
            None => None,
        };
        for (capability, action, input) in &capability_plan {
            let post_input = capability_input_for_path(input, &worktree);
            let invocation = self
                .invoke_capability(capability, action, post_input, "post_change")
                .await;
            if !used_capabilities.contains(capability) {
                used_capabilities.push(capability.clone());
            }
            capability_trace.push(invocation);
        }
        let diff_stat = collect_git_change_summary(&worktree).await?;
        let (applied, apply_error) =
            apply_verified_changes(&project, &worktree, &branch, &verification).await;
        let git_status = git(&worktree, &["status", "--short"]).await?;
        let result = ProjectTaskResult {
            task_id,
            project_path: project.display().to_string(),
            task: task.into(),
            proposal_id: None,
            approved: true,
            worktree: Some(worktree.display().to_string()),
            branch: Some(branch),
            proposal,
            agent_output,
            executor: default_project_executor(),
            sandbox_backend: project_pi_sandbox_backend(),
            used_capabilities,
            capability_trace,
            applied,
            apply_error,
            verification,
            skill_candidates,
            real_validation: None,
            feedback: None,
            git_status,
            diff_stat,
        };
        let dir = self.storage_dir.join("project_tasks");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(
            dir.join(format!("{}.json", result.task_id)),
            serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(result)
    }

    async fn select_capability_plan(
        &self,
        project: &Path,
        task: &str,
        verify_command: Option<&str>,
    ) -> Vec<(String, String, serde_json::Value)> {
        let Some(bus) = self.bus.as_ref() else {
            return Vec::new();
        };
        let infos = bus.introspect().await;
        let context = format!(
            "{} {} {} {}",
            task,
            verify_command.unwrap_or_default(),
            if project.join("Cargo.toml").exists() {
                "rust cargo"
            } else {
                ""
            },
            if project.join("package.json").exists() {
                "node npm"
            } else {
                ""
            },
        );
        let context_tokens = normalized_tokens(&context);

        // Project tasks may invoke evolved capabilities, but baseline/post-change
        // calls must remain bounded. Mutating actions (commit, install, add_dep,
        // clean, delete, upload, deploy, send) are intentionally excluded.
        let mut ranked = infos
            .iter()
            .flat_map(|info| {
                info.actions.iter().filter_map(|action| {
                    if !is_project_safe_action(action)
                        || !standard_capability_matches_project(&info.name, project)
                    {
                        return None;
                    }
                    let score = capability_match_score(info, action, &context_tokens, project);
                    (score > 0).then_some((score, info.name.clone(), action.clone()))
                })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let limit = std::env::var("ORCH_PROJECT_CAPABILITY_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3)
            .clamp(1, 5);
        let mut selected = Vec::new();
        for (score, capability, action) in ranked {
            let standard = matches!(
                capability.as_str(),
                "cargo_ops-v2" | "npm_ops-v1" | "git_ops-v1-v2" | "cmake_ops-v2" | "make_ops"
            );
            if score < 3 && !standard {
                continue;
            }
            if selected.iter().any(|(name, _, _)| name == &capability) {
                continue;
            }
            let input = capability_input(&capability, &action, project, task, verify_command);
            selected.push((capability, action, input));
            if selected.len() >= limit {
                break;
            }
        }
        if !selected.is_empty() {
            return selected;
        }

        // Stable fallback for repositories where the semantic matcher has no
        // evidence or the available capability store is still being loaded.
        if project.join("Cargo.toml").exists() && has_action(&infos, "cargo_ops-v2", "check") {
            return vec![(
                "cargo_ops-v2".into(),
                "check".into(),
                serde_json::json!({"path": project.display().to_string()}),
            )];
        }
        if project.join("package.json").exists() && has_action(&infos, "npm_ops-v1", "run_npm") {
            return vec![(
                "npm_ops-v1".into(),
                "run_npm".into(),
                capability_input("npm_ops-v1", "run_npm", project, task, verify_command),
            )];
        }
        if has_action(&infos, "git_ops-v1-v2", "git_diff_analysis") {
            vec![(
                "git_ops-v1-v2".into(),
                "git_diff_analysis".into(),
                serde_json::json!({"repo_path": project.display().to_string(), "branch": "HEAD"}),
            )]
        } else {
            Vec::new()
        }
    }

    async fn invoke_capability(
        &self,
        capability: &str,
        action: &str,
        input: serde_json::Value,
        phase: &str,
    ) -> CapabilityInvocation {
        let started = std::time::Instant::now();
        let (success, output_summary) = match self.bus.as_ref() {
            Some(bus) => {
                let message = Message::builder()
                    .from("project_worker")
                    .to(capability)
                    .action(action)
                    .payload(input.clone())
                    .metadata("execution_scope", "real_project_task")
                    .metadata("phase", phase)
                    .build();
                match bus.send(message).await {
                    Ok(response) => (
                        response
                            .payload
                            .get("success")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(true),
                        summarize_json(&response.payload),
                    ),
                    Err(error) => (false, error.to_string()),
                }
            }
            None => (false, "MessageBus 未注入".into()),
        };
        CapabilityInvocation {
            capability: capability.into(),
            action: action.into(),
            phase: phase.into(),
            input,
            output_summary,
            success,
            elapsed_ms: started.elapsed().as_millis() as u64,
        }
    }
}

fn summarize_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_default()
        .chars()
        .take(1200)
        .collect()
}

fn read_pi_capability_trace(path: &Path) -> Vec<CapabilityInvocation> {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .filter_map(|line| {
            let invocation = serde_json::from_str::<CapabilityInvocation>(line).ok()?;
            (invocation.phase == "agent_dynamic").then_some(invocation)
        })
        .collect()
}

fn normalized_tokens(value: &str) -> std::collections::HashSet<String> {
    value
        .to_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 2)
        .map(str::to_string)
        .collect()
}

fn is_project_safe_action(action: &str) -> bool {
    let action = action.to_lowercase();
    let blocked = [
        "commit",
        "push",
        "install",
        "add",
        "delete",
        "remove",
        "clean",
        "deploy",
        "upload",
        "send",
        "write",
        "create",
        "new_project",
        "publish",
        "checkout",
    ];
    !blocked.iter().any(|word| action.contains(word))
}

fn standard_capability_matches_project(capability: &str, project: &Path) -> bool {
    match capability {
        "cargo_ops-v2" => project.join("Cargo.toml").exists(),
        "npm_ops-v1" => project.join("package.json").exists(),
        "cmake_ops-v2" => project.join("CMakeLists.txt").exists(),
        "make_ops" => project.join("Makefile").exists() || project.join("CMakeLists.txt").exists(),
        "git_ops-v1-v2" => project.join(".git").exists(),
        _ => true,
    }
}

fn capability_match_score(
    info: &crate::orchestrator::CapabilityInfo,
    action: &str,
    context_tokens: &std::collections::HashSet<String>,
    project: &Path,
) -> u32 {
    let text = format!("{} {} {}", info.name, info.description, action);
    let tokens = normalized_tokens(&text);
    let overlap = tokens.intersection(context_tokens).count() as u32;
    let mut score = overlap;
    let name = info.name.to_lowercase();
    if project.join("Cargo.toml").exists() && (name.contains("cargo") || name.contains("rust")) {
        score += 3;
    }
    if project.join("package.json").exists() && (name.contains("npm") || name.contains("node")) {
        score += 3;
    }
    if project.join("CMakeLists.txt").exists() && name.contains("cmake") {
        score += 3;
    }
    if name.contains("git") && project.join(".git").exists() {
        score += 2;
    }
    if [
        "check", "test", "status", "diff", "analysis", "analyze", "build", "lint", "tree",
    ]
    .iter()
    .any(|word| action.to_lowercase().contains(word))
    {
        score += 1;
    }
    score
}

fn has_action(
    infos: &[crate::orchestrator::CapabilityInfo],
    capability: &str,
    action: &str,
) -> bool {
    infos
        .iter()
        .any(|info| info.name == capability && info.actions.iter().any(|item| item == action))
}

fn capability_input(
    capability: &str,
    action: &str,
    project: &Path,
    task: &str,
    verify_command: Option<&str>,
) -> serde_json::Value {
    let path = project.display().to_string();
    let command = verify_command.unwrap_or_default();
    if capability == "cargo_ops-v2" {
        return serde_json::json!({"path": path});
    }
    if capability == "npm_ops-v1" {
        let args = command
            .trim_start_matches("npm")
            .split_whitespace()
            .map(str::to_string)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        return serde_json::json!({"args": if args.is_empty() { vec!["test".to_string()] } else { args }, "cwd": path});
    }
    if capability == "cmake_ops-v2" {
        return serde_json::json!({
            "source_dir": path,
            "build_dir": project.join("build").display().to_string(),
            "build_type": "Release"
        });
    }
    if capability.contains("git") || action.contains("diff") || action.contains("status") {
        return serde_json::json!({"repo_path": path, "branch": "HEAD"});
    }
    let mut input = serde_json::json!({"path": path, "cwd": path, "repo_path": path});
    if let Some(object) = input.as_object_mut() {
        object.insert("task".into(), serde_json::Value::String(task.into()));
        object.insert("command".into(), serde_json::Value::String(command.into()));
    }
    input
}

fn capability_input_for_path(input: &serde_json::Value, path: &Path) -> serde_json::Value {
    let mut updated = input.clone();
    if let Some(object) = updated.as_object_mut() {
        let path_string = path.display().to_string();
        for key in ["path", "cwd", "repo_path", "source_dir"] {
            if object.contains_key(key) {
                object.insert(key.into(), serde_json::Value::String(path_string.clone()));
            }
        }
        if object.contains_key("build_dir") {
            object.insert(
                "build_dir".into(),
                serde_json::Value::String(path.join("build").display().to_string()),
            );
        }
    }
    updated
}

async fn apply_verified_changes(
    project: &Path,
    worktree: &Path,
    branch: &str,
    verification: &Option<CommandResult>,
) -> (bool, Option<String>) {
    let Some(result) = verification else {
        return (false, Some("未提供验证命令，保留隔离分支未应用".into()));
    };
    if !result.success {
        return (false, Some("验证失败，保留隔离分支未应用".into()));
    }
    let worktree_status = match git(worktree, &["status", "--short"]).await {
        Ok(status) => status,
        Err(error) => return (false, Some(format!("读取隔离工作区状态失败: {}", error))),
    };
    if worktree_status.trim().is_empty() {
        return (false, Some("验证通过但没有检测到待应用的修改".into()));
    }
    let root_status = match git(project, &["status", "--short"]).await {
        Ok(status) => status,
        Err(error) => return (false, Some(format!("读取项目工作区状态失败: {}", error))),
    };
    if !root_status.trim().is_empty() {
        return (
            false,
            Some("项目根目录存在未提交改动，为避免覆盖用户内容而未自动应用".into()),
        );
    }
    if let Err(error) = git(worktree, &["add", "-A"]).await {
        return (false, Some(format!("暂存隔离工作区改动失败: {}", error)));
    }
    if let Err(error) = git(
        worktree,
        &["commit", "-m", "orch: apply verified automatic task"],
    )
    .await
    {
        return (false, Some(format!("提交隔离工作区改动失败: {}", error)));
    }
    match git(project, &["merge", "--no-edit", branch]).await {
        Ok(_) => (true, None),
        Err(error) => {
            let _ = git(project, &["merge", "--abort"]).await;
            (false, Some(format!("验证通过但合并失败: {}", error)))
        }
    }
}

async fn collect_git_change_summary(path: &Path) -> Result<String, String> {
    let unstaged = git(path, &["diff", "--stat"]).await?;
    let staged = git(path, &["diff", "--cached", "--stat"]).await?;
    let status = git(path, &["status", "--short"]).await?;
    Ok(format_git_change_summary(&unstaged, &staged, &status))
}

fn format_git_change_summary(unstaged: &str, staged: &str, status: &str) -> String {
    let mut sections = Vec::new();
    if !unstaged.trim().is_empty() {
        sections.push(unstaged.trim().to_string());
    }
    if !staged.trim().is_empty() {
        sections.push(staged.trim().to_string());
    }
    let untracked = status
        .lines()
        .filter_map(|line| line.strip_prefix("?? "))
        .collect::<Vec<_>>();
    if !untracked.is_empty() {
        sections.push(format!(
            "{} 个未跟踪文件: {}",
            untracked.len(),
            untracked.join(", ")
        ));
    }
    sections.join("\n")
}

/// Map repository markers to stable, mature project skills. These are only
/// candidates: the daemon records signals only for capabilities that actually
/// exist in the current genome store.
fn project_skill_candidates(path: &Path) -> Vec<String> {
    let mut candidates = vec!["git_ops-v1-v2".to_string()];
    if path.join("Cargo.toml").exists() {
        candidates.push("cargo_ops-v2".into());
    }
    if path.join("package.json").exists() {
        candidates.push("npm_ops-v1".into());
    }
    if path.join("pyproject.toml").exists() || path.join("requirements.txt").exists() {
        candidates.push("pip_ops-v1-v2".into());
    }
    if path.join("CMakeLists.txt").exists() {
        candidates.push("cmake_ops-v2".into());
        candidates.push("make_ops".into());
    }
    candidates
}

#[cfg(test)]
async fn inspect_project(path: &Path) -> Result<DiscoveredProject, String> {
    inspect_project_with_health(path, None).await
}

async fn inspect_project_with_health(
    path: &Path,
    storage_dir: Option<&Path>,
) -> Result<DiscoveredProject, String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project")
        .to_string();
    let branch = git(path, &["branch", "--show-current"])
        .await?
        .trim()
        .to_string();
    let head = git(path, &["log", "-1", "--format=%h %s"])
        .await?
        .trim()
        .to_string();
    let status = git(path, &["status", "--short"]).await?;
    let kind = [
        ("rust", path.join("Cargo.toml")),
        ("node", path.join("package.json")),
        ("python", path.join("pyproject.toml")),
        ("python", path.join("requirements.txt")),
        ("cmake", path.join("CMakeLists.txt")),
    ]
    .into_iter()
    .filter_map(|(kind, marker)| marker.exists().then_some(kind.to_string()))
    .collect::<Vec<_>>();
    let verify_command = if path.join("Cargo.toml").exists() {
        Some("cargo test --workspace".into())
    } else if path.join("package.json").exists() {
        Some("npm test".into())
    } else if path.join("pyproject.toml").exists() {
        Some("pytest".into())
    } else if path.join("CMakeLists.txt").exists() {
        Some("cmake --build build".into())
    } else {
        None
    };
    let fingerprint = format!(
        "{}\n{}\n{}",
        head,
        status.trim(),
        verify_command.as_deref().unwrap_or("")
    );
    let (health_status, evidence, last_checked_at) = if !status.trim().is_empty() {
        let evidence = inspect_dirty_changes(path).await;
        (
            "observed".into(),
            if evidence.is_empty() {
                vec!["工作区有未提交变更，已完成只读差异分析".into()]
            } else {
                evidence
            },
            Some(unix_now()),
        )
    } else {
        match (storage_dir, verify_command.as_deref()) {
            (Some(storage), Some(command)) => {
                inspect_health(path, storage, &fingerprint, command).await
            }
            _ => ("unknown".into(), Vec::new(), None),
        }
    };
    let proposals = if health_status == "failing" {
        let command = verify_command.clone().unwrap_or_default();
        let evidence = if evidence.is_empty() {
            vec![format!("验证命令失败: {}", command)]
        } else {
            evidence.clone()
        };
        vec![ProjectProposal {
            id: format!("{}-health-{}", name, short_hash(&fingerprint)),
            goal_key: "fix_project_verification_failure".into(),
            learning_questions: Vec::new(),
            impact_scope: "user_interest".into(),
            leverage_score: 0.95,
            title: "修复项目验证失败".into(),
            reason: format!(
                "项目验证命令 `{}` 未通过，已发现可复现的失败证据。",
                command
            ),
            task: format!(
                "分析并修复 `{}` 的失败；只修改必要文件，完成后重新运行该验证命令并报告剩余风险。",
                command
            ),
            evidence,
            verify_command: verify_command.clone(),
            priority: "high".into(),
            status: "proposed".into(),
            expected_value: "恢复项目验证通过，降低回归风险".into(),
            risk: "修改范围可能扩大，需要在隔离 worktree 中验证".into(),
            category: "修复".into(),
            value_score: 0.95,
            risk_score: 0.6,
            attention_cost: 0.2,
            confidence: 0.9,
            initiative: crate::initiative::decide(0.9, 0.95, 0.6, 0.2),
        }]
    } else if health_status == "observed"
        && evidence.iter().any(|item| item.starts_with("高风险差异:"))
    {
        vec![ProjectProposal {
            id: format!("{}-diff-{}", name, short_hash(&fingerprint)),
            goal_key: "review_high_risk_diff".into(),
            learning_questions: vec!["该高风险差异是否对应用户预期行为，最小可复现路径是什么？".into()],
            impact_scope: "user_interest".into(), leverage_score: 0.82,
            title: "审阅高风险代码差异".into(),
            reason: "只读差异分析发现了需要人工确认的具体风险。".into(),
            task: "审阅提案中的差异证据，确认风险是否真实；如确认，提出最小修复并补充验证。不要直接提交或推送。".into(),
            evidence: evidence.clone(),
            verify_command: None,
            priority: "high".into(),
            status: "proposed".into(),
            expected_value: "避免把高风险差异带入后续提交".into(),
            risk: "需要人工确认差异上下文，不能仅凭标记修改".into(),
            category: "质量".into(),
            value_score: 0.8,
            risk_score: 0.7,
            attention_cost: 0.4,
            confidence: 0.7,
            initiative: crate::initiative::decide(0.7, 0.8, 0.7, 0.4),
        }]
    } else {
        let mut maintenance = Vec::new();
        let todo_count = count_markers(path).await;
        let has_readme = path.join("README.md").exists() || path.join("README").exists();
        let has_ci = path.join(".github/workflows").is_dir()
            || path.join(".gitlab-ci.yml").exists()
            || path.join(".circleci").is_dir();
        let goal = read_bounded(
            path,
            &[".orch-goal.md", ".orch-goal", "PROJECT_GOAL.md"],
            5000,
        );
        let goal_lower = goal.to_lowercase();
        let verify_for_artifact = Some("git diff --check".to_string());
        if todo_count >= 10 {
            maintenance.push(ProjectProposal {
                id: format!("{}-todo-{}", name, short_hash(&format!("{}:{}", fingerprint, todo_count))),
                goal_key: "triage_todo_markers".into(),
                learning_questions: Vec::new(),
                impact_scope: "supporting".into(), leverage_score: 0.28,
                title: "整理积累的 TODO/FIXME 标记".into(),
                reason: format!("仓库中检测到 {} 个 TODO/FIXME/HACK 标记，维护债务已经可量化。", todo_count),
                task: "逐项确认 TODO/FIXME/HACK 的有效性，优先处理高风险项；删除过期标记并为保留项补充 issue 或测试。".into(),
                evidence: vec![format!("检测到 {} 个 TODO/FIXME/HACK 标记", todo_count)],
                verify_command: verify_command.clone(), priority: "medium".into(), status: "proposed".into(),
                expected_value: "减少隐藏维护债务，让后续自主分析更聚焦".into(), risk: "标记可能包含仍需保留的长期计划，需要逐项确认".into(), category: "维护".into(),
                value_score: 0.62, risk_score: 0.35, attention_cost: 0.45, confidence: 0.62,
                initiative: crate::initiative::decide(0.62, 0.62, 0.35, 0.45),
            });
        }
        if !has_readme {
            maintenance.push(ProjectProposal {
                id: format!("{}-readme-{}", name, short_hash(&fingerprint)), title: "补充项目 README".into(),
                goal_key: "create_project_readme".into(),
                learning_questions: Vec::new(),
                impact_scope: "supporting".into(), leverage_score: 0.45,
                reason: "仓库缺少 README，用户和协作者缺少可验证的入口说明。".into(),
                task: "根据现有代码和构建配置补充 README：项目目标、安装、运行、测试和已知限制；不要编造不存在的功能。".into(),
                evidence: vec!["未发现 README.md 或 README".into()], verify_command: verify_command.clone(), priority: "medium".into(), status: "proposed".into(),
                expected_value: "降低项目上手成本并固定真实使用路径".into(), risk: "文档需要与实际行为保持一致，必须基于仓库证据编写".into(), category: "文档".into(),
                value_score: 0.58, risk_score: 0.2, attention_cost: 0.35, confidence: 0.58,
                initiative: crate::initiative::decide(0.58, 0.58, 0.2, 0.35),
            });
        }
        if !has_ci && verify_command.is_some() {
            maintenance.push(ProjectProposal {
                id: format!("{}-ci-{}", name, short_hash(&fingerprint)),
                title: "补充持续验证配置".into(),
                goal_key: "add_continuous_integration".into(),
                learning_questions: vec!["项目需要支持哪些运行平台和最低版本？".into()],
                impact_scope: "user_interest".into(),
                leverage_score: 0.68,
                reason: "项目有可执行验证命令，但未发现 CI 配置，回归可能无法自动暴露。".into(),
                task: format!(
                    "为 `{}` 增加最小 CI 检查配置，只运行已有验证命令并记录平台假设。",
                    verify_command.as_deref().unwrap_or("项目验证")
                ),
                evidence: vec![
                    format!("已有验证命令: {}", verify_command.as_deref().unwrap_or("")),
                    "未发现常见 CI 配置".into(),
                ],
                verify_command: verify_command.clone(),
                priority: "low".into(),
                status: "proposed".into(),
                expected_value: "让基础验证在每次变更时自动执行".into(),
                risk: "CI 平台和依赖缓存需要按项目环境调整".into(),
                category: "工程化".into(),
                value_score: 0.55,
                risk_score: 0.3,
                attention_cost: 0.4,
                confidence: 0.55,
                initiative: crate::initiative::decide(0.55, 0.55, 0.3, 0.4),
            });
        }
        let content_or_growth_goal = [
            "内容",
            "运营",
            "增长",
            "seo",
            "marketing",
            "content",
            "用户获取",
        ]
        .iter()
        .any(|term| goal_lower.contains(term));
        if content_or_growth_goal {
            let evidence = if goal.trim().is_empty() {
                vec!["项目目标包含内容/运营/增长方向".into()]
            } else {
                vec![format!(
                    "项目目标: {}",
                    goal.lines().next().unwrap_or(&goal)
                )]
            };
            maintenance.push(ProjectProposal {
                id: format!("{}-content-opportunity-{}", name, short_hash(&fingerprint)),
                goal_key: "create_content_growth_experiment".into(),
                learning_questions: vec!["目标用户最需要解决的核心场景是什么，使用什么真实指标判断内容实验有效？".into()],
                impact_scope: "differentiator".into(), leverage_score: 0.76,
                title: "形成可执行的内容与用户增长实验".into(),
                reason: "项目目标明确涉及内容、运营或增长，但当前系统尚未把目标转成可验收的内容资产和实验。".into(),
                task: "基于项目目标、现有 README、代码和内容目录，生成一份最小内容/增长实验方案：目标用户、3 个主题或功能入口、验收指标、发布顺序；将结果写入 docs/content-experiments.md，不要编造不存在的用户数据。".into(),
                evidence,
                verify_command: verify_for_artifact.clone(),
                priority: "medium".into(), status: "proposed".into(),
                expected_value: "把模糊的运营目标转成可执行、可评估的项目产物".into(),
                risk: "方案必须标注假设，不能把推测当作真实用户反馈".into(),
                category: "content".into(), value_score: 0.72, risk_score: 0.3, attention_cost: 0.4, confidence: 0.72,
                initiative: crate::initiative::decide(0.72, 0.72, 0.3, 0.4),
            });
        }
        let feature_goal = ["功能", "feature", "用户需求", "产品", "roadmap"]
            .iter()
            .any(|term| goal_lower.contains(term));
        if feature_goal {
            let evidence = if goal.trim().is_empty() {
                vec!["项目目标包含产品或功能方向".into()]
            } else {
                vec![format!(
                    "项目目标: {}",
                    goal.lines().next().unwrap_or(&goal)
                )]
            };
            maintenance.push(ProjectProposal {
                id: format!("{}-feature-experiment-{}", name, short_hash(&fingerprint)),
                goal_key: "explore_minimum_feature_experiment".into(),
                learning_questions: vec!["哪个用户场景最值得作为首个功能实验，成功标准是什么？".into()],
                impact_scope: "differentiator".into(), leverage_score: 0.82,
                title: "探索一个最小可验证功能".into(),
                reason: "项目目标包含产品/功能方向，适合先形成小范围方案并在隔离环境验证。".into(),
                task: "根据项目目标、现有模块和最近提交，提出 2 个最小功能方案，选择一个低风险方案，在 docs/feature-experiment.md 中写出用户场景、交互流程、实现边界和验证指标；不要直接扩展无证据的范围。".into(),
                evidence,
                verify_command: verify_for_artifact.clone(),
                priority: "medium".into(), status: "proposed".into(),
                expected_value: "把新功能构想变成可评审、可验证的最小实验".into(),
                risk: "功能方案仍是假设，必须经过用户确认和真实指标验证".into(),
                category: "feature".into(), value_score: 0.75, risk_score: 0.4, attention_cost: 0.5, confidence: 0.7,
                initiative: crate::initiative::decide(0.7, 0.75, 0.4, 0.5),
            });
        }
        let has_test_dir = path.join("tests").is_dir()
            || path.join("__tests__").is_dir()
            || path.join("test").is_dir();
        if verify_command.is_some()
            && !has_ci
            && !has_test_dir
            && (path.join("Cargo.toml").exists()
                || path.join("package.json").exists()
                || path.join("pyproject.toml").exists())
        {
            maintenance.push(ProjectProposal {
                id: format!("{}-test-gap-{}", name, short_hash(&fingerprint)),
                goal_key: "add_critical_path_regression_test".into(),
                learning_questions: vec!["哪个现有用户路径发生回归时损失最大？".into()],
                impact_scope: "core_frequency".into(), leverage_score: 0.78,
                title: "补充关键路径回归测试".into(),
                reason: "项目有可执行验证命令，但未发现独立测试目录，回归风险难以定位。".into(),
                task: format!("从项目入口和最近提交中选择一个关键路径，补充最小回归测试，并运行 `{}` 验证；只覆盖仓库中已有行为。", verify_command.as_deref().unwrap_or("项目验证")),
                evidence: vec!["未发现 tests、test 或 __tests__ 目录".into(), format!("已有验证命令: {}", verify_command.as_deref().unwrap_or(""))],
                verify_command: verify_command.clone(), priority: "medium".into(), status: "proposed".into(),
                expected_value: "让真实回归更早暴露并为后续自主修改提供安全网".into(),
                risk: "测试不能把当前错误行为固化为正确行为，需要先确认关键路径".into(),
                category: "test".into(), value_score: 0.7, risk_score: 0.35, attention_cost: 0.45, confidence: 0.68,
                initiative: crate::initiative::decide(0.68, 0.7, 0.35, 0.45),
            });
        }
        maintenance
    };
    Ok(DiscoveredProject {
        name,
        path: path.display().to_string(),
        branch,
        head,
        dirty: !status.trim().is_empty(),
        changed_files: status.lines().count(),
        kind,
        verify_command,
        health_status,
        evidence,
        last_checked_at,
        proposals,
        memory: ProjectMemorySummary::default(),
    })
}

/// Read-only analysis for dirty worktrees. It never runs project code and only
/// reports concrete added-line evidence that can justify an active proposal.
async fn inspect_dirty_changes(path: &Path) -> Vec<String> {
    let mut evidence = Vec::new();
    if let Ok(check) = git(path, &["diff", "--check"]).await {
        if !check.trim().is_empty() {
            evidence.push(format!(
                "高风险差异: Git 空白检查失败: {}",
                summarize_output(&check, "")
            ));
        }
    }
    let diff = match git(path, &["diff", "--unified=0"]).await {
        Ok(value) => value,
        Err(_) => return evidence,
    };
    let mut added = 0usize;
    for line in diff.lines() {
        if !line.starts_with('+') || line.starts_with("+++") {
            continue;
        }
        added += 1;
        let content = line[1..].trim();
        if content.contains("<<<<<<<") || content.contains(">>>>>>>") || content == "=======" {
            evidence.push(format!("高风险差异: 发现未解决的合并冲突标记: {}", content));
        }
        if (content.contains("TODO") || content.contains("FIXME") || content.contains("HACK"))
            && (content.contains("security")
                || content.contains("panic")
                || content.contains("temporary")
                || content.contains("临时"))
        {
            evidence.push(format!("高风险差异: 新增风险标记: {}", content));
        }
        if evidence.len() >= 5 {
            break;
        }
    }
    if added > 0 && evidence.is_empty() {
        evidence.push(format!(
            "只读差异分析完成: 发现 {} 行新增内容，未发现明确高风险标记",
            added
        ));
    }
    evidence
}

async fn count_markers(path: &Path) -> usize {
    tokio::process::Command::new("rg")
        .args([
            "-n",
            "--hidden",
            "-g",
            "!.git",
            "-g",
            "!node_modules",
            "-g",
            "!target",
            "-g",
            "!.venv",
            "-g",
            "!dist",
            "-g",
            "!build",
            "TODO|FIXME|HACK",
        ])
        .current_dir(path)
        .output()
        .await
        .map(|output| String::from_utf8_lossy(&output.stdout).lines().count())
        .unwrap_or_default()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HealthCache {
    fingerprint: String,
    status: String,
    evidence: Vec<String>,
    checked_at: Option<u64>,
    checking_started_at: Option<u64>,
}

async fn inspect_health(
    path: &Path,
    storage_dir: &Path,
    fingerprint: &str,
    command: &str,
) -> (String, Vec<String>, Option<u64>) {
    let cache_dir = storage_dir.join("project_health");
    if std::fs::create_dir_all(&cache_dir).is_err() {
        return ("unknown".into(), vec!["无法创建验证缓存目录".into()], None);
    }
    let cache_path = cache_dir.join(format!("{}.json", short_hash(&path.display().to_string())));
    let now = unix_now();
    let ttl = std::env::var("ORCH_PROJECT_VERIFY_TTL_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
    if let Ok(content) = std::fs::read_to_string(&cache_path) {
        if let Ok(cache) = serde_json::from_str::<HealthCache>(&content) {
            if cache.fingerprint == fingerprint {
                if let Some(started) = cache.checking_started_at {
                    if now.saturating_sub(started) < 900 {
                        return ("checking".into(), cache.evidence, cache.checked_at);
                    }
                }
                if let Some(checked) = cache.checked_at {
                    if now.saturating_sub(checked) < ttl {
                        return (cache.status, cache.evidence, cache.checked_at);
                    }
                }
            }
        }
    }

    // Mark the fingerprint as in-flight before awaiting the command. This keeps
    // the Electron polling loop from launching duplicate test processes.
    let checking = HealthCache {
        fingerprint: fingerprint.into(),
        status: "checking".into(),
        evidence: vec![format!("正在运行验证命令: {}", command)],
        checked_at: None,
        checking_started_at: Some(now),
    };
    let _ = std::fs::write(
        &cache_path,
        serde_json::to_vec_pretty(&checking).unwrap_or_default(),
    );

    let path = path.to_path_buf();
    let cache_path_for_job = cache_path.clone();
    let fingerprint_for_job = fingerprint.to_string();
    let command_for_job = command.to_string();
    let verification_worktree = storage_dir.join("worktrees").join(format!(
        "verify-{}",
        short_hash(&path.display().to_string())
    ));
    let worktree_for_job = verification_worktree.clone();
    let _ = std::fs::create_dir_all(worktree_for_job.parent().unwrap_or(storage_dir));
    let worktree_ready = git(
        &path,
        &[
            "worktree",
            "add",
            "--detach",
            worktree_for_job.to_string_lossy().as_ref(),
            "HEAD",
        ],
    )
    .await
    .is_ok();
    if !worktree_ready {
        let checked_at = unix_now();
        let cache = HealthCache {
            fingerprint: fingerprint.into(),
            status: "failing".into(),
            evidence: vec!["无法创建隔离验证 worktree，已跳过测试".into()],
            checked_at: Some(checked_at),
            checking_started_at: None,
        };
        let _ = std::fs::write(
            cache_path,
            serde_json::to_vec_pretty(&cache).unwrap_or_default(),
        );
        return ("failing".into(), cache.evidence, cache.checked_at);
    }
    tokio::spawn(async move {
        let timeout_secs = std::env::var("ORCH_PROJECT_VERIFY_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(120);
        let result = run_verification(&worktree_for_job, &command_for_job, timeout_secs).await;
        let checked_at = unix_now();
        let (status, evidence) = if result.success {
            (
                "passing".to_string(),
                vec![format!("验证通过: {}", command_for_job)],
            )
        } else {
            let summary = summarize_output(&result.stderr, &result.stdout);
            let mut evidence = vec![format!(
                "命令: {} (退出码: {})",
                command_for_job,
                result
                    .exit_code
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "无".into())
            )];
            if !summary.is_empty() {
                evidence.push(format!("输出摘要: {}", summary));
            }
            ("failing".to_string(), evidence)
        };
        let cache = HealthCache {
            fingerprint: fingerprint_for_job,
            status,
            evidence,
            checked_at: Some(checked_at),
            checking_started_at: None,
        };
        let _ = std::fs::write(
            cache_path_for_job,
            serde_json::to_vec_pretty(&cache).unwrap_or_default(),
        );
        let _ = git(
            &path,
            &[
                "worktree",
                "remove",
                "--force",
                worktree_for_job.to_string_lossy().as_ref(),
            ],
        )
        .await;
    });
    ("checking".into(), checking.evidence, None)
}

async fn run_verification(path: &Path, command: &str, timeout_secs: u64) -> CommandResult {
    let command_text = command.to_string();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        Command::new("sh")
            .args(["-lc", command])
            .current_dir(path)
            .output(),
    )
    .await;
    match result {
        Ok(Ok(output)) => CommandResult {
            command: command_text,
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code(),
        },
        Ok(Err(error)) => CommandResult {
            command: command_text,
            success: false,
            stdout: String::new(),
            stderr: format!("启动验证命令失败: {}", error),
            exit_code: None,
        },
        Err(_) => CommandResult {
            command: command_text,
            success: false,
            stdout: String::new(),
            stderr: format!("验证命令超时 ({}s)", timeout_secs),
            exit_code: None,
        },
    }
}

fn summarize_output(stderr: &str, stdout: &str) -> String {
    let text = if stderr.trim().is_empty() {
        stdout
    } else {
        stderr
    };
    let compact = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    compact.chars().take(500).collect()
}

fn short_hash(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

pub fn configured_project_roots() -> Vec<PathBuf> {
    if let Ok(raw) = std::env::var("ORCH_PROJECT_ROOTS") {
        return raw
            .split(':')
            .filter(|p| !p.trim().is_empty())
            .map(PathBuf::from)
            .collect();
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    [
        PathBuf::from(format!("{}/项目", home)),
        PathBuf::from(format!("{}/Projects", home)),
    ]
    .into_iter()
    .filter(|p| p.exists())
    .collect()
}

async fn git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

async fn run_shell(cwd: &Path, command: &str) -> Result<CommandResult, String> {
    let out = Command::new("sh")
        .args(["-lc", command])
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    Ok(CommandResult {
        command: command.into(),
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into(),
        stderr: String::from_utf8_lossy(&out.stderr).into(),
        exit_code: out.status.code(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command as StdCommand;

    #[test]
    fn pi_capability_extension_is_installed_and_refreshed() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        let extension = install_pi_capability_extension(dir.path()).expect("安装扩展");
        assert_eq!(
            fs::read_to_string(&extension).expect("读取扩展"),
            PI_CAPABILITY_EXTENSION_SOURCE
        );
        fs::write(&extension, "stale").expect("写入旧扩展");
        install_pi_capability_extension(dir.path()).expect("刷新扩展");
        assert_eq!(
            fs::read_to_string(extension).expect("读取刷新后的扩展"),
            PI_CAPABILITY_EXTENSION_SOURCE
        );
    }

    #[test]
    fn pi_runtime_bridge_scopes_inputs_and_reads_dynamic_trace() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        let worker = ProjectWorker::new(dir.path());
        let worktree = dir.path().join("worktree");
        fs::create_dir_all(&worktree).expect("创建 worktree");
        let plan = vec![(
            "cmake_ops-v2".to_string(),
            "check".to_string(),
            serde_json::json!({
                "source_dir": "/original",
                "build_dir": "/original/build"
            }),
        )];
        let bridge = worker
            .prepare_pi_runtime_bridge("task-test", &worktree, &plan)
            .expect("创建 bridge")
            .expect("bridge 应存在");
        assert_eq!(bridge.capabilities.len(), 1);
        assert_eq!(
            bridge.capabilities[0].input["source_dir"],
            worktree.display().to_string()
        );
        assert_eq!(
            bridge.capabilities[0].input["build_dir"],
            worktree.join("build").display().to_string()
        );
        fs::write(
            &bridge.trace_path,
            concat!(
                "not-json\n",
                r#"{"capability":"cmake_ops-v2","action":"check","phase":"agent_dynamic","input":{},"output_summary":"ok","success":true,"elapsed_ms":7}"#,
                "\n",
                r#"{"capability":"ignored","action":"check","phase":"baseline","input":{},"output_summary":"ok","success":true,"elapsed_ms":1}"#,
                "\n"
            ),
        )
        .expect("写入 trace");
        let trace = read_pi_capability_trace(&bridge.trace_path);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].capability, "cmake_ops-v2");
        assert_eq!(trace[0].phase, "agent_dynamic");
    }

    #[test]
    fn git_change_summary_includes_untracked_only_changes() {
        let summary = format_git_change_summary("", "", "?? RUNTIME_BRIDGE.md\n");
        assert!(!summary.is_empty());
        assert!(summary.contains("1 个未跟踪文件"));
        assert!(summary.contains("RUNTIME_BRIDGE.md"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pi_runtime_bridge_injects_extension_tools_and_environment() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        let fake_pi = dir.path().join("fake-pi");
        fs::write(
            &fake_pi,
            r#"#!/bin/sh
printf 'ARGS=%s\n' "$*"
printf 'CAPS=%s\n' "$ORCH_PI_CAPABILITIES"
printf 'WORKTREE=%s\n' "$ORCH_PI_WORKTREE"
printf 'TRACE=%s\n' "$ORCH_PI_CAPABILITY_TRACE"
printf 'DAEMON=%s\n' "$ORCH_DAEMON_URL"
"#,
        )
        .expect("写入假 pi");
        let mut permissions = fs::metadata(&fake_pi).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_pi, permissions).unwrap();
        let extension = dir.path().join("extension.ts");
        fs::write(&extension, "export default () => {};").expect("写入扩展");
        let trace_path = dir.path().join("trace.jsonl");
        let bridge = PiRuntimeBridge {
            extension: extension.clone(),
            worktree: dir.path().to_path_buf(),
            trace_path: trace_path.clone(),
            capabilities: vec![PiCapabilityToolSpec {
                capability: "git_ops-v1-v2".into(),
                action: "git_diff_analysis".into(),
                input: serde_json::json!({"repo_path": dir.path()}),
            }],
        };
        let output = run_pi_once(
            fake_pi.to_str().unwrap(),
            "provider/model",
            dir.path(),
            "test",
            3,
            Some(&bridge),
            None,
        )
        .await
        .expect("假 pi 应成功");
        assert!(output.contains("runtime_capabilities,runtime_capability_call"));
        assert!(output.contains(&format!("--extension {}", extension.display())));
        assert!(output.contains("git_ops-v1-v2"));
        assert!(output.contains(&format!("WORKTREE={}", dir.path().display())));
        assert!(output.contains(&format!("TRACE={}", trace_path.display())));
        assert!(output.contains("DAEMON=http://127.0.0.1:7331"));
    }

    #[test]
    fn pi_macos_sandbox_profile_limits_writes_to_task_paths() {
        let sandbox = PiProcessSandbox {
            worktree: PathBuf::from("/workspace/task"),
            temp_dir: PathBuf::from("/storage/pi_tmp/task"),
            trace_dir: Some(PathBuf::from("/storage/pi_capability_traces")),
        };
        let profile = pi_macos_sandbox_profile(&sandbox);
        assert!(profile.contains("(deny file-write*)"));
        assert!(profile.contains("/workspace/task"));
        assert!(profile.contains("/storage/pi_tmp/task"));
        assert!(profile.contains("/storage/pi_capability_traces"));
        assert!(!profile.contains("(deny network*)"));
        assert!(!profile.contains("/Users/zhao/Documents"));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn pi_process_sandbox_allows_worktree_and_denies_other_writes() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        let worktree = dir.path().join("worktree");
        let temp_dir = dir.path().join("pi-tmp");
        fs::create_dir_all(&worktree).expect("创建 worktree");
        fs::create_dir_all(&temp_dir).expect("创建临时目录");
        let allowed = worktree.join("allowed.txt");
        let blocked = dir.path().join("blocked.txt");
        let fake_pi = dir.path().join("fake-pi");
        fs::write(
            &fake_pi,
            format!(
                "#!/bin/sh\nprintf 'allowed' > '{}'\nprintf 'blocked' > '{}' 2>/dev/null || true\nprintf 'sandbox-ok\n'\n",
                allowed.display(),
                blocked.display()
            ),
        )
        .expect("写入假 pi");
        let mut permissions = fs::metadata(&fake_pi).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_pi, permissions).unwrap();
        let sandbox = PiProcessSandbox {
            worktree: worktree.clone(),
            temp_dir,
            trace_dir: None,
        };
        let output = run_pi_once(
            fake_pi.to_str().unwrap(),
            "provider/model",
            &worktree,
            "test",
            3,
            None,
            Some(&sandbox),
        )
        .await
        .expect("沙箱内假 pi 应成功");
        assert!(output.contains("sandbox-ok"));
        assert_eq!(fs::read_to_string(allowed).unwrap(), "allowed");
        assert!(!blocked.exists(), "沙箱不应允许写入 worktree 外部");
    }

    #[test]
    fn pi_fallback_error_classification_covers_runtime_failures() {
        assert!(pi_error_allows_model_fallback("pi 执行超时 (15s)"));
        assert!(pi_error_allows_model_fallback("退出码 143: 无错误输出"));
        assert!(pi_error_allows_model_fallback("429: authorization failed"));
        assert!(!pi_error_allows_model_fallback(
            "启动 pi 失败: No such file"
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pi_fallback_model_takes_over_after_preferred_timeout() {
        let dir = tempfile::tempdir().expect("创建临时目录");
        let fake_pi = dir.path().join("fake-pi");
        fs::write(
            &fake_pi,
            r#"#!/bin/sh
model=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--model" ]; then model="$2"; shift 2; else shift; fi
done
if [ "$model" = "slow/model" ]; then exec sleep 5; fi
printf 'fallback-ok\n'
"#,
        )
        .expect("写入假 pi");
        let mut permissions = fs::metadata(&fake_pi).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_pi, permissions).unwrap();

        let models = vec!["slow/model".to_string(), "fast/model".to_string()];
        let output = run_pi_with_models(
            fake_pi.to_str().unwrap(),
            &models,
            dir.path(),
            "test",
            3,
            1,
            None,
            None,
        )
        .await
        .expect("备用模型应接管");
        assert_eq!(output, "fallback-ok");
    }

    fn git_ok(dir: &Path, args: &[&str]) {
        let output = StdCommand::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("启动 git");
        assert!(
            output.status.success(),
            "git {:?} 失败: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[tokio::test]
    async fn ordinary_git_state_does_not_create_interruptive_proposal() {
        let dir = tempfile::tempdir().expect("创建临时项目");
        git_ok(dir.path(), &["init", "-q"]);
        git_ok(dir.path(), &["config", "user.email", "test@example.com"]);
        git_ok(dir.path(), &["config", "user.name", "Test"]);
        fs::write(dir.path().join("package.json"), "{}\n").expect("写入项目文件");
        fs::write(dir.path().join("README.md"), "# Fixture\n").expect("写入 README");
        fs::create_dir_all(dir.path().join(".github/workflows")).expect("创建 CI 目录");
        fs::write(
            dir.path().join(".github/workflows/test.yml"),
            "name: test\n",
        )
        .expect("写入 CI 配置");
        git_ok(dir.path(), &["add", "."]);
        git_ok(dir.path(), &["commit", "-qm", "initial"]);

        let clean = inspect_project(dir.path()).await.expect("扫描干净项目");
        assert!(!clean.dirty);
        assert!(clean.verify_command.is_some());
        assert!(clean.proposals.is_empty());

        fs::write(dir.path().join("package.json"), "{\"name\":\"changed\"}\n")
            .expect("修改项目文件");
        let dirty = inspect_project(dir.path()).await.expect("扫描脏项目");
        assert!(dirty.dirty);
        assert_eq!(dirty.changed_files, 1);
        assert_eq!(dirty.health_status, "observed");
        assert!(dirty.proposals.is_empty());
    }

    #[test]
    fn project_skill_candidates_follow_project_markers() {
        let dir = tempfile::tempdir().expect("创建临时项目");
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        fs::write(dir.path().join("package.json"), "{}\n").unwrap();
        let candidates = project_skill_candidates(dir.path());
        assert!(candidates.contains(&"git_ops-v1-v2".into()));
        assert!(candidates.contains(&"cargo_ops-v2".into()));
        assert!(candidates.contains(&"npm_ops-v1".into()));
        assert!(!candidates.contains(&"pip_ops-v1-v2".into()));
    }

    #[test]
    fn project_task_result_deserializes_legacy_record_without_feedback_fields() {
        let value = serde_json::json!({
            "task_id": "project-old",
            "project_path": "/tmp/project",
            "task": "test",
            "approved": true,
            "worktree": null,
            "branch": null,
            "proposal": "old",
            "agent_output": "ok",
            "verification": null,
            "git_status": "",
            "diff_stat": ""
        });
        let result: ProjectTaskResult = serde_json::from_value(value).expect("兼容旧任务记录");
        assert_eq!(result.sandbox_backend, "git_worktree");
        assert!(result.skill_candidates.is_empty());
        assert!(result.real_validation.is_none());
        assert!(result.feedback.is_none());
    }

    #[test]
    fn project_capability_matching_prefers_safe_project_action() {
        let dir = tempfile::tempdir().expect("创建临时项目");
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        assert!(is_project_safe_action("check"));
        assert!(is_project_safe_action("git_diff_analysis"));
        assert!(!is_project_safe_action("git_status_and_commit"));
        assert!(!is_project_safe_action("install_dependencies"));

        let info = crate::orchestrator::CapabilityInfo {
            name: "cargo_ops-v2".into(),
            version: "1".into(),
            actions: vec!["check".into()],
            description: "通过 cargo 检查 Rust 项目".into(),
        };
        let tokens = normalized_tokens("Rust 项目 cargo check 类型检查");
        assert!(capability_match_score(&info, "check", &tokens, dir.path()) >= 3);
    }

    #[test]
    fn opportunity_signals_cover_project_value_workflows() {
        let dir = tempfile::tempdir().expect("创建临时项目");
        fs::write(
            dir.path().join("package.json"),
            "{\"scripts\":{\"build\":\"vite build\"}}\n",
        )
        .unwrap();
        let signals = collect_opportunity_signals(
            dir.path(),
            "帮助用户增长并持续运营内容",
            "# Demo\n",
            "scripts build",
        );
        assert!(signals.iter().any(|item| item.contains("内容运营机会")));
        assert!(signals.iter().any(|item| item.contains("用户目标信号")));
        assert!(signals.iter().any(|item| item.contains("质量机会")));
    }

    #[test]
    fn project_goal_parser_requires_evidence_from_context() {
        let dir = tempfile::tempdir().expect("创建临时项目");
        fs::write(dir.path().join("README.md"), "A realtime dashboard\n").unwrap();
        let context = ProjectContext::read(dir.path(), "fp").unwrap();
        let raw = r#"[{"title":"Add tests","reason":"quality","task":"Add a smoke test","evidence":["not present"],"priority":"high"}]"#;
        let memory = ProjectMemory::default();
        assert!(parse_project_proposals(raw, dir.path(), &context, &memory).is_empty());
        let raw = r#"[{"title":"Add tests","reason":"quality","task":"Add a smoke test","evidence":["realtime dashboard"],"expected_value":"fewer regressions","risk":"low","category":"质量","priority":"high","value_score":0.9,"risk_score":0.2,"attention_cost":0.2,"verify_command":"npm test"}]"#;
        let proposals = parse_project_proposals(raw, dir.path(), &context, &memory);
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].category, "质量");
    }

    #[test]
    fn paraphrased_project_goals_are_deduplicated() {
        assert!(project_goals_are_similar(
            "编写 SSH 连接与 better-sqlite3 原生模块的集成测试",
            "使用 mock SSH server 验证连接，并验证 better-sqlite3 CRUD",
            "补充 SSH 和本地数据库集成测试",
            "创建 mock SSH server，覆盖连接流程和 better-sqlite3 的增删改查",
        ));
        assert!(!project_goals_are_similar(
            "补充 README 产品截图",
            "替换截图占位符",
            "修复 SSH 凭据明文存储",
            "使用 safeStorage 加密密码和私钥",
        ));
        assert!(project_goals_are_similar(
            "验证 macOS Intel 构建缺失与 README 宣称不一致的回归风险",
            "为 macOS dmg 添加 x64 架构",
            "排查 README 声明的 macOS Intel 支持与构建配置不一致问题",
            "确认 arm64-only 配置并决定增加 x64 或修改文档",
        ));
    }

    #[test]
    fn project_memory_roundtrip_preserves_direction_and_feedback() {
        let dir = tempfile::tempdir().expect("创建存储目录");
        let project = tempfile::tempdir().expect("创建项目目录");
        let path = project.path().display().to_string();
        let mut memory = ProjectMemory {
            project_path: path.clone(),
            vision: "稳定服务真实用户".into(),
            priorities: vec!["可靠性".into(), "体验".into()],
            ..Default::default()
        };
        memory.feedback.push(ProjectMemoryFeedback {
            task_id: "t1".into(),
            useful: true,
            note: "验证通过".into(),
            recorded_at: 1,
        });
        save_project_memory(dir.path(), &memory).unwrap();
        let loaded = load_project_memory(dir.path(), project.path());
        assert_eq!(loaded.vision, "稳定服务真实用户");
        assert_eq!(loaded.priorities.len(), 2);
        assert_eq!(loaded.feedback.len(), 1);
    }
}
