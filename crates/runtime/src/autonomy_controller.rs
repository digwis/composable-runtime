//! Persistent, conservative control loop for proactive workspace work.
//!
//! The controller is deliberately separate from the evolution loop. Evolution
//! improves capabilities; this loop decides when those capabilities should be
//! applied to real projects. Every decision is persisted so the UI can explain
//! what happened and the controller can avoid repeating the same interruption.

use crate::experiments::{ExperimentEngine, ExperimentRequest, ExperimentVariant};
use crate::initiative::{InitiativeAction, InitiativeDecision};
use crate::project_worker::{DiscoveredProject, ProjectProposal, ProjectWorker};
use crate::value_energy::{ValueEnergyAllocation, ValueOpportunityInput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyDecision {
    pub id: String,
    pub project_path: String,
    pub project_name: String,
    pub proposal_id: String,
    pub objective: String,
    pub fingerprint: String,
    pub action: String,
    pub status: String,
    pub confidence: f64,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub impact_scope: String,
    #[serde(default)]
    pub leverage_score: f64,
    #[serde(default)]
    pub value_energy: f64,
    #[serde(default)]
    pub energy_units: u32,
    #[serde(default)]
    pub personalized_value: f64,
    pub rationale: String,
    pub created_at: u64,
    #[serde(default)]
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyPrompt {
    pub id: String,
    pub project_path: String,
    pub project_name: String,
    pub proposal_id: String,
    #[serde(default)]
    pub goal_key: String,
    pub title: String,
    pub reason: String,
    pub task: String,
    #[serde(default)]
    pub expected_value: String,
    #[serde(default)]
    pub risk: String,
    pub verify_command: Option<String>,
    pub evidence: Vec<String>,
    pub confidence: f64,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub impact_scope: String,
    #[serde(default)]
    pub leverage_score: f64,
    #[serde(default)]
    pub value_energy: f64,
    #[serde(default)]
    pub energy_units: u32,
    pub rationale: String,
    pub status: String,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueAllocationRecord {
    pub project_path: String,
    pub project_name: String,
    pub proposal_id: String,
    pub title: String,
    pub category: String,
    #[serde(default)]
    pub impact_scope: String,
    #[serde(default)]
    pub leverage_score: f64,
    pub net_value: f64,
    pub personalized_value: f64,
    pub energy_units: u32,
    pub selected: bool,
    #[serde(default)]
    pub selection_reason: String,
    pub action: String,
    pub rationale: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyState {
    pub enabled: bool,
    pub paused: bool,
    pub last_tick_at: Option<u64>,
    pub last_error: Option<String>,
    pub decisions: Vec<AutonomyDecision>,
    pub prompts: Vec<AutonomyPrompt>,
    #[serde(default)]
    pub cooldowns: HashMap<String, u64>,
    #[serde(default)]
    pub energy_budget: u32,
    #[serde(default)]
    pub energy_spent: u32,
    #[serde(default)]
    pub exploitation_budget: u32,
    #[serde(default)]
    pub exploration_budget: u32,
    #[serde(default)]
    pub exploitation_spent: u32,
    #[serde(default)]
    pub exploration_spent: u32,
    #[serde(default)]
    pub value_allocations: Vec<ValueAllocationRecord>,
}

impl Default for AutonomyState {
    fn default() -> Self {
        Self {
            enabled: autonomy_enabled(),
            paused: false,
            last_tick_at: None,
            last_error: None,
            decisions: Vec::new(),
            prompts: Vec::new(),
            cooldowns: HashMap::new(),
            energy_budget: 0,
            energy_spent: 0,
            exploitation_budget: 0,
            exploration_budget: 0,
            exploitation_spent: 0,
            exploration_spent: 0,
            value_allocations: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct ValueCandidate {
    project: DiscoveredProject,
    proposal: ProjectProposal,
    decision: InitiativeDecision,
    allocation: ValueEnergyAllocation,
}

pub struct AutonomyController {
    storage_dir: PathBuf,
    roots: Vec<PathBuf>,
    llm: Option<Arc<dyn crate::driver::EvolutionDriver>>,
    shared: Arc<tokio::sync::Mutex<crate::daemon::SharedState>>,
    bus: Arc<crate::message_bus::MessageBus>,
    interval_secs: u64,
}

impl AutonomyController {
    pub fn new(
        storage_dir: impl Into<PathBuf>,
        roots: Vec<PathBuf>,
        llm: Option<Arc<dyn crate::driver::EvolutionDriver>>,
        shared: Arc<tokio::sync::Mutex<crate::daemon::SharedState>>,
        bus: Arc<crate::message_bus::MessageBus>,
        interval_secs: u64,
    ) -> Self {
        Self {
            storage_dir: storage_dir.into(),
            roots,
            llm,
            shared,
            bus,
            interval_secs: interval_secs.max(30),
        }
    }

    pub async fn run(self) {
        let mut first = true;
        loop {
            if !first {
                tokio::time::sleep(std::time::Duration::from_secs(self.interval_secs)).await;
            }
            first = false;
            if let Err(error) = self.tick().await {
                tracing::warn!("自主控制器本轮失败: {}", error);
                let mut state = load_state(&self.storage_dir);
                state.last_error = Some(error);
                state.last_tick_at = Some(now());
                trim_state(&mut state);
                let _ = save_state(&self.storage_dir, &state);
            }
        }
    }

    pub async fn tick(&self) -> Result<AutonomyState, String> {
        let mut state = load_state(&self.storage_dir);
        suppress_duplicate_prompts(&mut state);
        if !state.enabled || state.paused {
            state.last_tick_at = Some(now());
            save_state(&self.storage_dir, &state)?;
            return Ok(state);
        }

        // Refresh the graph first. This is read-only and gives the UI a fresh
        // snapshot even when the LLM is unavailable this round.
        let _ = crate::workspace::observe(&self.roots, &self.storage_dir).await;
        let worker = ProjectWorker::new(self.storage_dir.clone()).with_bus(self.bus.clone());
        let projects = worker
            .discover_projects_with_driver(&self.roots, self.llm.clone())
            .await;
        crate::learning_agenda::update(&self.storage_dir, &projects)
            .map_err(|error| format!("学习议程更新失败: {}", error))?;
        let max_actions = std::env::var("ORCH_AUTONOMY_MAX_ACTIONS_PER_TICK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3)
            .clamp(1, 12);
        let energy_budget = std::env::var("ORCH_VALUE_ENERGY_BUDGET")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(10)
            .clamp(1, 50);
        let cooldown = std::env::var("ORCH_AUTONOMY_COOLDOWN_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(21_600);
        let profile = crate::value_energy::load_profile(&self.storage_dir);
        let mut candidates = Vec::new();
        for project in projects {
            let memory =
                crate::project_worker::load_project_memory_for(&self.storage_dir, &project.path);
            for proposal in project
                .proposals
                .iter()
                .filter(|item| item.status == "proposed")
            {
                let fingerprint = proposal_fingerprint(&project, proposal);
                if proposal.evidence.is_empty()
                    || memory.completed_goals.iter().any(|event| {
                        event.proposal_id == proposal.id
                            || crate::project_worker::project_goals_are_similar(
                                &proposal.title,
                                &proposal.task,
                                &event.title,
                                &event.task,
                            )
                    })
                    || memory.rejected_goals.iter().any(|event| {
                        event.proposal_id == proposal.id
                            || crate::project_worker::project_goals_are_similar(
                                &proposal.title,
                                &proposal.task,
                                &event.title,
                                &event.task,
                            )
                    })
                    || state.prompts.iter().any(|prompt| {
                        prompt.project_path == project.path
                            && matches!(
                                prompt.status.as_str(),
                                "approved" | "rejected" | "dismissed"
                            )
                            && prompts_match_proposal(prompt, proposal)
                    })
                {
                    continue;
                }
                let allocation = crate::value_energy::allocate_for_project(
                    &profile,
                    &project.path,
                    ValueOpportunityInput {
                        category: &proposal.category,
                        value: proposal.value_score,
                        confidence: proposal.confidence,
                        leverage_score: crate::project_worker::effective_leverage_score(proposal),
                        risk: proposal.risk_score,
                        attention_cost: proposal.attention_cost,
                        resource_cost: estimate_resource_cost(proposal),
                    },
                );
                let decision = crate::initiative::decide(
                    proposal.confidence,
                    allocation.personalized_value,
                    proposal.risk_score,
                    proposal.attention_cost,
                );
                if let Some(existing) = state.prompts.iter_mut().find(|prompt| {
                    prompt.project_path == project.path
                        && prompt.status == "pending"
                        && prompts_match_proposal(prompt, proposal)
                }) {
                    refresh_pending_prompt(existing, proposal, &decision, &allocation);
                    continue;
                }
                if state
                    .cooldowns
                    .get(&fingerprint)
                    .is_some_and(|last| now().saturating_sub(*last) < cooldown)
                {
                    continue;
                }
                candidates.push(ValueCandidate {
                    project: project.clone(),
                    proposal: proposal.clone(),
                    decision,
                    allocation,
                });
            }
        }
        candidates.sort_by(|left, right| {
            right
                .allocation
                .net_value
                .partial_cmp(&left.allocation.net_value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.project.name.cmp(&right.project.name))
                .then_with(|| left.proposal.title.cmp(&right.proposal.title))
        });

        state.energy_budget = energy_budget;
        state.energy_spent = 0;
        state.exploitation_budget = energy_budget.saturating_mul(4) / 5;
        state.exploration_budget = energy_budget.saturating_sub(state.exploitation_budget);
        state.exploitation_spent = 0;
        state.exploration_spent = 0;
        state.value_allocations = candidates
            .iter()
            .map(|candidate| allocation_record(candidate, false))
            .collect();
        let energy_plan = select_energy_indices(
            &candidates
                .iter()
                .map(|candidate| candidate.allocation.energy_units)
                .collect::<Vec<_>>(),
            &candidates.iter().map(portfolio_key).collect::<Vec<_>>(),
            &candidates.iter().map(exploration_score).collect::<Vec<_>>(),
            energy_budget,
            max_actions,
        );
        for selection in energy_plan {
            let index = selection.index;
            let candidate = &candidates[index];
            let units = candidate.allocation.energy_units;
            state.energy_spent = state.energy_spent.saturating_add(units);
            if selection.reason == "explore" {
                state.exploration_spent = state.exploration_spent.saturating_add(units);
            } else {
                state.exploitation_spent = state.exploitation_spent.saturating_add(units);
            }
            if let Some(record) = state.value_allocations.get_mut(index) {
                record.selected = true;
                record.selection_reason = selection.reason.into();
            }
            self.handle_proposal(
                &candidate.project,
                &candidate.proposal,
                &candidate.decision,
                &candidate.allocation,
                &mut state,
            )
            .await;
        }
        state.last_tick_at = Some(now());
        state.last_error = None;
        trim_state(&mut state);
        save_state(&self.storage_dir, &state)?;
        Ok(state)
    }

    async fn handle_proposal(
        &self,
        project: &DiscoveredProject,
        proposal: &ProjectProposal,
        decision: &InitiativeDecision,
        allocation: &ValueEnergyAllocation,
        state: &mut AutonomyState,
    ) {
        let fingerprint = proposal_fingerprint(project, proposal);
        let action: String = action_name(decision.effective_action).into();
        let decision_id = format!("decision-{}", short_hash(&fingerprint));
        let mut record = AutonomyDecision {
            id: decision_id,
            project_path: project.path.clone(),
            project_name: project.name.clone(),
            proposal_id: proposal.id.clone(),
            objective: proposal.task.clone(),
            fingerprint: fingerprint.clone(),
            action: action.clone(),
            status: "observed".into(),
            confidence: decision.confidence,
            category: allocation.category.clone(),
            impact_scope: proposal.impact_scope.clone(),
            leverage_score: allocation.leverage_score,
            value_energy: allocation.net_value,
            energy_units: allocation.energy_units,
            personalized_value: allocation.personalized_value,
            rationale: format!("{}；{}", decision.rationale, allocation.rationale),
            created_at: now(),
            detail: String::new(),
        };

        match decision.effective_action {
            InitiativeAction::Observe => {
                record.detail = "证据不足，继续观察并等待新的工作区信号".into();
            }
            InitiativeAction::AskUser => {
                let prompt_id = format!("prompt-{}", short_hash(&fingerprint));
                if let Some(existing) = state
                    .prompts
                    .iter_mut()
                    .find(|prompt| prompt.id == prompt_id && prompt.status == "pending")
                {
                    refresh_pending_prompt(existing, proposal, decision, allocation);
                } else {
                    state.prompts.push(AutonomyPrompt {
                        id: prompt_id,
                        project_path: project.path.clone(),
                        project_name: project.name.clone(),
                        proposal_id: proposal.id.clone(),
                        goal_key: proposal.goal_key.clone(),
                        title: proposal.title.clone(),
                        reason: proposal.reason.clone(),
                        task: proposal.task.clone(),
                        expected_value: proposal.expected_value.clone(),
                        risk: proposal.risk.clone(),
                        verify_command: proposal.verify_command.clone(),
                        evidence: proposal.evidence.clone(),
                        confidence: decision.confidence,
                        category: allocation.category.clone(),
                        impact_scope: proposal.impact_scope.clone(),
                        leverage_score: allocation.leverage_score,
                        value_energy: allocation.net_value,
                        energy_units: allocation.energy_units,
                        rationale: format!("{}；{}", decision.rationale, allocation.rationale),
                        status: "pending".into(),
                        created_at: now(),
                        updated_at: now(),
                    });
                }
                record.status = "prompted".into();
                record.detail = "已加入主动提示队列，等待用户批准或拒绝".into();
            }
            InitiativeAction::Experiment => {
                record.status = "experiment_queued".into();
                record.detail = "将通过 Explorer 生成多方案，并在隔离 worktree 中验证".into();
                let storage = self.storage_dir.clone();
                let path = project.path.clone();
                let objective = proposal.task.clone();
                let verify_command = proposal.verify_command.clone();
                let llm = self.llm.clone();
                let max_variants = allocation.energy_units.saturating_add(1).clamp(2, 6) as usize;
                let decision_id = record.id.clone();
                let state_storage = self.storage_dir.clone();
                tokio::spawn(async move {
                    let engine = ExperimentEngine::new(storage);
                    match engine.explore(&path, &objective, llm, max_variants).await {
                        Ok(exploration) => {
                            let variants = exploration
                                .proposals
                                .into_iter()
                                .map(|item| ExperimentVariant {
                                    id: item.id,
                                    title: item.title,
                                    task: item.task,
                                })
                                .collect::<Vec<_>>();
                            if variants.len() >= 2 {
                                let request = ExperimentRequest {
                                    project_path: path,
                                    objective,
                                    variants,
                                    verify_command,
                                    benchmark_command: None,
                                };
                                let batch_id = format!("autonomy-{}", decision_id);
                                match engine.run_batch(&batch_id, &request).await {
                                    Ok(batch) => {
                                        let passed = batch
                                            .runs
                                            .iter()
                                            .filter(|run| run.status == "passed")
                                            .count();
                                        let status =
                                            if passed > 0 { "completed" } else { "failed" };
                                        let detail = format!(
                                            "实验批次 {} 完成，{} / {} 个方案通过验证",
                                            batch.batch_id,
                                            passed,
                                            batch.runs.len()
                                        );
                                        let _ = update_decision(
                                            &state_storage,
                                            &decision_id,
                                            status,
                                            &detail,
                                        );
                                    }
                                    Err(error) => {
                                        tracing::warn!("自主实验失败: {}", error);
                                        let _ = update_decision(
                                            &state_storage,
                                            &decision_id,
                                            "failed",
                                            &format!("实验启动失败: {}", error),
                                        );
                                    }
                                }
                            } else {
                                let _ = update_decision(
                                    &state_storage,
                                    &decision_id,
                                    "failed",
                                    "可验证方案不足两个，未消耗更多执行资源",
                                );
                            }
                        }
                        Err(error) => {
                            tracing::warn!("自主探索失败: {}", error);
                            let _ = update_decision(
                                &state_storage,
                                &decision_id,
                                "failed",
                                &format!("自主探索失败: {}", error),
                            );
                        }
                    }
                });
            }
            InitiativeAction::AutoExecute => {
                record.status = "execution_queued".into();
                record.detail =
                    "已加入可恢复任务队列；任务将在稳定 worktree 中执行并记录事件".into();
                let task_id = format!("project-{}", uuid::Uuid::new_v4());
                if let Err(error) = crate::durable_run::enqueue_project_run(
                    self.storage_dir.clone(),
                    self.shared.clone(),
                    self.bus.clone(),
                    crate::durable_run::ProjectRunSpec {
                        id: task_id,
                        source: "autonomy_auto_execute".into(),
                        project_path: project.path.clone(),
                        task: proposal.task.clone(),
                        proposal_id: Some(proposal.id.clone()),
                        verify_command: proposal.verify_command.clone(),
                        decision_id: Some(record.id.clone()),
                    },
                ) {
                    record.status = "failed".into();
                    record.detail = format!("持久任务入队失败: {}", error);
                }
            }
        }
        state.cooldowns.insert(fingerprint, now());
        state.decisions.push(record);
    }
}

fn proposal_fingerprint(project: &DiscoveredProject, proposal: &ProjectProposal) -> String {
    format!(
        "{}:{}",
        project.path,
        if proposal.goal_key.is_empty() {
            &proposal.id
        } else {
            &proposal.goal_key
        }
    )
}

fn prompts_match_proposal(prompt: &AutonomyPrompt, proposal: &ProjectProposal) -> bool {
    prompt.proposal_id == proposal.id
        || (!prompt.goal_key.is_empty()
            && !proposal.goal_key.is_empty()
            && prompt.goal_key == proposal.goal_key)
        || crate::project_worker::project_goals_are_similar(
            &prompt.title,
            &prompt.task,
            &proposal.title,
            &proposal.task,
        )
}

fn prompts_are_similar(left: &AutonomyPrompt, right: &AutonomyPrompt) -> bool {
    left.project_path == right.project_path
        && (left.proposal_id == right.proposal_id
            || (!left.goal_key.is_empty()
                && !right.goal_key.is_empty()
                && left.goal_key == right.goal_key)
            || crate::project_worker::project_goals_are_similar(
                &left.title,
                &left.task,
                &right.title,
                &right.task,
            ))
}

fn suppress_duplicate_prompts(state: &mut AutonomyState) {
    let snapshot = state.prompts.clone();
    for index in 0..state.prompts.len() {
        if state.prompts[index].status != "pending" {
            continue;
        }
        let current = &snapshot[index];
        let decided = snapshot.iter().any(|other| {
            matches!(other.status.as_str(), "approved" | "rejected" | "dismissed")
                && prompts_are_similar(current, other)
        });
        let newer_pending = snapshot.iter().enumerate().any(|(other_index, other)| {
            other_index > index && other.status == "pending" && prompts_are_similar(current, other)
        });
        if decided || newer_pending {
            state.prompts[index].status = "dismissed".into();
            state.prompts[index].updated_at = now();
        }
    }
}

fn refresh_pending_prompt(
    prompt: &mut AutonomyPrompt,
    proposal: &ProjectProposal,
    decision: &InitiativeDecision,
    allocation: &ValueEnergyAllocation,
) {
    prompt.reason = proposal.reason.clone();
    prompt.proposal_id = proposal.id.clone();
    prompt.goal_key = proposal.goal_key.clone();
    prompt.task = proposal.task.clone();
    prompt.expected_value = proposal.expected_value.clone();
    prompt.risk = proposal.risk.clone();
    prompt.verify_command = proposal.verify_command.clone();
    prompt.evidence = proposal.evidence.clone();
    prompt.confidence = decision.confidence;
    prompt.category = allocation.category.clone();
    prompt.impact_scope = proposal.impact_scope.clone();
    prompt.leverage_score = allocation.leverage_score;
    prompt.value_energy = allocation.net_value;
    prompt.energy_units = allocation.energy_units;
    prompt.rationale = format!("{}；{}", decision.rationale, allocation.rationale);
    prompt.updated_at = now();
}

fn estimate_resource_cost(proposal: &ProjectProposal) -> f64 {
    let category = crate::value_energy::normalize_category(&proposal.category);
    let base: f64 = match category.as_str() {
        "feature" => 0.72,
        "security" => 0.68,
        "growth" => 0.62,
        "bug" => 0.55,
        "dependency" => 0.52,
        "content" => 0.45,
        "test" => 0.42,
        "docs" => 0.32,
        _ => 0.50,
    };
    (base
        - if proposal.verify_command.is_some() {
            0.08
        } else {
            0.0
        })
    .clamp(0.0, 1.0)
}

fn allocation_record(candidate: &ValueCandidate, selected: bool) -> ValueAllocationRecord {
    ValueAllocationRecord {
        project_path: candidate.project.path.clone(),
        project_name: candidate.project.name.clone(),
        proposal_id: candidate.proposal.id.clone(),
        title: candidate.proposal.title.clone(),
        category: candidate.allocation.category.clone(),
        impact_scope: candidate.proposal.impact_scope.clone(),
        leverage_score: candidate.allocation.leverage_score,
        net_value: candidate.allocation.net_value,
        personalized_value: candidate.allocation.personalized_value,
        energy_units: candidate.allocation.energy_units,
        selected,
        selection_reason: String::new(),
        action: action_name(candidate.decision.effective_action).into(),
        rationale: candidate.allocation.rationale.clone(),
        created_at: now(),
    }
}

fn portfolio_key(candidate: &ValueCandidate) -> String {
    let title = candidate
        .proposal
        .title
        .to_lowercase()
        .chars()
        .filter(|character| character.is_alphanumeric())
        .collect::<String>();
    format!("{}:{}", candidate.allocation.category, title)
}

fn exploration_score(candidate: &ValueCandidate) -> f64 {
    let uncertainty = 1.0 - candidate.proposal.confidence.clamp(0.0, 1.0);
    let learning_bonus = if candidate.proposal.learning_questions.is_empty() {
        0.0
    } else {
        0.15
    };
    (uncertainty * 0.65 + candidate.allocation.leverage_score * 0.20 + learning_bonus)
        .clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EnergySelection {
    index: usize,
    reason: &'static str,
}

fn select_energy_indices(
    units: &[u32],
    portfolio_keys: &[String],
    exploration_scores: &[f64],
    budget: u32,
    max_actions: usize,
) -> Vec<EnergySelection> {
    let mut selected = Vec::new();
    let mut spent = 0u32;
    let mut selected_keys = std::collections::HashSet::new();
    let exploitation_budget = if max_actions > 1 {
        budget.saturating_mul(4) / 5
    } else {
        budget
    };
    let exploration_budget = budget.saturating_sub(exploitation_budget);
    for (index, units) in units.iter().copied().enumerate() {
        if selected.len() >= max_actions {
            break;
        }
        let key = portfolio_keys.get(index).cloned().unwrap_or_default();
        if units == 0
            || spent.saturating_add(units) > exploitation_budget
            || (!key.is_empty() && selected_keys.contains(&key))
        {
            continue;
        }
        spent = spent.saturating_add(units);
        if !key.is_empty() {
            selected_keys.insert(key);
        }
        selected.push(EnergySelection {
            index,
            reason: "exploit",
        });
    }

    if selected.len() < max_actions && exploration_budget > 0 {
        let exploration = units
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, units)| {
                let key = portfolio_keys.get(*index).cloned().unwrap_or_default();
                *units > 0
                    && *units <= exploration_budget
                    && !selected.iter().any(|selection| selection.index == *index)
                    && (key.is_empty() || !selected_keys.contains(&key))
            })
            .max_by(|(left, _), (right, _)| {
                exploration_scores
                    .get(*left)
                    .copied()
                    .unwrap_or_default()
                    .total_cmp(&exploration_scores.get(*right).copied().unwrap_or_default())
                    .then_with(|| right.cmp(left))
            });
        if let Some((index, units)) = exploration {
            let key = portfolio_keys.get(index).cloned().unwrap_or_default();
            spent = spent.saturating_add(units);
            if !key.is_empty() {
                selected_keys.insert(key);
            }
            selected.push(EnergySelection {
                index,
                reason: "explore",
            });
        }
    }

    // Indivisible energy units may not fit the exploration reserve. Return
    // unused capacity to exploitation instead of wasting the tick budget.
    for (index, units) in units.iter().copied().enumerate() {
        if selected.len() >= max_actions {
            break;
        }
        let key = portfolio_keys.get(index).cloned().unwrap_or_default();
        if units == 0
            || spent.saturating_add(units) > budget
            || selected.iter().any(|selection| selection.index == index)
            || (!key.is_empty() && selected_keys.contains(&key))
        {
            continue;
        }
        spent = spent.saturating_add(units);
        if !key.is_empty() {
            selected_keys.insert(key);
        }
        selected.push(EnergySelection {
            index,
            reason: "exploit",
        });
    }
    selected
}

fn action_name(action: InitiativeAction) -> &'static str {
    match action {
        InitiativeAction::Observe => "observe",
        InitiativeAction::AskUser => "ask_user",
        InitiativeAction::Experiment => "experiment",
        InitiativeAction::AutoExecute => "auto_execute",
    }
}

pub fn load_state(storage_dir: &Path) -> AutonomyState {
    std::fs::read_to_string(state_path(storage_dir))
        .ok()
        .and_then(|raw| serde_json::from_str::<AutonomyState>(&raw).ok())
        .unwrap_or_default()
}

pub fn save_state(storage_dir: &Path, state: &AutonomyState) -> Result<(), String> {
    let path = state_path(storage_dir);
    std::fs::create_dir_all(path.parent().unwrap_or(storage_dir)).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(state).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(tmp, path).map_err(|e| e.to_string())
}

pub fn set_paused(storage_dir: &Path, paused: bool) -> Result<AutonomyState, String> {
    let mut state = load_state(storage_dir);
    state.paused = paused;
    state.enabled = true;
    state.last_error = None;
    save_state(storage_dir, &state)?;
    Ok(state)
}

pub fn update_decision(
    storage_dir: &Path,
    decision_id: &str,
    status: &str,
    detail: &str,
) -> Result<AutonomyState, String> {
    let mut state = load_state(storage_dir);
    if let Some(decision) = state
        .decisions
        .iter_mut()
        .find(|decision| decision.id == decision_id)
    {
        decision.status = status.to_string();
        decision.detail = detail.to_string();
        save_state(storage_dir, &state)?;
    }
    Ok(state)
}

pub fn state_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join("autonomy").join("state.json")
}

fn trim_state(state: &mut AutonomyState) {
    if state.decisions.len() > 200 {
        let keep_from = state.decisions.len() - 200;
        state.decisions.drain(0..keep_from);
    }
    if state.prompts.len() > 100 {
        let keep_from = state.prompts.len() - 100;
        state.prompts.drain(0..keep_from);
    }
    if state.value_allocations.len() > 100 {
        state.value_allocations.truncate(100);
    }
    let cutoff = now().saturating_sub(7 * 24 * 60 * 60);
    state.cooldowns.retain(|_, timestamp| *timestamp >= cutoff);
}

fn autonomy_enabled() -> bool {
    std::env::var("ORCH_AUTONOMY_ENABLED")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

fn short_hash(value: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = AutonomyState::default();
        state.paused = true;
        save_state(dir.path(), &state).unwrap();
        assert!(load_state(dir.path()).paused);
    }

    #[test]
    fn energy_plan_respects_budget_and_order() {
        let keys = ["a", "b", "c", "d"].map(str::to_string);
        assert_eq!(
            select_energy_indices(&[5, 4, 3, 2], &keys, &[0.1, 0.2, 0.3, 0.9], 8, 3),
            vec![
                EnergySelection {
                    index: 0,
                    reason: "exploit"
                },
                EnergySelection {
                    index: 3,
                    reason: "explore"
                },
            ]
        );
        assert_eq!(
            select_energy_indices(&[0, 4, 3], &keys[..3], &[0.0, 0.1, 0.9], 10, 1),
            vec![EnergySelection {
                index: 1,
                reason: "exploit"
            }]
        );
    }

    #[test]
    fn energy_plan_avoids_duplicate_objectives_per_tick() {
        let keys = ["bug:same", "bug:same", "docs:other"].map(str::to_string);
        assert_eq!(
            select_energy_indices(&[4, 4, 3], &keys, &[0.1, 0.9, 0.2], 10, 3),
            vec![
                EnergySelection {
                    index: 0,
                    reason: "exploit"
                },
                EnergySelection {
                    index: 2,
                    reason: "exploit"
                },
            ]
        );
    }
}
