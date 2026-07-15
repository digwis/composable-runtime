//! Persistent, goal-directed learning plans for real projects.
//!
//! This layer separates "what should the system learn next?" from project
//! execution. It is intentionally evidence-only: it may rank a knowledge gap,
//! but it cannot turn an unsupported idea into an executable task.

use crate::project_worker::{DiscoveredProject, ProjectMemory, ProjectProposal};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningAgendaState {
    pub projects: Vec<ProjectLearningAgenda>,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectLearningAgenda {
    pub project_path: String,
    pub project_name: String,
    pub north_star: String,
    pub milestones: Vec<String>,
    pub active_goals: Vec<LearningGoal>,
    pub knowledge_gaps: Vec<KnowledgeGap>,
    pub learned_patterns: Vec<LearnedPattern>,
    pub review_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningGoal {
    pub id: String,
    pub title: String,
    pub category: String,
    pub expected_value: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub id: String,
    pub question: String,
    pub related_goal: String,
    pub evidence: Vec<String>,
    pub confidence: f64,
    pub goal_alignment: f64,
    #[serde(default)]
    pub leverage_score: f64,
    pub information_gain: f64,
    pub reuse_probability: f64,
    pub research_cost: f64,
    pub learning_value: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pub id: String,
    pub summary: String,
    pub source: String,
    pub confidence: f64,
    pub learned_at: u64,
}

pub fn load(storage_dir: &Path) -> LearningAgendaState {
    std::fs::read_to_string(state_path(storage_dir))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn update(
    storage_dir: &Path,
    projects: &[DiscoveredProject],
) -> Result<LearningAgendaState, String> {
    let previous = load(storage_dir);
    let now = unix_now();
    let agendas = projects
        .iter()
        .map(|project| {
            let memory = crate::project_worker::load_project_memory_for(storage_dir, &project.path);
            let previous_project = previous
                .projects
                .iter()
                .find(|item| item.project_path == project.path);
            build_project_agenda(project, &memory, previous_project, now)
        })
        .collect();
    let state = LearningAgendaState {
        projects: agendas,
        updated_at: now,
    };
    save(storage_dir, &state)?;
    Ok(state)
}

fn build_project_agenda(
    project: &DiscoveredProject,
    memory: &ProjectMemory,
    previous: Option<&ProjectLearningAgenda>,
    now: u64,
) -> ProjectLearningAgenda {
    let active_goals = project
        .proposals
        .iter()
        .filter(|proposal| proposal.status == "proposed")
        .take(8)
        .map(|proposal| LearningGoal {
            id: proposal.id.clone(),
            title: proposal.title.clone(),
            category: crate::value_energy::normalize_category(&proposal.category),
            expected_value: proposal.expected_value.clone(),
            status: "active".into(),
        })
        .collect::<Vec<_>>();

    let mut knowledge_gaps = project
        .proposals
        .iter()
        .flat_map(knowledge_gaps_for)
        .collect::<Vec<_>>();
    if memory.vision.trim().is_empty() {
        knowledge_gaps.push(KnowledgeGap {
            id: format!(
                "{}-user-goal",
                crate::project_worker::project_memory_key(&project.path)
            ),
            question: "用户希望这个项目在未来阶段创造什么结果？".into(),
            related_goal: "明确项目北极星".into(),
            evidence: vec!["项目记忆中尚未设置明确愿景".into()],
            confidence: 1.0,
            goal_alignment: 1.0,
            leverage_score: 1.0,
            information_gain: 0.95,
            reuse_probability: 1.0,
            research_cost: 0.05,
            learning_value: 0.96,
            status: "needs_user".into(),
        });
    }
    knowledge_gaps.sort_by(|left, right| right.learning_value.total_cmp(&left.learning_value));
    knowledge_gaps.dedup_by(|left, right| left.question == right.question);
    knowledge_gaps.truncate(12);

    let mut learned_patterns = memory
        .completed_goals
        .iter()
        .rev()
        .take(8)
        .map(|event| LearnedPattern {
            id: event.proposal_id.clone(),
            summary: format!("{}：{}", event.title, event.task),
            source: "verified_project_outcome".into(),
            confidence: 0.8,
            learned_at: event.recorded_at,
        })
        .collect::<Vec<_>>();
    if let Some(previous) = previous {
        for pattern in &previous.learned_patterns {
            if !learned_patterns.iter().any(|item| item.id == pattern.id) {
                learned_patterns.push(pattern.clone());
            }
        }
    }
    learned_patterns.sort_by(|left, right| right.learned_at.cmp(&left.learned_at));
    learned_patterns.truncate(20);

    ProjectLearningAgenda {
        project_path: project.path.clone(),
        project_name: project.name.clone(),
        north_star: if memory.vision.trim().is_empty() {
            "待用户确认".into()
        } else {
            memory.vision.clone()
        },
        milestones: memory.priorities.clone(),
        active_goals,
        knowledge_gaps,
        learned_patterns,
        review_at: now.saturating_add(7 * 24 * 60 * 60),
        updated_at: now,
    }
}

fn knowledge_gaps_for(proposal: &ProjectProposal) -> Vec<KnowledgeGap> {
    proposal
        .learning_questions
        .iter()
        .enumerate()
        .map(|(index, question)| {
            let uncertainty = (1.0 - proposal.confidence).clamp(0.0, 1.0);
            let goal_alignment = proposal.value_score.clamp(0.0, 1.0);
            let leverage_score = crate::project_worker::effective_leverage_score(proposal);
            let information_gain = (0.55 + uncertainty * 0.45).clamp(0.0, 1.0);
            let reuse_probability = reuse_probability(&proposal.category);
            let research_cost = if proposal.verify_command.is_some() {
                0.3
            } else {
                0.5
            };
            let learning_value = (goal_alignment * 0.25
                + information_gain * 0.25
                + leverage_score * 0.25
                + reuse_probability * 0.15
                + uncertainty * 0.10
                - research_cost * 0.20)
                .clamp(0.0, 1.0);
            KnowledgeGap {
                id: format!("{}-gap-{}", proposal.id, index),
                question: question.clone(),
                related_goal: proposal.title.clone(),
                evidence: proposal.evidence.clone(),
                confidence: proposal.confidence,
                goal_alignment,
                leverage_score,
                information_gain,
                reuse_probability,
                research_cost,
                learning_value,
                status: "open".into(),
            }
        })
        .collect()
}

fn reuse_probability(category: &str) -> f64 {
    match crate::value_energy::normalize_category(category).as_str() {
        "test" | "security" | "dependency" | "maintenance" => 0.85,
        "docs" | "content" | "growth" => 0.65,
        "bug" => 0.6,
        "feature" => 0.45,
        _ => 0.5,
    }
}

fn save(storage_dir: &Path, state: &LearningAgendaState) -> Result<(), String> {
    let path = state_path(storage_dir);
    let parent = path
        .parent()
        .ok_or_else(|| "学习议程路径无父目录".to_string())?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(state).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    std::fs::rename(temporary, path).map_err(|error| error.to_string())
}

fn state_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join("learning_agenda").join("state.json")
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
    fn learning_value_rewards_information_and_reuse() {
        assert!(reuse_probability("security") > reuse_probability("feature"));
    }
}
