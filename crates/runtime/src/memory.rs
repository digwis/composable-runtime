use crate::genome::CapabilityGenome;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// 持久化记忆 — 跨会话保存进化成果
///
/// 存储内容：
/// - 成功的工作流模板（学习成果）
/// - 失败记录（教训）
/// - 能力基因组（进化产物）
/// - 进化历史（谱系）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistentMemory {
    /// 成功的工作流模板
    pub workflow_templates: Vec<WorkflowTemplate>,
    /// 失败记录
    pub failed_attempts: Vec<FailedRecord>,
    /// 能力基因组库
    pub genomes: Vec<CapabilityGenome>,
    /// 进化历史
    pub evolution_history: Vec<EvolutionRecord>,
    /// 统计信息
    pub stats: MemoryStats,
}

/// 工作流模板 — 从成功执行中学习
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub task: String,
    pub steps: Vec<TemplateStep>,
    pub success_count: u32,
    pub last_used: String,
    /// 适应度评分
    pub fitness: f64,
}

/// 模板步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStep {
    pub name: String,
    pub capability: String,
    pub action: String,
    pub input: serde_json::Value,
}

/// 失败记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedRecord {
    pub task: String,
    pub step: String,
    pub error: String,
    pub timestamp: String,
}

/// 进化记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRecord {
    pub event_type: String,
    pub capability: String,
    pub description: String,
    pub generation: u32,
    pub timestamp: String,
}

/// 记忆统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_sessions: u32,
    pub total_tasks: u32,
    pub total_successes: u32,
    pub total_failures: u32,
    pub total_evolution_events: u32,
    pub total_capabilities_created: u32,
}

impl PersistentMemory {
    /// 从磁盘加载记忆
    pub fn load(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let path = dir.join("memory.json");
        
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(memory) = serde_json::from_str::<PersistentMemory>(&content) {
                tracing::info!("加载持久化记忆: {} 模板, {} 基因组, {} 事件",
                    memory.workflow_templates.len(),
                    memory.genomes.len(),
                    memory.evolution_history.len());
                return memory;
            }
        }

        PersistentMemory::default()
    }

    /// 保存到磁盘
    pub fn save(&self, dir: impl Into<PathBuf>) {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).ok();
        
        let path = dir.join("memory.json");
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!("保存记忆失败: {}", e);
            }
        }
    }

    /// 记录成功的工作流
    pub fn record_success(&mut self, task: &str, steps: &[TemplateStep]) {
        self.stats.total_tasks += 1;
        self.stats.total_successes += 1;

        // 查找已有模板
        let existing = self.workflow_templates.iter_mut()
            .find(|w| w.task == task);

        if let Some(w) = existing {
            w.success_count += 1;
            w.last_used = now_string();
            w.fitness = (w.success_count as f64) / (w.success_count as f64 + 1.0);
        } else {
            self.workflow_templates.push(WorkflowTemplate {
                task: task.to_string(),
                steps: steps.to_vec(),
                success_count: 1,
                last_used: now_string(),
                fitness: 0.5,
            });
        }
    }

    /// 记录失败
    pub fn record_failure(&mut self, task: &str, step: &str, error: &str) {
        self.stats.total_tasks += 1;
        self.stats.total_failures += 1;
        self.failed_attempts.push(FailedRecord {
            task: task.to_string(),
            step: step.to_string(),
            error: error.to_string(),
            timestamp: now_string(),
        });
    }

    /// 记录进化事件
    pub fn record_evolution(&mut self, event: EvolutionRecord) {
        self.stats.total_evolution_events += 1;
        if event.event_type == "generation" || event.event_type == "mutation" {
            self.stats.total_capabilities_created += 1;
        }
        self.evolution_history.push(event);
    }

    /// 查找匹配的工作流模板
    pub fn find_template(&self, task: &str) -> Option<&WorkflowTemplate> {
        // 精确匹配
        if let Some(w) = self.workflow_templates.iter().find(|w| w.task == task) {
            return Some(w);
        }
        // 关键词匹配
        let task_lower = task.to_lowercase();
        self.workflow_templates.iter()
            .find(|w| w.task.to_lowercase().contains(&task_lower) || task_lower.contains(&w.task.to_lowercase()))
    }

    /// 获取记忆摘要（给 AI 看）
    pub fn summary(&self) -> String {
        let mut s = String::new();
        
        if !self.workflow_templates.is_empty() {
            s.push_str("已学习的工作流模板:\n");
            for w in &self.workflow_templates {
                s.push_str(&format!(
                    "  • '{}' (成功 {} 次, {} 步)\n",
                    w.task, w.success_count, w.steps.len()
                ));
            }
        }

        if !self.failed_attempts.is_empty() {
            s.push_str("\n失败教训:\n");
            for f in self.failed_attempts.iter().rev().take(5) {
                s.push_str(&format!("  • '{}': {}\n", f.step, f.error));
            }
        }

        if !self.genomes.is_empty() {
            s.push_str(&format!("\n已进化能力: {} 个\n", self.genomes.len()));
        }

        if s.is_empty() {
            "（首次运行，无历史记忆）".to_string()
        } else {
            s
        }
    }

    /// 开始新会话
    pub fn new_session(&mut self) {
        self.stats.total_sessions += 1;
    }
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}
