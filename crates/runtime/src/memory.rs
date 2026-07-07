use crate::genome::CapabilityGenome;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════
//  多层记忆架构
//
//  短期记忆 (ShortTermMemory)  — 当前任务上下文，进程内，不持久化
//  长期记忆 (LongTermMemory)   — 成功/失败模式，持久化到 memory.json
//  进化记忆 (EvolutionMemory)  — 基因组+谱系，持久化到 genomes.json
// ═══════════════════════════════════════════════════════════════

/// 短期记忆 — 当前任务会话的上下文
///
/// 生命周期：单次任务执行
/// 存储：进程内 HashMap
/// 用途：跨步骤上下文传递、当前任务失败模式
#[derive(Debug, Clone, Default)]
pub struct ShortTermMemory {
    /// 当前任务描述
    pub current_task: String,
    /// 已执行步骤的输出
    pub step_outputs: HashMap<String, serde_json::Value>,
    /// 当前任务失败记录
    pub task_failures: Vec<TaskFailure>,
    /// 当前会话开始时间
    pub session_start: u64,
    /// 迭代次数
    pub iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailure {
    pub step: String,
    pub capability: String,
    pub action: String,
    pub error: String,
    pub retry_count: u32,
}

impl ShortTermMemory {
    pub fn new() -> Self {
        Self {
            session_start: now_secs(),
            ..Default::default()
        }
    }

    pub fn start_task(&mut self, task: &str) {
        self.current_task = task.to_string();
        self.step_outputs.clear();
        self.task_failures.clear();
        self.iterations = 0;
        self.session_start = now_secs();
    }

    pub fn record_step_output(&mut self, step: &str, output: serde_json::Value) {
        self.step_outputs.insert(step.to_string(), output);
    }

    pub fn record_failure(&mut self, failure: TaskFailure) {
        self.task_failures.push(failure);
    }

    pub fn increment_iteration(&mut self) {
        self.iterations += 1;
    }

    pub fn failure_count(&self) -> usize {
        self.task_failures.len()
    }

    pub fn has_repeated_failure(&self, step: &str, threshold: u32) -> bool {
        self.task_failures.iter()
            .filter(|f| f.step == step)
            .count() as u32 >= threshold
    }

    /// 转为 LLM 可读的上下文
    pub fn context(&self) -> String {
        let mut s = String::new();
        if !self.current_task.is_empty() {
            s.push_str(&format!("当前任务: {}\n", self.current_task));
        }
        if !self.step_outputs.is_empty() {
            s.push_str(&format!("已完成步骤: {} 个\n", self.step_outputs.len()));
        }
        if !self.task_failures.is_empty() {
            s.push_str(&format!("当前任务失败: {} 次\n", self.task_failures.len()));
            for f in self.task_failures.iter().rev().take(3) {
                s.push_str(&format!("  • {} ({}): {}\n", f.step, f.capability, f.error));
            }
        }
        s
    }
}

/// 长期记忆 — 跨会话的成功/失败模式
///
/// 生命周期：永久
/// 存储：memory.json
/// 用途：工作流模板复用、失败教训、能力使用统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LongTermMemory {
    /// 成功的工作流模板
    pub workflow_templates: Vec<WorkflowTemplate>,
    /// 失败记录（最近 N 条）
    pub failed_attempts: Vec<FailedRecord>,
    /// 能力使用统计
    pub capability_stats: HashMap<String, CapabilityUsageStat>,
    /// 进化历史
    pub evolution_history: Vec<EvolutionRecord>,
    /// 全局统计
    pub stats: MemoryStats,
}

/// 工作流模板 — 从成功执行中学习
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub task: String,
    pub steps: Vec<TemplateStep>,
    pub success_count: u32,
    pub last_used: String,
    pub fitness: f64,
}

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

/// 能力使用统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityUsageStat {
    pub total_calls: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_latency_ms: f64,
    pub last_used: String,
}

impl CapabilityUsageStat {
    pub fn success_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.successes as f64 / self.total_calls as f64
    }
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

impl LongTermMemory {
    /// 从磁盘加载
    pub fn load(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let path = dir.join("memory.json");

        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(memory) = serde_json::from_str::<LongTermMemory>(&content) {
                tracing::info!(
                    "加载长期记忆: {} 模板, {} 失败记录, {} 能力统计",
                    memory.workflow_templates.len(),
                    memory.failed_attempts.len(),
                    memory.capability_stats.len()
                );
                return memory;
            }
        }
        LongTermMemory::default()
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
        // 保留最近 200 条
        if self.failed_attempts.len() > 200 {
            let drain_count = self.failed_attempts.len() - 200;
            self.failed_attempts.drain(0..drain_count);
        }
    }

    /// 记录能力调用
    pub fn record_capability_call(&mut self, name: &str, success: bool, latency_ms: u64) {
        let stat = self.capability_stats.entry(name.to_string())
            .or_default();
        stat.total_calls += 1;
        if success {
            stat.successes += 1;
        } else {
            stat.failures += 1;
        }
        // 滚动平均
        let n = stat.total_calls as f64;
        stat.avg_latency_ms = (stat.avg_latency_ms * (n - 1.0) + latency_ms as f64) / n;
        stat.last_used = now_string();
    }

    /// 记录进化事件
    pub fn record_evolution(&mut self, event: EvolutionRecord) {
        self.stats.total_evolution_events += 1;
        if event.event_type == "generation" || event.event_type == "mutation" {
            self.stats.total_capabilities_created += 1;
        }
        self.evolution_history.push(event);
        // 保留最近 500 条
        if self.evolution_history.len() > 500 {
            let drain_count = self.evolution_history.len() - 500;
            self.evolution_history.drain(0..drain_count);
        }
    }

    /// 查找匹配的工作流模板
    pub fn find_template(&self, task: &str) -> Option<&WorkflowTemplate> {
        if let Some(w) = self.workflow_templates.iter().find(|w| w.task == task) {
            return Some(w);
        }
        let task_lower = task.to_lowercase();
        self.workflow_templates.iter()
            .find(|w| w.task.to_lowercase().contains(&task_lower) || task_lower.contains(&w.task.to_lowercase()))
    }

    /// 获取低成功率能力（需要进化的候选）
    pub fn weak_capabilities(&self, min_calls: u64) -> Vec<(String, f64)> {
        self.capability_stats.iter()
            .filter(|(_, s)| s.total_calls >= min_calls && s.success_rate() < 0.7)
            .map(|(k, s)| (k.clone(), s.success_rate()))
            .collect()
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

        if !self.capability_stats.is_empty() {
            let total: u64 = self.capability_stats.values().map(|s| s.total_calls).sum();
            let avg_success = self.capability_stats.values()
                .map(|s| s.success_rate())
                .sum::<f64>() / self.capability_stats.len() as f64;
            s.push_str(&format!("\n能力使用统计: {} 个能力, {} 次调用, 平均成功率 {:.0}%\n",
                self.capability_stats.len(), total, avg_success * 100.0));
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

/// 进化记忆 — 基因组库的元信息
///
/// 生命周期：永久
/// 存储：genomes.json（由 EvolutionEngine 管理）
/// 用途：能力谱系、适应度历史
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionMemory {
    /// 能力基因组快照（只读引用，实际数据由 EvolutionEngine 管理）
    pub genomes: Vec<CapabilityGenome>,
    /// 进化历史摘要
    pub total_generations: u32,
    pub total_mutations: u32,
    pub total_crossovers: u32,
    pub total_eliminations: u32,
}

// ═══════════════════════════════════════════════════════════════
//  向后兼容 — PersistentMemory 作为多层记忆的统一入口
// ═══════════════════════════════════════════════════════════════

/// 持久化记忆 — 多层记忆的统一入口
///
/// 三层结构：
/// - short_term: 当前任务上下文（进程内）
/// - long_term: 成功/失败模式（memory.json）
/// - evolution: 基因组+谱系（genomes.json，由 EvolutionEngine 管理）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistentMemory {
    pub workflow_templates: Vec<WorkflowTemplate>,
    pub failed_attempts: Vec<FailedRecord>,
    pub genomes: Vec<CapabilityGenome>,
    pub evolution_history: Vec<EvolutionRecord>,
    pub stats: MemoryStats,
    #[serde(skip)]
    pub short_term: ShortTermMemory,
    #[serde(skip)]
    pub capability_stats: HashMap<String, CapabilityUsageStat>,
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
                let mut mem = memory;
                mem.short_term = ShortTermMemory::new();
                return mem;
            }
        }

        let mut mem = PersistentMemory::default();
        mem.short_term = ShortTermMemory::new();
        mem
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
        // 同时记录到短期记忆
        self.short_term.record_failure(TaskFailure {
            step: step.to_string(),
            capability: String::new(),
            action: String::new(),
            error: error.to_string(),
            retry_count: 0,
        });
        // 保留最近 200 条
        if self.failed_attempts.len() > 200 {
            let drain_count = self.failed_attempts.len() - 200;
            self.failed_attempts.drain(0..drain_count);
        }
    }

    /// 记录能力调用统计
    pub fn record_capability_call(&mut self, name: &str, success: bool, latency_ms: u64) {
        let stat = self.capability_stats.entry(name.to_string())
            .or_default();
        stat.total_calls += 1;
        if success {
            stat.successes += 1;
        } else {
            stat.failures += 1;
        }
        let n = stat.total_calls as f64;
        stat.avg_latency_ms = (stat.avg_latency_ms * (n - 1.0) + latency_ms as f64) / n;
        stat.last_used = now_string();
    }

    /// 记录进化事件
    pub fn record_evolution(&mut self, event: EvolutionRecord) {
        self.stats.total_evolution_events += 1;
        if event.event_type == "generation" || event.event_type == "mutation" {
            self.stats.total_capabilities_created += 1;
        }
        self.evolution_history.push(event);
        if self.evolution_history.len() > 500 {
            let drain_count = self.evolution_history.len() - 500;
            self.evolution_history.drain(0..drain_count);
        }
    }

    /// 查找匹配的工作流模板
    pub fn find_template(&self, task: &str) -> Option<&WorkflowTemplate> {
        if let Some(w) = self.workflow_templates.iter().find(|w| w.task == task) {
            return Some(w);
        }
        let task_lower = task.to_lowercase();
        self.workflow_templates.iter()
            .find(|w| w.task.to_lowercase().contains(&task_lower) || task_lower.contains(&w.task.to_lowercase()))
    }

    /// 获取低成功率能力
    pub fn weak_capabilities(&self, min_calls: u64) -> Vec<(String, f64)> {
        self.capability_stats.iter()
            .filter(|(_, s)| s.total_calls >= min_calls && s.success_rate() < 0.7)
            .map(|(k, s)| (k.clone(), s.success_rate()))
            .collect()
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

        if !self.capability_stats.is_empty() {
            let total: u64 = self.capability_stats.values().map(|s| s.total_calls).sum();
            let avg_success = self.capability_stats.values()
                .map(|s| s.success_rate())
                .sum::<f64>() / self.capability_stats.len() as f64;
            s.push_str(&format!("\n能力使用统计: {} 个能力, {} 次调用, 平均成功率 {:.0}%\n",
                self.capability_stats.len(), total, avg_success * 100.0));
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
        self.short_term = ShortTermMemory::new();
    }

    /// 获取短期记忆引用
    pub fn short_term(&self) -> &ShortTermMemory {
        &self.short_term
    }

    /// 获取短期记忆可变引用
    pub fn short_term_mut(&mut self) -> &mut ShortTermMemory {
        &mut self.short_term
    }
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
