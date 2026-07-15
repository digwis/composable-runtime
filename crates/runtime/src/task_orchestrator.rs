//! TaskOrchestrator — 任务驱动进化的执行引擎
//!
//! 用户提交自然语言任务 → LLM 规划能力调用步骤 → 逐步执行 → 观察 → 循环
//! 执行结果 + 用户反馈 → 驱动能力进化（运行时调用统计 / 人类信号 / lessons / FailureDriver）

use crate::driver::EvolutionDriver;
use crate::evolution::EvolutionLesson;
use crate::failure_driver::FailureEvent;
use crate::message::Message;
use crate::message_bus::MessageBus;
use crate::orchestrator::CapabilityInfo;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// 任务编排器 — 持有 LLM、消息总线、进化引擎的引用
pub struct TaskOrchestrator {
    llm: Arc<dyn EvolutionDriver>,
    bus: Arc<MessageBus>,
    shared: Arc<Mutex<crate::daemon::SharedState>>,
    storage_dir: String,
}

/// 单步执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// 步骤序号（从 1 开始）
    pub step_num: usize,
    /// LLM 的思考过程
    pub thinking: String,
    /// 调用的能力名
    pub capability: String,
    /// 调用的动作名
    pub action: String,
    /// 输入参数
    pub input: Value,
    /// 执行结果
    pub output: Value,
    /// 是否成功
    pub success: bool,
    /// 错误信息（失败时）
    pub error: Option<String>,
    /// 耗时（毫秒）
    pub elapsed_ms: u64,
}

/// 任务执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// 任务 ID
    pub task_id: String,
    /// 任务描述
    pub description: String,
    /// 执行步骤
    pub steps: Vec<TaskStep>,
    /// 是否成功
    pub success: bool,
    /// LLM 的最终总结
    pub final_result: String,
    /// 总耗时（毫秒）
    pub elapsed_ms: u64,
    /// 创建时间
    pub created_at: String,
    /// 用户反馈（评价后填充）
    pub feedback: Option<TaskFeedback>,
}

/// 用户对任务结果的评价
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFeedback {
    /// 用户认为任务是否成功
    pub success: bool,
    /// 文字反馈
    pub note: String,
    /// 评价时间
    pub rated_at: String,
}

impl TaskOrchestrator {
    pub fn new(
        llm: Arc<dyn EvolutionDriver>,
        bus: Arc<MessageBus>,
        shared: Arc<Mutex<crate::daemon::SharedState>>,
        storage_dir: String,
    ) -> Self {
        Self {
            llm,
            bus,
            shared,
            storage_dir,
        }
    }

    /// 执行任务 — Plan-Execute-Observe 循环
    pub async fn run_task(&self, description: &str) -> Result<TaskResult, String> {
        let task_id = format!("task-{}-{}", now_secs(), &rand_simple_id(6));
        let created_at = now_string();
        let start = std::time::Instant::now();

        // 获取可用能力列表
        let capabilities = self.bus.introspect().await;
        let caps_desc = format_capabilities(&capabilities);

        let system_prompt = build_system_prompt(&caps_desc);

        let mut steps = Vec::new();
        let mut conversation: Vec<(String, String)> = Vec::new(); // (role, content)

        // 首条消息：任务描述
        let initial_msg = format!(
            "任务：{}\n\n请规划第一步：选择一个能力 + 动作 + 输入参数来开始解决这个任务。\n返回严格 JSON：{{\"thinking\": \"思考过程\", \"capability\": \"能力名\", \"action\": \"动作名\", \"input\": {{}}, \"done\": false, \"reply\": \"\"}}\n如果任务无需调用能力即可回答，设 done=true 并在 reply 中给出答案。",
            description
        );
        conversation.push(("user".into(), initial_msg));

        let max_steps = 10;
        let mut final_result = String::new();
        let mut success = false;

        for step_num in 1..=max_steps {
            // Plan: 调 LLM 决定下一步
            let conv_prompt = build_conversation_prompt(&conversation);
            let plan_str = self
                .llm
                .execute(&conv_prompt, "smart:task", Some(&system_prompt))
                .await
                .map_err(|e| format!("LLM 规划失败: {}", e))?;

            let plan: PlanResponse = parse_plan(&plan_str)?;

            // 如果 LLM 认为完成了
            if plan.done {
                final_result = plan.reply.clone();
                success = true;
                break;
            }

            // Execute: 调用能力
            let exec_start = std::time::Instant::now();
            let msg = Message::builder()
                .from("task_orchestrator")
                .to(&plan.capability)
                .action(&plan.action)
                .payload(plan.input.clone())
                .build();

            let (output, exec_success, error) = match self.bus.send(msg).await {
                Ok(resp) => {
                    let payload = &resp.payload;
                    let success = payload
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    if success {
                        (payload.clone(), true, None)
                    } else {
                        let err = payload
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("执行返回 success=false")
                            .to_string();
                        (payload.clone(), false, Some(err))
                    }
                }
                Err(e) => {
                    let err = format!("{:?}", e);
                    (json!({"error": &err}), false, Some(err))
                }
            };
            let elapsed_ms = exec_start.elapsed().as_millis() as u64;

            let step = TaskStep {
                step_num,
                thinking: plan.thinking.clone(),
                capability: plan.capability.clone(),
                action: plan.action.clone(),
                input: plan.input.clone(),
                output: output.clone(),
                success: exec_success,
                error: error.clone(),
                elapsed_ms,
            };
            steps.push(step);

            // Observe: 把结果反馈给 LLM
            conversation.push(("assistant".into(), plan_str));
            let observe_msg = format!(
                "步骤 {} 执行{}。结果：\n{}\n\n请根据结果决定下一步。返回相同 JSON 格式。如果任务已完成，设 done=true 并在 reply 中总结结果。",
                step_num,
                if exec_success { "成功" } else { "失败" },
                truncate_json(&output, 500)
            );
            conversation.push(("user".into(), observe_msg));

            // 失败时也记录到 FailureDriver
            if !exec_success {
                if let Ok(mut guard) = self.shared.try_lock() {
                    if let Some(fd) = guard.failure_driver.as_mut() {
                        fd.record_failure(FailureEvent {
                            task: format!("task:{}", description),
                            capability: plan.capability.clone(),
                            action: plan.action.clone(),
                            input: plan.input.clone(),
                            error: error.clone().unwrap_or_default(),
                            timestamp: now_string(),
                        });
                    }
                }
            }
        }

        // 如果循环结束仍未 done，让 LLM 做个总结
        if !success && !steps.is_empty() {
            let summary_prompt = format!(
                "任务：{}\n已执行 {} 步，最终未明确完成。请根据以上步骤总结结果（1-2 句话）。",
                description,
                steps.len()
            );
            conversation.push(("user".into(), summary_prompt));
            let conv_prompt = build_conversation_prompt(&conversation);
            if let Ok(summary) = self
                .llm
                .execute(&conv_prompt, "fast:task", Some(&system_prompt))
                .await
            {
                final_result = summary;
            } else {
                final_result = "任务未完成".into();
            }
        }

        let result = TaskResult {
            task_id: task_id.clone(),
            description: description.to_string(),
            steps,
            success,
            final_result,
            elapsed_ms: start.elapsed().as_millis() as u64,
            created_at,
            feedback: None,
        };

        // 持久化
        self.save_task(&result)?;

        Ok(result)
    }

    /// 处理用户反馈 — 驱动能力进化
    pub async fn apply_feedback(
        &self,
        task_id: &str,
        feedback: TaskFeedback,
    ) -> Result<(), String> {
        // 加载任务
        let mut task = self
            .load_task(task_id)
            .await
            .ok_or_else(|| format!("任务 {} 不存在", task_id))?;

        if task.feedback.is_some() {
            return Err(format!("任务 {} 已提交过反馈，拒绝重复计票", task_id));
        }

        let used_capabilities: std::collections::BTreeSet<String> = task
            .steps
            .iter()
            .map(|step| step.capability.clone())
            .collect();

        let note = &feedback.note;
        let success = feedback.success;

        // 锁忙时明确失败，不得像 try_lock 那样静默丢掉反馈却返回成功。
        let mut guard = tokio::time::timeout(std::time::Duration::from_secs(3), self.shared.lock())
            .await
            .map_err(|_| "进化系统正忙，反馈尚未应用，请稍后重试".to_string())?;

        // 第一次检查发生在锁外；并发请求可能同时读到 feedback=None。
        // 取锁后必须从原子任务文件重读一次，确保同一 task 只计一票。
        task = self
            .load_task(task_id)
            .await
            .ok_or_else(|| format!("任务 {} 不存在", task_id))?;
        if task.feedback.is_some() {
            return Err(format!("任务 {} 已提交过反馈，拒绝重复计票", task_id));
        }

        let previous_memory = guard.evolution.memory().clone();
        let previous_fitness = if used_capabilities.len() == 1 {
            let cap = used_capabilities.iter().next().unwrap();
            guard
                .evolution
                .genomes()
                .get(cap)
                .map(|g| (cap.clone(), g.fitness.clone()))
        } else {
            None
        };

        // 只有单能力任务能把“整项任务是否有价值”无歧义归因到具体能力。
        // 多能力任务先保留任务级 lesson；技术成败已由每个 TaskStep 的运行时调用记录。
        if used_capabilities.len() == 1 {
            let cap = used_capabilities.iter().next().unwrap();
            if let Some(g) = guard.evolution.genomes_mut().get_mut(cap) {
                g.fitness.record_human_signal(success);
            }
        }

        // 文字反馈保留任务上下文；多能力任务标成 task 级，避免伪造单能力归因。
        if !note.is_empty() {
            let capability = if used_capabilities.len() == 1 {
                used_capabilities.iter().next().cloned().unwrap_or_default()
            } else {
                format!("task:{}", task_id)
            };
            let step_summary = task
                .steps
                .iter()
                .map(|step| {
                    format!(
                        "{}.{}={}",
                        step.capability,
                        step.action,
                        if step.success { "成功" } else { "失败" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            guard.evolution.record_lesson(EvolutionLesson {
                lesson: format!(
                    "任务「{}」反馈：{}（{}；步骤：{}）",
                    task.description,
                    note,
                    if success { "成功" } else { "失败" },
                    step_summary
                ),
                capability,
                failure_type: if success {
                    "task_success".into()
                } else {
                    "task_failure".into()
                },
                learned_at: now_string(),
                referenced_count: 0,
            });
        }

        task.feedback = Some(feedback.clone());
        let persist_result = guard
            .evolution
            .save_fitness()
            .and_then(|_| guard.evolution.save_memory())
            .and_then(|_| self.save_task(&task));
        if let Err(e) = persist_result {
            if let Some((cap, fitness)) = previous_fitness {
                if let Some(g) = guard.evolution.genomes_mut().get_mut(&cap) {
                    g.fitness = fitness;
                }
            }
            *guard.evolution.memory_mut() = previous_memory;
            let _ = guard.evolution.save_fitness();
            let _ = guard.evolution.save_memory();
            return Err(format!("任务反馈持久化失败，已回滚: {}", e));
        }

        // 失败任务只把实际失败的步骤送入 FailureDriver。
        if !success {
            if let Some(fd) = guard.failure_driver.as_mut() {
                for step in &task.steps {
                    if !step.success {
                        fd.record_failure(FailureEvent {
                            task: format!("user_task:{}", task.description),
                            capability: step.capability.clone(),
                            action: step.action.clone(),
                            input: step.input.clone(),
                            error: step.error.clone().unwrap_or_default(),
                            timestamp: now_string(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// 获取最近的任务列表
    pub async fn list_tasks(&self) -> Vec<TaskResult> {
        let dir = format!("{}/tasks", self.storage_dir);
        let mut tasks = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    if let Ok(task) = serde_json::from_str::<TaskResult>(&content) {
                        tasks.push(task);
                    }
                }
            }
        }
        // 按创建时间倒序
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        tasks.truncate(50); // 最多 50 条
        tasks
    }

    fn save_task(&self, task: &TaskResult) -> Result<(), String> {
        use std::io::Write;

        let dir = format!("{}/tasks", self.storage_dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = format!("{}/{}.json", dir, task.task_id);
        let content = serde_json::to_string_pretty(task).map_err(|e| e.to_string())?;
        let tmp = format!("{}.tmp-{}", path, uuid::Uuid::new_v4());
        let result = (|| -> Result<(), String> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?;
            file.write_all(content.as_bytes())
                .map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }

    async fn load_task(&self, task_id: &str) -> Option<TaskResult> {
        let path = format!("{}/tasks/{}.json", self.storage_dir, task_id);
        let content = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

use serde_json::json;
use serde_json::Value;

/// LLM 返回的规划
#[derive(Debug, Deserialize)]
struct PlanResponse {
    thinking: String,
    capability: String,
    action: String,
    #[serde(default)]
    input: Value,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    reply: String,
}

fn parse_plan(raw: &str) -> Result<PlanResponse, String> {
    // 尝试直接 parse
    if let Ok(plan) = serde_json::from_str::<PlanResponse>(raw) {
        return Ok(plan);
    }
    // 尝试从 markdown 代码块中提取
    if let Some(start) = raw.find('{') {
        if let Some(end) = raw.rfind('}') {
            if let Ok(plan) = serde_json::from_str::<PlanResponse>(&raw[start..=end]) {
                return Ok(plan);
            }
        }
    }
    Err(format!(
        "无法解析 LLM 规划响应: {}",
        &raw[..raw.len().min(200)]
    ))
}

fn build_conversation_prompt(conversation: &[(String, String)]) -> String {
    conversation
        .iter()
        .map(|(role, content)| format!("[{}]: {}", role, content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn build_system_prompt(caps_desc: &str) -> String {
    format!(
        r#"你是一个任务编排引擎。你的工作是：根据用户的任务描述，选择合适的能力和动作来逐步完成任务。

可用能力列表：
{}

规则：
1. 每次只规划一步：选择一个能力 + 一个动作 + 输入参数
2. 返回严格 JSON：{{"thinking": "思考", "capability": "能力名", "action": "动作名", "input": {{}}, "done": false, "reply": ""}}
3. 如果任务无需调用能力，设 done=true 并在 reply 中回答
4. 如果所有必要步骤都已执行完，设 done=true 并在 reply 中总结结果
5. input 中的字段必须与能力的动作 schema 匹配
6. 如果某步失败，可以尝试换一个能力或调整输入
7. 最多 10 步，请高效规划"#,
        caps_desc
    )
}

fn format_capabilities(caps: &[CapabilityInfo]) -> String {
    caps.iter()
        .map(|c| {
            format!(
                "- {} (v{}): {} | 动作: {}",
                c.name,
                c.version,
                c.description,
                c.actions.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate_json(value: &Value, max_chars: usize) -> String {
    let s = serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".into());
    if s.len() > max_chars {
        format!("{}...(已截断)", &s[..max_chars])
    } else {
        s
    }
}

fn rand_simple_id(len: usize) -> String {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let chars = "abcdefghijklmnopqrstuvwxyz0123456789";
    let mut result = String::new();
    let mut state = seed as u64;
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (state >> 33) as usize % chars.len();
        result.push(chars.chars().nth(idx).unwrap());
    }
    result
}

/// 当前时间的 Unix 秒字符串（与 genome.rs 的 now_string 一致）
fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default()
}

/// 当前 Unix 秒（用于生成 task_id）
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task_step(capability: &str, success: bool, step_num: usize) -> TaskStep {
        TaskStep {
            step_num,
            thinking: String::new(),
            capability: capability.into(),
            action: "run".into(),
            input: json!({}),
            output: json!({"success": success}),
            success,
            error: (!success).then(|| "step failed".into()),
            elapsed_ms: 10,
        }
    }

    fn task(task_id: &str, steps: Vec<TaskStep>) -> TaskResult {
        TaskResult {
            task_id: task_id.into(),
            description: "测试任务".into(),
            success: steps.iter().all(|step| step.success),
            steps,
            final_result: String::new(),
            elapsed_ms: 10,
            created_at: now_string(),
            feedback: None,
        }
    }

    fn orchestrator_harness(
        storage: &std::path::Path,
        capabilities: &[&str],
    ) -> (TaskOrchestrator, Arc<Mutex<crate::daemon::SharedState>>) {
        let mut evolution = crate::evolution::EvolutionEngine::new(storage);
        for capability in capabilities {
            evolution
                .register_genome(crate::genome::CapabilityGenome::new(
                    *capability,
                    "feedback test",
                ))
                .unwrap();
        }
        let shared = Arc::new(Mutex::new(crate::daemon::SharedState {
            evolution,
            failure_driver: None,
            total_evolutions: 0,
        }));
        let llm = Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let orchestrator = TaskOrchestrator::new(
            llm,
            Arc::new(MessageBus::new()),
            shared.clone(),
            storage.to_string_lossy().into_owned(),
        );
        (orchestrator, shared)
    }

    #[tokio::test]
    async fn single_capability_feedback_is_persisted_once_without_fake_call() {
        let storage = tempfile::tempdir().unwrap();
        let (orchestrator, shared) = orchestrator_harness(storage.path(), &["only-cap"]);
        let task = task("single-feedback", vec![task_step("only-cap", true, 1)]);
        orchestrator.save_task(&task).unwrap();

        orchestrator
            .apply_feedback(
                &task.task_id,
                TaskFeedback {
                    success: false,
                    note: "执行成功但结果没有价值".into(),
                    rated_at: now_string(),
                },
            )
            .await
            .unwrap();

        let guard = shared.lock().await;
        let fitness = &guard.evolution.genomes()["only-cap"].fitness;
        assert_eq!(fitness.human_signals_count, 1);
        assert_eq!(fitness.human_score, 0.0);
        assert_eq!(fitness.call_count, 0, "反馈不得伪造一次能力执行");
        drop(guard);

        let reloaded = crate::evolution::EvolutionEngine::new(storage.path());
        assert_eq!(
            reloaded.genomes()["only-cap"].fitness.human_signals_count,
            1
        );
        assert!(reloaded.memory().lessons.iter().any(|lesson| {
            lesson.capability == "only-cap" && lesson.lesson.contains("没有价值")
        }));
        assert!(orchestrator
            .apply_feedback(
                &task.task_id,
                TaskFeedback {
                    success: true,
                    note: String::new(),
                    rated_at: now_string(),
                },
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn multi_capability_feedback_stays_at_task_level() {
        let storage = tempfile::tempdir().unwrap();
        let (orchestrator, shared) =
            orchestrator_harness(storage.path(), &["first-cap", "second-cap"]);
        let task = task(
            "multi-feedback",
            vec![
                task_step("first-cap", true, 1),
                task_step("second-cap", false, 2),
                task_step("first-cap", true, 3),
            ],
        );
        orchestrator.save_task(&task).unwrap();

        orchestrator
            .apply_feedback(
                &task.task_id,
                TaskFeedback {
                    success: false,
                    note: "第二步失败".into(),
                    rated_at: now_string(),
                },
            )
            .await
            .unwrap();

        let guard = shared.lock().await;
        assert_eq!(
            guard.evolution.genomes()["first-cap"]
                .fitness
                .human_signals_count,
            0
        );
        assert_eq!(
            guard.evolution.genomes()["second-cap"]
                .fitness
                .human_signals_count,
            0
        );
        assert!(guard.evolution.memory().lessons.iter().any(|lesson| {
            lesson.capability == "task:multi-feedback"
                && lesson.lesson.contains("first-cap.run=成功")
                && lesson.lesson.contains("second-cap.run=失败")
        }));
    }
}
