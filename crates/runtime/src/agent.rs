use crate::orchestrator::{CapabilityInfo, Orchestrator, StepOutput};
use crate::workflow::Step;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// AI Agent — LLM 驱动的能力编排器
///
/// 核心循环：
/// 1. 自省：获取所有已注册能力及其动作
/// 2. 规划：将用户意图 + 能力清单发送给 LLM，获取步骤计划
/// 3. 执行：通过 Orchestrator 逐步执行
/// 4. 观察：将执行结果反馈给 LLM
/// 5. 适应：LLM 根据结果决定下一步（继续/重试/调整/完成）
///
/// 自我进化：
/// - 成功的工作流自动保存为模板，下次遇到类似任务可复用
/// - 失败的尝试被记录，LLM 可参考避免重复错误
pub struct Agent {
    orchestrator: Arc<Orchestrator>,
    client: LlmClient,
    memory: AgentMemory,
    max_iterations: usize,
}

/// LLM 客户端配置
pub struct LlmClient {
    api_key: String,
    model: String,
    base_url: String,
    http: reqwest::Client,
}

/// Agent 记忆 — 存储成功工作流和失败记录
#[derive(Debug, Clone, Default)]
pub struct AgentMemory {
    /// 成功的工作流模板
    pub successful_workflows: Vec<WorkflowTemplate>,
    /// 失败记录
    pub failed_attempts: Vec<FailedAttempt>,
    /// 会话历史
    pub conversation: Vec<Message>,
}

/// 工作流模板 — 从成功执行中学习
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub task: String,
    pub steps: Vec<Step>,
    pub success_count: u32,
}

/// 失败记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedAttempt {
    pub task: String,
    pub step: String,
    pub error: String,
    pub timestamp: String,
}

/// 对话消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Agent 执行结果
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub task: String,
    pub success: bool,
    pub iterations: usize,
    pub outputs: Vec<StepOutput>,
    pub context: HashMap<String, serde_json::Value>,
    pub learned: bool,
    pub summary: String,
}

/// LLM 返回的计划
#[derive(Debug, Clone, Deserialize)]
struct LlmPlan {
    /// 思考过程
    thinking: String,
    /// 要执行的步骤（JSON 格式的 Step）
    steps: Vec<serde_json::Value>,
    /// 是否认为任务已完成
    done: bool,
    /// 给用户的回复
    reply: String,
}

/// Anthropic API 请求
#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<AnthropicMessage>,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

impl Agent {
    /// 创建新的 AI Agent
    pub fn new(orchestrator: Orchestrator, api_key: impl Into<String>) -> Self {
        Self {
            orchestrator: Arc::new(orchestrator),
            client: LlmClient::new(api_key),
            memory: AgentMemory::default(),
            max_iterations: 10,
        }
    }

    /// 设置最大迭代次数
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// 设置模型
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.client.model = model.into();
        self
    }

    /// 设置 base URL（支持代理或兼容 API）
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.client.base_url = url.into();
        self
    }

    /// 执行用户任务 — Plan-Execute-Observe 循环
    pub async fn run(&mut self, task: &str) -> anyhow::Result<AgentResult> {
        println!("\n🤖 AI Agent 启动");
        println!("   任务: {}\n", task);

        // 1. 自省：获取所有能力
        let capabilities = self.orchestrator.introspect().await;
        let cap_description = self.describe_capabilities(&capabilities);

        // 2. 检查记忆中是否有匹配的工作流模板
        let memory_context = self.build_memory_context(task);

        // 3. 构建系统提示
        let system_prompt = self.build_system_prompt(&cap_description, &memory_context);

        // 4. 初始化对话
        let mut context: HashMap<String, serde_json::Value> = HashMap::new();
        let mut all_outputs = Vec::new();
        let mut conversation = vec![Message {
            role: "user".into(),
            content: format!(
                "任务: {}\n\n请规划并执行步骤来完成这个任务。\
                返回 JSON 格式：{{\"thinking\": \"...\", \"steps\": [...], \"done\": false, \"reply\": \"...\"}}\n\
                每个 step 格式: {{\"name\": \"...\", \"capability\": \"...\", \"action\": \"...\", \"input\": {{...}}}}",
                task
            ),
        }];

        let mut iterations = 0;
        let mut success = false;
        let mut final_reply = String::new();

        while iterations < self.max_iterations {
            iterations += 1;
            println!("── 迭代 #{} ──", iterations);

            // 规划：调用 LLM
            let plan = self.call_llm(&system_prompt, &conversation).await?;

            println!("  💭 思考: {}", plan.thinking);

            if !plan.reply.is_empty() {
                println!("  💬 回复: {}", plan.reply);
                final_reply = plan.reply.clone();
            }

            // 先执行步骤，再检查是否完成
            if plan.steps.is_empty() && plan.done {
                success = true;
                println!("  ✅ Agent 判断任务完成\n");
                break;
            }

            if plan.steps.is_empty() {
                println!("  ⏸️  无步骤可执行，结束\n");
                break;
            }

            // 执行：逐步执行 LLM 返回的步骤
            for step_json in &plan.steps {
                let step_name = step_json
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unnamed")
                    .to_string();
                let cap_name = step_json
                    .get("capability")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                let action_name = step_json
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();

                println!("  ▶️  执行: {} -> {}:{}", step_name, cap_name, action_name);

                match self.orchestrator.execute_json(step_json.clone(), &context).await {
                    Ok((output, retries, failed)) => {
                        let retries_str = if retries > 0 {
                            format!(" (重试 {} 次)", retries)
                        } else {
                            String::new()
                        };

                        if failed > 0 {
                            let err = output.result.as_ref().err().cloned().unwrap_or_default();
                            println!("  ❌ 失败{}: {}", retries_str, err);

                            // 记录失败
                            self.memory.failed_attempts.push(FailedAttempt {
                                task: task.to_string(),
                                step: step_name.clone(),
                                error: err.clone(),
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| format!("{}ms", d.as_millis()))
                                    .unwrap_or_default(),
                            });

                            context.insert(
                                step_name.clone(),
                                serde_json::json!({"error": err}),
                            );
                        } else {
                            if let Ok(ref payload) = output.result {
                                println!(
                                    "  ✅ 成功{}: {}",
                                    retries_str,
                                    serde_json::to_string(payload).unwrap_or_default()
                                );
                                context.insert(step_name.clone(), payload.clone());
                            }
                        }

                        all_outputs.push(output);
                    }
                    Err(e) => {
                        println!("  ❌ 步骤解析失败: {}", e);
                        context.insert(
                            step_name.clone(),
                            serde_json::json!({"error": e.to_string()}),
                        );
                    }
                }
            }

            // 如果 LLM 标记完成，结束循环
            if plan.done {
                success = true;
                println!("  ✅ Agent 判断任务完成\n");
                break;
            }

            // 观察：将执行结果反馈给 LLM
            let observation = self.build_observation(&context);
            conversation.push(Message {
                role: "assistant".into(),
                content: format!(
                    "{{\"thinking\": \"{}\", \"steps\": [...], \"done\": {}, \"reply\": \"{}\"}}",
                    plan.thinking, plan.done, plan.reply
                ),
            });
            conversation.push(Message {
                role: "user".into(),
                content: format!(
                    "执行结果:\n{}\n\n请根据结果决定下一步。\
                    如果任务完成，设 done=true。\
                    如果需要调整，返回新的 steps。\
                    返回 JSON: {{\"thinking\": \"...\", \"steps\": [...], \"done\": bool, \"reply\": \"...\"}}",
                    observation
                ),
            });

            println!();
        }

        // 自我进化：成功的执行保存为模板
        let mut learned = false;
        if success {
            let template = WorkflowTemplate {
                task: task.to_string(),
                steps: all_outputs
                    .iter()
                    .filter(|o| o.result.is_ok())
                    .map(|o| {
                        Step::new(
                            &o.step,
                            &o.capability,
                            &o.action,
                            serde_json::Value::Null,
                        )
                    })
                    .collect(),
                success_count: 1,
            };

            // 检查是否已有类似模板
            let existing = self
                .memory
                .successful_workflows
                .iter_mut()
                .find(|w| w.task == task);

            if let Some(w) = existing {
                w.success_count += 1;
                println!("🧠 强化学习: 已有模板 '{}' 成功次数 +1 (总计 {})", task, w.success_count);
            } else {
                println!("🧠 新学习: 保存工作流模板 '{}'", task);
                self.memory.successful_workflows.push(template);
            }
            learned = true;
        }

        if iterations >= self.max_iterations && !success {
            final_reply = format!("达到最大迭代次数 ({})，任务未完成", self.max_iterations);
            println!("⚠️  {}", final_reply);
        }

        Ok(AgentResult {
            task: task.to_string(),
            success,
            iterations,
            outputs: all_outputs,
            context,
            learned,
            summary: final_reply,
        })
    }

    /// 获取记忆
    pub fn memory(&self) -> &AgentMemory {
        &self.memory
    }

    /// 描述能力清单（给 LLM 看）
    fn describe_capabilities(&self, caps: &[CapabilityInfo]) -> String {
        let mut desc = String::from("可用能力:\n");
        for cap in caps {
            desc.push_str(&format!(
                "  - {}: {} (动作: {})\n",
                cap.name,
                cap.description,
                cap.actions.join(", ")
            ));
        }
        desc
    }

    /// 构建记忆上下文
    fn build_memory_context(&self, task: &str) -> String {
        let mut ctx = String::new();

        if !self.memory.successful_workflows.is_empty() {
            ctx.push_str("过往成功经验:\n");
            for w in &self.memory.successful_workflows {
                let similarity = if w.task == task { "完全匹配" } else { "参考" };
                ctx.push_str(&format!(
                    "  [{}] '{}' (成功 {} 次, {} 步)\n",
                    similarity,
                    w.task,
                    w.success_count,
                    w.steps.len()
                ));
            }
        }

        if !self.memory.failed_attempts.is_empty() {
            ctx.push_str("\n过往失败教训:\n");
            for f in &self.memory.failed_attempts {
                ctx.push_str(&format!(
                    "  - 任务 '{}' 步骤 '{}': {}\n",
                    f.task, f.step, f.error
                ));
            }
        }

        if ctx.is_empty() {
            "无过往经验（首次执行）".to_string()
        } else {
            ctx
        }
    }

    /// 构建系统提示
    fn build_system_prompt(&self, capabilities: &str, memory: &str) -> String {
        format!(
            r#"你是一个运行在可组合能力运行时上的 AI Agent。

你的职责：
1. 分析用户任务，规划步骤
2. 每步调用一个能力的某个动作
3. 根据执行结果决定下一步
4. 任务完成后返回 done=true

{capabilities}

{memory}

规则：
- 每个 step 必须包含: name, capability, action, input
- capability 和 action 必须是上面列出的可用值
- input 是 JSON 对象，包含该动作需要的参数
- 一次可以返回多个步骤，它们会按顺序执行
- 如果上一步失败，调整策略重试
- 变量引用: 可以用 ${{step_name.field}} 引用之前步骤的输出

返回格式（严格 JSON）:
{{
  "thinking": "你的思考过程",
  "steps": [
    {{"name": "step-name", "capability": "cap-name", "action": "action-name", "input": {{...}}}}
  ],
  "done": false,
  "reply": "给用户的简短说明"
}}"#
        )
    }

    /// 构建观察结果（给 LLM 看）
    fn build_observation(&self, context: &HashMap<String, serde_json::Value>) -> String {
        let mut obs = String::new();
        for (k, v) in context {
            let v_str = if v.to_string().len() > 200 {
                format!("{}...(截断)", &v.to_string()[..200])
            } else {
                serde_json::to_string_pretty(v).unwrap_or_default()
            };
            obs.push_str(&format!("  {} = {}\n", k, v_str));
        }
        if obs.is_empty() {
            "（无输出）".to_string()
        } else {
            obs
        }
    }

    /// 调用 LLM
    async fn call_llm(&self, system: &str, messages: &[Message]) -> anyhow::Result<LlmPlan> {
        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let request = AnthropicRequest {
            model: self.client.model.clone(),
            max_tokens: 2048,
            system: system.to_string(),
            messages: anthropic_messages,
        };

        let url = format!("{}/v1/messages", self.client.base_url);

        let resp = self
            .client
            .http
            .post(&url)
            .header("x-api-key", &self.client.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API 错误 ({}): {}", status, body);
        }

        let anthropic_resp: AnthropicResponse = resp.json().await?;

        let text = anthropic_resp
            .content
            .iter()
            .filter(|c| c.content_type == "text")
            .filter_map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("");

        // 解析 JSON（LLM 可能返回带 markdown 代码块的 JSON）
        let json_str = extract_json(&text);
        let plan: LlmPlan = serde_json::from_str(json_str)
            .map_err(|e| {
                anyhow::anyhow!("LLM 返回解析失败: {} | 原始: {}", e, text)
            })?;

        Ok(plan)
    }
}

impl LlmClient {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "claude-sonnet-4-20250514".into(),
            base_url: "https://api.anthropic.com".into(),
            http: reqwest::Client::new(),
        }
    }
}

/// 从可能包含 markdown 代码块的文本中提取 JSON
fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // 尝试提取 ```json ... ``` 块
    if let Some(start) = trimmed.find("```json") {
        if let Some(end) = trimmed.rfind("```") {
            let inner = &trimmed[start + 7..end].trim();
            return inner;
        }
    }

    // 尝试提取 ``` ... ``` 块
    if let Some(start) = trimmed.find("```") {
        if let Some(end) = trimmed.rfind("```") {
            if end > start {
                let inner = &trimmed[start + 3..end].trim();
                return inner;
            }
        }
    }

    // 尝试直接找 JSON 对象
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                return &trimmed[start..=end];
            }
        }
    }

    trimmed
}
