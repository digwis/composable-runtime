use crate::evolution::EvolutionEngine;
use crate::genome::{CapabilityGenome, LlmExecutor};
use crate::memory::{PersistentMemory, TemplateStep};
use crate::orchestrator::{CapabilityInfo, Orchestrator, StepOutput};
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// AI Agent — LLM 驱动的自我进化能力编排器
///
/// 核心进化循环：
/// 1. 自省：获取所有已注册能力及其动作
/// 2. 记忆：加载持久化记忆中的工作流模板和进化历史
/// 3. 规划：LLM 根据任务+能力+记忆生成步骤计划
/// 4. 创造：如果现有能力不足，AI 生成新能力基因组
/// 5. 执行：通过 Orchestrator 逐步执行
/// 6. 观察：将执行结果反馈给 LLM
/// 7. 进化：成功的工作流保存为模板，失败记录教训
/// 8. 变异：AI 可对现有能力做变异优化
pub struct Agent {
    orchestrator: Arc<Orchestrator>,
    client: LlmClient,
    memory: PersistentMemory,
    evolution: Option<EvolutionEngine>,
    llm_executor: Option<Arc<LlmExecutor>>,
    platform: Platform,
    max_iterations: usize,
    storage_dir: String,
}

/// LLM 客户端配置
pub struct LlmClient {
    api_key: String,
    model: String,
    base_url: String,
    http: reqwest::Client,
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
    pub capabilities_created: Vec<String>,
    pub summary: String,
}

/// LLM 返回的计划
#[derive(Debug, Clone, Deserialize)]
struct LlmPlan {
    thinking: String,
    steps: Vec<serde_json::Value>,
    done: bool,
    reply: String,
    /// 新能力基因组（如果 AI 认为需要创造新能力）
    #[serde(default)]
    new_capability: Option<serde_json::Value>,
    /// 变异请求（如果 AI 认为需要变异现有能力）
    #[serde(default)]
    mutate: Option<MutateRequest>,
}

/// AI 发起的变异请求
#[derive(Debug, Clone, Deserialize)]
struct MutateRequest {
    capability: String,
    action: Option<String>,
    new_prompt: Option<String>,
    new_description: Option<String>,
    new_model: Option<String>,
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
        let platform = Platform::detect();
        let storage_dir = platform.storage_dir();
        Self {
            orchestrator: Arc::new(orchestrator),
            client: LlmClient::new(api_key),
            memory: PersistentMemory::load(&storage_dir),
            evolution: None,
            llm_executor: None,
            platform,
            max_iterations: 10,
            storage_dir,
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

    /// 设置 base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.client.base_url = url.into();
        self
    }

    /// 启用进化引擎
    pub fn with_evolution(mut self) -> Self {
        let llm = Arc::new(LlmExecutor::new(
            self.client.api_key.clone(),
            self.client.base_url.clone(),
        ));
        self.llm_executor = Some(llm.clone());
        self.evolution = Some(EvolutionEngine::new(&self.storage_dir).with_llm_executor(llm));
        self
    }

    /// 执行用户任务 — Plan-Create-Execute-Observe-Evolve 循环
    pub async fn run(&mut self, task: &str) -> anyhow::Result<AgentResult> {
        println!("\n🤖 AI Agent 启动");
        println!("   任务: {}", task);
        println!("   平台: {} ({})\n", self.platform.os, self.platform.arch);

        self.memory.new_session();

        // 注册已加载的进化基因组到 MessageBus（检查平台兼容性）
        if let Some(evo) = &self.evolution {
            if let Some(llm) = &self.llm_executor {
                let bus = self.orchestrator.bus().clone();
                let genomes: Vec<_> = evo.genomes().values().cloned().collect();
                for genome in &genomes {
                    if genome.actions.is_empty() {
                        continue;
                    }
                    if !self.platform.is_compatible(genome) {
                        println!("  ⚠️  跳过不兼容能力: {} (平台 {} 不支持)", genome.name, self.platform.id);
                        continue;
                    }
                    let cap = crate::genome::ScriptedCapability::from_genome(genome.clone())
                        .with_llm(llm.clone())
                        .with_bus(bus.clone());
                    self.orchestrator.bus().register(Arc::new(cap)).await;
                    println!("  🧬 加载进化能力: {} ({})", genome.name, genome.action_names().join(", "));
                }
            }
        }

        // 1. 自省：获取所有能力
        let capabilities = self.orchestrator.introspect().await;
        let cap_description = self.describe_capabilities(&capabilities);

        // 2. 加载记忆 + 进化基因组
        let memory_context = self.memory.summary();
        let evolution_context = if let Some(evo) = &self.evolution {
            let mut ctx = String::from("\n已进化能力基因组:\n");
            for (name, genome) in evo.genomes() {
                ctx.push_str(&genome.describe());
            }
            ctx
        } else {
            String::new()
        };

        // 3. 构建系统提示
        let platform_context = self.platform.describe();
        let system_prompt = self.build_system_prompt(&cap_description, &memory_context, &evolution_context, &platform_context);

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
        let mut capabilities_created = Vec::new();

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

            // 🧬 创造新能力
            if let Some(genome_json) = &plan.new_capability {
                if let Some(evo) = &mut self.evolution {
                    match serde_json::from_value::<CapabilityGenome>(genome_json.clone()) {
                        Ok(genome) => {
                            let cap_name = genome.name.clone();
                            println!("  🧬 创造新能力: {} ({})", cap_name, genome.action_names().join(", "));
                            evo.register_genome(genome);
                            if let Some(llm) = &self.llm_executor {
                                let bus = self.orchestrator.bus().clone();
                                let cap = crate::genome::ScriptedCapability::from_genome(
                                    evo.genomes().get(&cap_name).unwrap().clone()
                                ).with_llm(llm.clone())
                                 .with_bus(bus);
                                self.orchestrator.bus().register(Arc::new(cap)).await;
                            }
                            capabilities_created.push(cap_name);
                            self.memory.record_evolution(crate::memory::EvolutionRecord {
                                event_type: "generation".into(),
                                capability: capabilities_created.last().unwrap().clone(),
                                description: "AI 生成新能力".into(),
                                generation: 1,
                                timestamp: now_string(),
                            });
                        }
                        Err(e) => {
                            println!("  ⚠️  能力基因组解析失败: {}", e);
                        }
                    }
                }
            }

            // 🧬 变异现有能力
            if let Some(mutate_req) = &plan.mutate {
                if let Some(evo) = &mut self.evolution {
                    if let Some(new_prompt) = &mutate_req.new_prompt {
                        if let Some(action) = &mutate_req.action {
                            let result = evo.mutate(&mutate_req.capability,
                                crate::evolution::Mutation::PromptChange {
                                    action: action.clone(),
                                    new_prompt: new_prompt.clone(),
                                });
                            if let Ok(new_genome) = result {
                                println!("  🧬 变异能力: {} → {}", mutate_req.capability, new_genome.name);
                                capabilities_created.push(new_genome.name.clone());
                            }
                        }
                    }
                    if let Some(new_desc) = &mutate_req.new_description {
                        let _ = evo.mutate(&mutate_req.capability,
                            crate::evolution::Mutation::DescriptionChange {
                                new_description: new_desc.clone(),
                            });
                    }
                }
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

                            self.memory.record_failure(task, &step_name, &err);

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
            let template_steps: Vec<TemplateStep> = all_outputs
                .iter()
                .filter(|o| o.result.is_ok())
                .map(|o| TemplateStep {
                    name: o.step.clone(),
                    capability: o.capability.clone(),
                    action: o.action.clone(),
                    input: serde_json::Value::Null,
                })
                .collect();

            self.memory.record_success(task, &template_steps);
            println!("🧠 保存工作流模板 '{}'", task);
            learned = true;
        }

        // 保存记忆到磁盘
        self.memory.save(&self.storage_dir);

        // 保存进化引擎
        if let Some(evo) = &mut self.evolution {
            // 自然选择：淘汰适应度低于 0.3 的能力
            let eliminated = evo.natural_selection(0.3);
            if !eliminated.is_empty() {
                println!("🗑️  自然选择: 淘汰 {} 个低适应度能力", eliminated.len());
            }

            // 自主进化：自省 → 归因 → 变异 → 测试 → 选择
            if let Some(llm) = &self.llm_executor {
                println!("\n🧬 自主进化循环启动...");
                let mut auto = crate::auto_evolve::AutoEvolver::new(
                    llm.clone(),
                    self.orchestrator.bus().clone(),
                    self.platform.clone(),
                );
                match auto.evolve_once(evo).await {
                    Ok(actions) => {
                        if !actions.is_empty() {
                            println!("  自主进化动作:");
                            for a in &actions {
                                println!("    • {}", a);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ⚠️  自主进化出错: {}", e);
                    }
                }
                println!("{}", auto.report());
            }
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
            capabilities_created,
            summary: final_reply,
        })
    }

    /// 获取记忆
    pub fn memory(&self) -> &PersistentMemory {
        &self.memory
    }

    /// 获取进化引擎
    pub fn evolution(&self) -> Option<&EvolutionEngine> {
        self.evolution.as_ref()
    }

    /// 获取进化引擎（可变）
    pub fn evolution_mut(&mut self) -> Option<&mut EvolutionEngine> {
        self.evolution.as_mut()
    }

    /// 获取 orchestrator 引用
    pub fn orchestrator(&self) -> &Orchestrator {
        &self.orchestrator
    }

    /// 获取平台信息
    pub fn platform(&self) -> &Platform {
        &self.platform
    }

    /// 获取 LLM 执行器
    pub fn llm_executor(&self) -> Option<&Arc<LlmExecutor>> {
        self.llm_executor.as_ref()
    }

    /// 获取进化报告
    pub fn evolution_report(&self) -> Option<String> {
        self.evolution.as_ref().map(|evo| evo.report())
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
        if let Some(template) = self.memory.find_template(task) {
            format!(
                "找到匹配的工作流模板: '{}' (成功 {} 次, {} 步)",
                template.task, template.success_count, template.steps.len()
            )
        } else {
            self.memory.summary()
        }
    }

    /// 构建系统提示
    fn build_system_prompt(&self, capabilities: &str, memory: &str, evolution: &str, platform: &str) -> String {
        format!(
            r#"你是一个运行在可组合能力运行时上的自我进化 AI Agent。

你的职责：
1. 分析用户任务，规划步骤
2. 每步调用一个能力的某个动作
3. 根据执行结果决定下一步
4. 任务完成后返回 done=true
5. 如果现有能力无法完成任务，你可以创造新能力！
6. 你可以变异现有能力来优化它

{platform}

{capabilities}

{memory}
{evolution}

规则：
- 每个 step 必须包含: name, capability, action, input
- capability 和 action 必须是上面列出的可用值，或你新创造的能力
- input 是 JSON 对象，包含该动作需要的参数
- 一次可以返回多个步骤，它们会按顺序执行
- 如果上一步失败，调整策略重试
- 变量引用: 可以用 ${{step_name.field}} 引用之前步骤的输出

创造新能力（当现有能力不足时）：
在返回中添加 "new_capability" 字段，格式为基因组。

实现方式有四种，根据场景选择：

1. Script（持久化脚本，最强大）— AI 编写代码并保存为基因组，可复用可变异。
   适合需要复杂逻辑、数据处理、算法实现的能力：
{{
  "name": "json_formatter",
  "version": "0.1.0",
  "description": "格式化 JSON 字符串",
  "actions": [
    {{
      "name": "format",
      "description": "将 JSON 字符串格式化为缩进形式",
      "input_schema": {{"properties": {{"json_str": {{"type": "string"}}}}}},
      "implementation": {{
        "type": "Script",
        "language": "python",
        "code": "import json, sys\ndata = json.loads('{{{{json_str}}}}')\nprint(json.dumps(data, indent=2, ensure_ascii=False))",
        "timeout_secs": 10
      }}
    }}
  ],
  "fitness": {{}},
  "lineage": {{}}
}}

2. Composite（组合现有能力）— 把多个原生能力步骤组合成新能力：
{{
  "name": "timestamp_writer",
  "version": "0.1.0",
  "description": "获取系统时间并写入文件",
  "actions": [
    {{
      "name": "write",
      "description": "获取时间戳并写入指定路径",
      "input_schema": {{"properties": {{"path": {{"type": "string"}}}}}},
      "implementation": {{
        "type": "Composite",
        "steps": [
          {{"name": "get_time", "capability": "shell", "action": "exec", "input": {{"command": "date", "args": ["+%Y-%m-%d %H:%M:%S"]}}}},
          {{"name": "write_file", "capability": "fs", "action": "write", "input": {{"path": "{{{{path}}}}", "content": "{{{{get_time.stdout}}}}"}}}}
        ]
      }}
    }}
  ],
  "fitness": {{}},
  "lineage": {{}}
}}

3. Native（委托原生能力）— 直接转发给已有能力：
{{
  "implementation": {{
    "type": "Native",
    "capability": "fs",
    "action": "read"
  }}
}}

4. Llm（LLM 推理）— 仅用于需要语言理解的任务：
{{
  "implementation": {{
    "type": "Llm",
    "prompt": "提示模板，用 {{{{var}}}} 插值",
    "model": "claude-sonnet-4-6",
    "system": "可选系统提示"
  }}
}}

选择建议：
- 需要复杂逻辑/算法/数据处理 → Script（代码持久化，可复用）
- 需要组合多个能力步骤 → Composite（编排现有能力）
- 需要语言理解/生成/分析 → Llm
- 只需转发到已有能力 → Native

Script 和 Composite 中可以用 {{{{var}}}} 引用输入参数，Composite 步骤间可用 {{{{step_name.field}}}} 引用前序步骤输出。

变异现有能力：
在返回中添加 "mutate" 字段：
{{
  "capability": "能力名",
  "action": "动作名",
  "new_prompt": "新提示模板"
}}

返回格式（严格 JSON）:
{{
  "thinking": "你的思考过程",
  "steps": [
    {{"name": "step-name", "capability": "cap-name", "action": "action-name", "input": {{...}}}}
  ],
  "done": false,
  "reply": "给用户的简短说明",
  "new_capability": null,
  "mutate": null
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

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}
