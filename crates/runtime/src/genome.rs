use crate::capability::Capability;
use crate::message::{Message, MessageError, MessageResult};
use crate::message_bus::MessageBus;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 能力基因组（Capability Genome）— 能力的 DNA
///
/// 这是开创性的设计：能力不再是编译好的代码，而是数据驱动的基因组。
/// AI 可以像修改 DNA 一样创造、变异、淘汰能力。
///
/// 基因组包含：
/// - 身份基因：名称、版本、描述
/// - 接口基因：动作列表
/// - 行为基因：每个动作的实现方式（LLM 调用 / 规则映射 / 组合调用）
/// - 适应度基因：成功率、调用次数、变异历史
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityGenome {
    /// 身份基因
    pub name: String,
    pub version: String,
    pub description: String,

    /// 接口基因 — 声明可用动作
    pub actions: Vec<ActionGene>,

    /// 适应度基因 — 进化评估指标
    #[serde(default)]
    pub fitness: FitnessGene,

    /// 谱系基因 — 进化历史
    #[serde(default)]
    pub lineage: LineageGene,
}

/// 动作基因 — 描述一个动作的接口和实现
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGene {
    /// 动作名称
    pub name: String,
    /// 动作描述（给 AI 看）
    pub description: String,
    /// 输入参数模式（JSON Schema 风格）
    #[serde(default)]
    pub input_schema: serde_json::Value,
    /// 实现方式
    pub implementation: ActionImpl,
}

/// 动作实现方式 — 决定动作如何被执行
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionImpl {
    /// LLM 调用 — 用大语言模型执行
    Llm {
        /// 提示模板，支持 {{var}} 变量插值
        prompt: String,
        /// 模型名称
        #[serde(default = "default_model")]
        model: String,
        /// 系统提示
        #[serde(default)]
        system: Option<String>,
    },
    /// 规则映射 — 简单的输入输出映射
    Rule {
        /// JSON 映射规则
        /// 支持模板: {"result": "{{a}} + {{b}}"}
        template: serde_json::Value,
    },
    /// 组合调用 — 调用其他能力组合完成
    Composite {
        /// 子步骤（引用其他能力）
        steps: Vec<CompositeStep>,
    },
    /// 原生代码 — 由 Rust 代码实现（不可变异）
    Native {
        /// 原生能力名称
        capability: String,
        /// 原生动作名称
        action: String,
    },
    /// 脚本能力 — AI 编写的代码持久化为基因组，可复用可变异
    ///
    /// 这是 AI "长出新器官" 的关键机制：
    /// AI 编写 Python/Node 代码，保存为基因组，
    /// 下次直接调用，不需要重写。
    Script {
        /// 脚本语言: "python" 或 "node"
        language: String,
        /// 脚本代码（支持 {{var}} 模板插值，变量来自输入参数）
        code: String,
        /// 执行超时（秒）
        #[serde(default = "default_timeout")]
        timeout_secs: u64,
    },
    /// 自定义执行器 — 引用 ExecutorRegistry 中动态注册的执行器类型
    ///
    /// 这是元进化的关键：系统可以在运行时创造新的执行器类型，
    /// 而不需要修改 Rust 代码。Custom 变体打破了内置 5 种类型的硬编码限制。
    ///
    /// executor_type 引用 ExecutorRegistry 中注册的类型名（如 "cached_script"），
    /// params 是该执行器特有的参数（由执行器的 params_schema 定义）。
    Custom {
        /// 执行器类型名（在 ExecutorRegistry 中注册）
        executor_type: String,
        /// 执行器参数（由执行器的 params_schema 定义）
        #[serde(default)]
        params: serde_json::Value,
    },
}

fn default_model() -> String {
    "claude-sonnet-4-6".into()
}

fn default_timeout() -> u64 {
    30
}

/// 组合步骤 — 引用其他能力的动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeStep {
    pub name: String,
    pub capability: String,
    pub action: String,
    pub input: serde_json::Value,
}

/// 适应度基因 — 衡量能力的进化适应性
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FitnessGene {
    /// 总调用次数（含自测试）
    pub call_count: u32,
    /// 自动测试调用次数（不含真实业务调用）
    ///
    /// 区分"自测试"与"真实业务调用"是负反馈的关键：
    /// 自测试只能证明能力"能跑通"，不能证明能力"有用"。
    /// 只有真实业务调用才能让能力免于淘汰。
    #[serde(default)]
    pub auto_test_count: u32,
    /// 成功次数
    pub success_count: u32,
    /// 失败次数
    pub failure_count: u32,
    /// 成功率（0.0 ~ 1.0）
    pub success_rate: f64,
    /// 平均执行时间（毫秒）
    pub avg_latency_ms: f64,
    /// 适应度评分（综合指标）
    pub score: f64,
    /// 最后评估时间
    pub last_evaluated: Option<String>,
    /// 连续休眠轮数（未被真实业务调用）
    ///
    /// 注意：休眠判断基于 real_call_count，而非 call_count。
    /// 这样自测试过的能力如果没有真实业务调用，仍会被淘汰。
    #[serde(default)]
    pub rounds_dormant: u32,
}

impl FitnessGene {
    /// 真实业务调用次数 = 总调用 - 自测试调用
    pub fn real_call_count(&self) -> u32 {
        self.call_count.saturating_sub(self.auto_test_count)
    }

    /// 记录一次真实业务调用
    ///
    /// 真实业务调用是能力的"生存证明"——只有真实调用才能让能力免于淘汰，
    /// 并通过 speed_factor 完整计算适应度分数。
    pub fn record_real_call(&mut self, success: bool, latency_ms: f64) {
        self.call_count += 1;
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        // 滚动平均延迟
        let n = self.call_count as f64;
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1.0) + latency_ms) / n;
        self.success_rate = self.success_count as f64 / self.call_count as f64;
        // 综合评分：成功率 * 速度因子（满分 1.0）
        let speed_factor = 1.0 / (1.0 + self.avg_latency_ms / 1000.0);
        self.score = self.success_rate * speed_factor;
        self.last_evaluated = Some(now_string());
        // 真实调用清零休眠计数
        self.rounds_dormant = 0;
    }

    /// 记录一次自动测试调用
    ///
    /// 自测试只证明能力"能跑通"，不能证明能力"有用"：
    /// - 只给低基础分（0.1 * success_rate），不享受 speed_factor 加成
    /// - 不清零 rounds_dormant（自测试不等于被使用）
    /// - 计入 auto_test_count，用于 real_call_count 计算
    pub fn record_auto_test(&mut self, success: bool, latency_ms: f64) {
        self.call_count += 1;
        self.auto_test_count += 1;
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        let n = self.call_count as f64;
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1.0) + latency_ms) / n;
        self.success_rate = self.success_count as f64 / self.call_count as f64;
        // 自测试只给低基础分，真实业务调用才会通过 record_real_call 提升分数
        self.score = self.success_rate * 0.1;
        self.last_evaluated = Some(now_string());
        // 注意：不清零 rounds_dormant
    }
}

/// 谱系基因 — 记录进化历史
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageGene {
    /// 创建方式
    #[serde(default)]
    pub origin: Origin,
    /// 父代基因组（变异来源）
    #[serde(default)]
    pub parent: Option<String>,
    /// 变异代数
    #[serde(default)]
    pub generation: u32,
    /// 变异历史
    #[serde(default)]
    pub mutations: Vec<MutationRecord>,
}

/// 能力来源
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum Origin {
    #[default]
    Native,
    /// AI 生成
    Generated,
    /// 变异产生
    Mutated,
    /// 交叉产生
    Crossbred,
    /// 其他来源（LLM 返回的未知值）
    #[serde(other)]
    Other,
}

/// 变异记录
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MutationRecord {
    /// 变异类型
    pub mutation_type: String,
    /// 变异描述
    pub description: String,
    /// 变异时间
    pub timestamp: String,
}

impl CapabilityGenome {
    /// 创建新的基因组
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "0.1.0".into(),
            description: description.into(),
            actions: Vec::new(),
            fitness: FitnessGene::default(),
            lineage: LineageGene::default(),
        }
    }

    /// 添加动作基因
    pub fn with_action(mut self, action: ActionGene) -> Self {
        self.actions.push(action);
        self
    }

    /// 获取动作名称列表
    pub fn action_names(&self) -> Vec<String> {
        self.actions.iter().map(|a| a.name.clone()).collect()
    }

    /// 获取动作描述（给 AI 看）
    pub fn describe(&self) -> String {
        let mut desc = format!("  - {} (v{}): {}\n", self.name, self.version, self.description);
        for action in &self.actions {
            desc.push_str(&format!("    · {}: {}\n", action.name, action.description));
        }
        desc
    }

    /// 记录变异
    pub fn record_mutation(&mut self, mutation_type: impl Into<String>, description: impl Into<String>) {
        self.lineage.mutations.push(MutationRecord {
            mutation_type: mutation_type.into(),
            description: description.into(),
            timestamp: now_string(),
        });
        self.lineage.generation += 1;
    }

    /// 从 JSON 创建
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// 序列化为 JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// 脚本化能力 — 由基因组驱动的能力实现
///
/// 这是革命性的：能力不再需要编译，而是由基因组数据驱动。
/// AI 可以在运行时创建新的基因组，立即获得新能力。
pub struct ScriptedCapability {
    genome: CapabilityGenome,
    /// LLM 客户端（用于 Llm 类型实现）
    llm_client: Option<Arc<LlmExecutor>>,
    /// 消息总线引用（用于 Composite 和 Native 类型实现）
    bus: Option<Arc<MessageBus>>,
    /// 执行器注册表（用于 Custom 类型实现 — 元进化产物）
    executor_registry: Option<Arc<crate::meta_evolve::ExecutorRegistry>>,
    /// 运行时适应度（与 genome.fitness 同步，支持 &self 更新）
    runtime_fitness: Arc<tokio::sync::RwLock<FitnessGene>>,
}

impl ScriptedCapability {
    /// 从基因组创建
    pub fn from_genome(genome: CapabilityGenome) -> Self {
        let fitness = genome.fitness.clone();
        Self {
            genome,
            llm_client: None,
            bus: None,
            executor_registry: None,
            runtime_fitness: Arc::new(tokio::sync::RwLock::new(fitness)),
        }
    }

    /// 从基因组创建，带 LLM 客户端
    pub fn with_llm(mut self, client: Arc<LlmExecutor>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// 绑定消息总线（使 Composite 和 Native 实现可执行）
    pub fn with_bus(mut self, bus: Arc<MessageBus>) -> Self {
        self.bus = Some(bus);
        self
    }

    /// 绑定执行器注册表（使 Custom 实现可执行 — 元进化产物）
    pub fn with_executor_registry(mut self, registry: Arc<crate::meta_evolve::ExecutorRegistry>) -> Self {
        self.executor_registry = Some(registry);
        self
    }

    /// 获取当前运行时适应度快照
    pub async fn runtime_fitness(&self) -> FitnessGene {
        self.runtime_fitness.read().await.clone()
    }

    /// 获取基因组引用
    pub fn genome(&self) -> &CapabilityGenome {
        &self.genome
    }

    /// 获取基因组可变引用
    pub fn genome_mut(&mut self) -> &mut CapabilityGenome {
        &mut self.genome
    }

    /// 执行动作
    async fn execute_action(&self, action: &str, input: &serde_json::Value) -> Result<serde_json::Value, String> {
        let action_gene = self.genome.actions.iter().find(|a| a.name == action)
            .ok_or_else(|| format!("动作 '{}' 不存在于能力 '{}'", action, self.genome.name))?;

        match &action_gene.implementation {
            ActionImpl::Llm { prompt, model, system } => {
                let client = self.llm_client.as_ref()
                    .ok_or_else(|| "LLM 客户端未配置".to_string())?;
                
                let rendered = render_template(prompt, input);
                let system_prompt = system.as_ref().map(|s| render_template(s, input));
                
                let result = client.execute(&rendered, model, system_prompt.as_deref()).await;
                result.map(|text| serde_json::json!({"result": text}))
            }
            ActionImpl::Rule { template } => {
                Ok(render_template_value(template, input))
            }
            ActionImpl::Composite { steps } => {
                // 组合调用：按步骤编排，每步调用其他能力
                let bus = self.bus.as_ref()
                    .ok_or_else(|| "组合能力需要消息总线绑定".to_string())?;

                let mut step_results = serde_json::Map::new();
                let mut context = input.clone();

                for step in steps {
                    // 渲染步骤输入（支持引用前序步骤的输出）
                    let step_input = render_template_value(&step.input, &context);

                    let msg = Message::builder()
                        .from(&self.genome.name)
                        .to(&step.capability)
                        .action(&step.action)
                        .payload(step_input)
                        .build();

                    let resp = bus.send(msg).await.map_err(|e| {
                        format!("组合步骤 '{}' 调用 {}.{} 失败: {}",
                            step.name, step.capability, step.action, e)
                    })?;

                    // 将步骤结果存入上下文，供后续步骤引用
                    context.as_object_mut().map(|obj| {
                        obj.insert(step.name.clone(), resp.payload.clone());
                    });
                    step_results.insert(step.name.clone(), resp.payload);
                }

                Ok(serde_json::Value::Object(step_results))
            }
            ActionImpl::Native { capability, action } => {
                // 委托给原生能力：通过消息总线转发
                let bus = self.bus.as_ref()
                    .ok_or_else(|| "原生委托需要消息总线绑定".to_string())?;

                let msg = Message::builder()
                    .from(&self.genome.name)
                    .to(capability)
                    .action(action)
                    .payload(input.clone())
                    .build();

                let resp = bus.send(msg).await.map_err(|e| {
                    format!("原生委托 {}.{} 失败: {}", capability, action, e)
                })?;

                Ok(resp.payload)
            }
            ActionImpl::Script { language, code, timeout_secs } => {
                // 脚本能力：AI 编写的代码持久化在基因组中
                // 模板渲染后写入临时文件执行
                let rendered_code = render_template(code, input);

                let ext = match language.as_str() {
                    "python" | "py" => "py",
                    "node" | "js" | "javascript" => "js",
                    _ => return Err(format!("不支持的脚本语言: {}", language)),
                };

                let runner = match language.as_str() {
                    "python" | "py" => "python3",
                    "node" | "js" | "javascript" => "node",
                    _ => return Err(format!("不支持的脚本语言: {}", language)),
                };

                let tmp = std::env::temp_dir().join(format!(
                    "script_{}_{}.{}",
                    self.genome.name,
                    uuid::Uuid::new_v4(),
                    ext
                ));

                tokio::fs::write(&tmp, &rendered_code)
                    .await
                    .map_err(|e| format!("写入脚本文件失败: {}", e))?;

                let mut cmd = tokio::process::Command::new(runner);
                cmd.arg(&tmp);
                cmd.stdout(std::process::Stdio::piped());
                cmd.stderr(std::process::Stdio::piped());

                let child = cmd.spawn()
                    .map_err(|e| format!("启动 {} 失败: {}", runner, e))?;

                let output = tokio::time::timeout(
                    std::time::Duration::from_secs(*timeout_secs),
                    child.wait_with_output()
                ).await;

                let _ = tokio::fs::remove_file(&tmp).await;

                match output {
                    Ok(Ok(out)) => {
                        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                        let exit_code = out.status.code();
                        let success = out.status.success();

                        Ok(serde_json::json!({
                            "language": language,
                            "stdout": stdout,
                            "stderr": stderr,
                            "exit_code": exit_code,
                            "success": success,
                        }))
                    }
                    Ok(Err(e)) => Err(format!("脚本执行失败: {}", e)),
                    Err(_) => Err(format!("脚本执行超时 ({}s)", timeout_secs)),
                }
            }
            ActionImpl::Custom { executor_type, params } => {
                let registry = self.executor_registry.as_ref()
                    .ok_or_else(|| "自定义执行器注册表未配置".to_string())?;

                let context = crate::meta_evolve::ExecutorContext {
                    capability_name: self.genome.name.clone(),
                    action_name: action.to_string(),
                };

                registry.execute(executor_type, params, input, &context).await
            }
        }
    }
}

#[async_trait::async_trait]
impl Capability for ScriptedCapability {
    fn name(&self) -> &str {
        &self.genome.name
    }

    fn version(&self) -> &str {
        &self.genome.version
    }

    fn actions(&self) -> Vec<&str> {
        self.genome.actions.iter().map(|a| a.name.as_str()).collect()
    }

    fn describe(&self) -> String {
        self.genome.description.clone()
    }

    fn is_native(&self) -> bool {
        false
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        // 特殊动作：返回当前运行时适应度
        if msg.action == "__fitness__" {
            let fitness = self.runtime_fitness.read().await.clone();
            return Ok(Message::builder()
                .from(&self.genome.name)
                .to(msg.from.as_deref().unwrap_or("orchestrator"))
                .action("__fitness__")
                .payload(serde_json::json!({"fitness": fitness}))
                .build());
        }

        let start = std::time::Instant::now();
        
        match self.execute_action(&msg.action, &msg.payload).await {
            Ok(result) => {
                let latency = start.elapsed().as_millis() as f64;
                tracing::info!(
                    "脚本能力 '{}' 执行 '{}' 成功 ({:.1}ms)",
                    self.genome.name, msg.action, latency
                );

                // 更新运行时适应度（真实业务调用）
                {
                    let mut fitness = self.runtime_fitness.write().await;
                    // Script 执行返回 Ok 时，检查 success 字段判断是否真正成功
                    // 注意：没有 success 字段时视为成功（非 Script 实现如 Llm/Rule 总是 Ok=成功）
                    let actual_success = result.get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    fitness.record_real_call(actual_success, latency);
                }

                Ok(Message::builder()
                    .from(&self.genome.name)
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action(&msg.action)
                    .payload(result)
                    .build())
            }
            Err(e) => {
                tracing::warn!("脚本能力 '{}' 执行 '{}' 失败: {}", self.genome.name, msg.action, e);

                // 更新运行时适应度（真实业务调用失败）
                {
                    let mut fitness = self.runtime_fitness.write().await;
                    fitness.record_real_call(false, 0.0);
                }

                Err(MessageError::Internal {
                    capability: self.genome.name.clone(),
                    detail: e,
                })
            }
        }
    }
}

/// LLM 执行器 — 用于脚本化能力的 LLM 调用
/// 当 base_url 为 "devin" 时，通过 `devin -p` 命令行工具调用 LLM
/// 当 base_url 为 "claude" 时，通过 `claude -p` (Claude Code CLI) 调用 LLM
pub struct LlmExecutor {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
    /// 是否使用 CLI 后端（devin 或 claude）
    use_cli: bool,
    /// CLI 命令名（"devin" 或 "claude"）
    cli_cmd: String,
    /// CLI 使用的模型
    cli_model: String,
}

/// LLM API 响应 — Anthropic 格式
#[derive(Deserialize)]
struct LlmResp {
    content: Vec<LlmContent>,
}

#[derive(Deserialize)]
struct LlmContent {
    #[serde(rename = "type")]
    ct: String,
    text: Option<String>,
    /// thinking 类型 block 的内容
    thinking: Option<String>,
}

/// LLM API 响应 — OpenAI 兼容格式
#[derive(Deserialize)]
struct OpenAiResp {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
}

impl LlmExecutor {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let base_url_str = base_url.into();
        let (use_cli, cli_cmd) = match base_url_str.as_str() {
            "devin" => (true, "devin".to_string()),
            "claude" => (true, "claude".to_string()),
            _ => (false, String::new()),
        };
        let cli_model = std::env::var("CLAUDE_MODEL")
            .or_else(|_| std::env::var("DEVIN_MODEL"))
            .unwrap_or_else(|_| "sonnet".to_string());
        Self {
            api_key: api_key.into(),
            base_url: base_url_str,
            http: reqwest::Client::new(),
            use_cli,
            cli_cmd,
            cli_model,
        }
    }

    /// 是否有可用的 LLM 后端（CLI 模式或已配置 api_key）
    pub fn has_llm_backend(&self) -> bool {
        self.use_cli || !self.api_key.is_empty()
    }

    /// 通过 CLI（devin 或 claude）执行 LLM 调用
    async fn execute_cli(&self, prompt: &str, _model: &str, system: Option<&str>) -> Result<String, String> {
        let full_prompt = if let Some(sys) = system {
            format!("{}\n\n{}", sys, prompt)
        } else {
            prompt.to_string()
        };

        // 重试机制：最多 3 次
        let mut last_err = String::new();
        for attempt in 1..=3u32 {
            if attempt > 1 {
                tracing::warn!("{} CLI 调用重试 {}/3...", self.cli_cmd, attempt);
                tokio::time::sleep(tokio::time::Duration::from_secs(2 * attempt as u64)).await;
            }

            let cmd = tokio::process::Command::new(&self.cli_cmd)
                .args(&["-p", &full_prompt, "--model", &self.cli_model])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output();

            let result = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                cmd,
            ).await;

            match result {
                Ok(Ok(output)) => {
                    if output.status.success() {
                        let text = String::from_utf8_lossy(&output.stdout).to_string();
                        if text.trim().is_empty() {
                            last_err = format!("{} CLI 返回空输出", self.cli_cmd);
                            continue;
                        }
                        return Ok(text);
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                        last_err = format!("{} CLI 错误: {}", self.cli_cmd, stderr.trim());
                        if stderr.contains("402") || stderr.contains("401") || stderr.contains("Payment") {
                            return Err(last_err);
                        }
                        continue;
                    }
                }
                Ok(Err(e)) => {
                    last_err = format!("{} CLI 执行失败: {}", self.cli_cmd, e);
                    continue;
                }
                Err(_) => {
                    last_err = format!("{} CLI 调用超时 (60s)", self.cli_cmd);
                    continue;
                }
            }
        }

        Err(format!("{} CLI 调用失败（重试 3 次）: {}", self.cli_cmd, last_err))
    }

    pub async fn execute(&self, prompt: &str, model: &str, system: Option<&str>) -> Result<String, String> {
        // CLI 后端（devin 或 claude）
        if self.use_cli {
            return self.execute_cli(prompt, model, system).await;
        }

        // 判断 API 格式：包含 "anthropic" 用 Anthropic 格式，否则用 OpenAI 兼容格式
        let is_anthropic = self.base_url.contains("anthropic");

        if is_anthropic {
            self.execute_anthropic(prompt, model, system).await
        } else {
            self.execute_openai(prompt, model, system).await
        }
    }

    /// OpenAI 兼容格式调用
    async fn execute_openai(&self, prompt: &str, model: &str, system: Option<&str>) -> Result<String, String> {
        use serde::Serialize;

        #[derive(Serialize)]
        struct OpenAiReq {
            model: String,
            max_tokens: u32,
            messages: Vec<OpenAiMsg>,
        }

        #[derive(Serialize)]
        struct OpenAiMsg {
            role: String,
            content: String,
        }

        let mut messages = vec![];
        if let Some(sys) = system {
            messages.push(OpenAiMsg {
                role: "system".into(),
                content: sys.to_string(),
            });
        }
        messages.push(OpenAiMsg {
            role: "user".into(),
            content: prompt.to_string(),
        });

        let resolved_model = if model == "auto" {
            std::env::var("ORCH_MODEL").unwrap_or_else(|_| "glm-5.1".to_string())
        } else {
            model.to_string()
        };

        let req = OpenAiReq {
            model: resolved_model,
            max_tokens: 8192,
            messages,
        };

        let base = self.base_url.trim_end_matches('/');

        // 允许通过环境变量指定精确 API 路径
        let custom_path = std::env::var("ORCH_API_PATH").ok();
        let urls: Vec<String> = if let Some(path) = custom_path {
            vec![format!("{}{}", base, path)]
        } else {
            vec![format!("{}/v1/chat/completions", base)]
        };

        let mut last_err = String::new();
        for url in &urls {
            for attempt in 1..=2u32 {
                if attempt > 1 {
                    tracing::warn!("OpenAI API 调用重试 {}/2...", attempt);
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }

                let resp = match self.http
                    .post(url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("content-type", "application/json")
                    .json(&req)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = format!("OpenAI API 请求失败: {}", e);
                        continue;
                    }
                };

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    if status.is_server_error() {
                        last_err = format!("OpenAI API 错误 ({}): {}", status, &body[..200.min(body.len())]);
                        continue;
                    }
                    return Err(format!("OpenAI API 错误 ({}): {}", status, body));
                }

                let body_text = resp.text().await.unwrap_or_default();
                let r: OpenAiResp = match serde_json::from_str(&body_text) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("OpenAI API 原始响应: {}", &body_text[..500.min(body_text.len())]);
                        last_err = format!("OpenAI API 响应解析失败: {}", e);
                        continue;
                    }
                };

                if r.choices.is_empty() {
                    return Err("OpenAI API 返回空 choices".to_string());
                }

                let text = r.choices[0].message.content.clone().unwrap_or_default();
                if text.trim().is_empty() {
                    last_err = "OpenAI API 返回空内容".to_string();
                    continue;
                }
                return Ok(text);
            }
        }

        Err(format!("OpenAI API 调用失败: {}", last_err))
    }

    /// Anthropic 格式调用
    async fn execute_anthropic(&self, prompt: &str, model: &str, system: Option<&str>) -> Result<String, String> {
        use serde::Serialize;

        #[derive(Serialize)]
        struct Req {
            model: String,
            max_tokens: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            system: Option<String>,
            messages: Vec<Msg>,
        }

        #[derive(Serialize)]
        struct Msg {
            role: String,
            content: String,
        }

        let req = Req {
            model: model.to_string(),
            max_tokens: 8192,
            system: system.map(|s| s.to_string()),
            messages: vec![Msg {
                role: "user".into(),
                content: prompt.to_string(),
            }],
        };

        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let mut last_err = String::new();
        for attempt in 1..=3u32 {
            if attempt > 1 {
                tracing::warn!("Anthropic API 调用重试 {}/3...", attempt);
                tokio::time::sleep(tokio::time::Duration::from_secs(2 * attempt as u64)).await;
            }

            let resp = match self.http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("Anthropic API 请求失败: {}", e);
                    continue;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                if status.is_server_error() {
                    last_err = format!("Anthropic API 错误 ({}): {}", status, &body[..200.min(body.len())]);
                    continue;
                }
                return Err(format!("Anthropic API 错误 ({}): {}", status, body));
            }

            let r: LlmResp = match resp.json().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("Anthropic API 响应解析失败: {}", e);
                    continue;
                }
            };

            return Self::parse_response(r);
        }

        Err(format!("Anthropic API 调用失败（重试 3 次）: {}", last_err))
    }

    /// 解析 LLM 响应
    fn parse_response(r: LlmResp) -> Result<String, String> {
        if r.content.is_empty() {
            return Err("LLM 返回空 content".to_string());
        }

        // 记录所有 content block 类型（调试）
        let block_types: Vec<&str> = r.content.iter().map(|c| c.ct.as_str()).collect();
        tracing::debug!("LLM content blocks: {:?}", block_types);

        let text = r.content.iter()
            .filter(|c| c.ct == "text")
            .filter_map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            tracing::warn!("LLM 返回空文本 (block types: {:?})", block_types);
            // 尝试取所有 block 的 text 字段，不管 type
            let all_text: String = r.content.iter()
                .filter_map(|c| c.text.clone())
                .collect();
            if !all_text.is_empty() {
                return Ok(all_text);
            }
            // 尝试从 thinking block 提取内容作为 fallback
            let thinking_text: String = r.content.iter()
                .filter(|c| c.ct == "thinking")
                .filter_map(|c| c.thinking.clone())
                .collect::<Vec<_>>()
                .join("\n");
            if !thinking_text.is_empty() {
                tracing::warn!("使用 thinking block 内容作为 fallback ({} 字符)", thinking_text.len());
                return Ok(thinking_text);
            }
            return Err(format!("LLM 返回空内容 (block types: {:?})", block_types));
        }

        Ok(text)
    }
}

/// 模板渲染 — 将 {{var}} 或 {{nested.path}} 替换为输入中的值
fn render_template(template: &str, input: &serde_json::Value) -> String {
    let mut result = template.to_string();

    // 支持 {{a.b.c}} 形式的嵌套路径引用
    // 用正则找到所有 {{...}} 占位符
    let re = regex::Regex::new(r"\{\{([\w.]+)\}\}").unwrap();
    for cap in re.captures_iter(&template.to_string()) {
        let path = &cap[1];
        let placeholder = format!("{{{{{}}}}}", path);

        // 按点号分割路径，逐层深入 JSON
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = input;
        let mut found = true;
        for part in &parts {
            current = match current {
                serde_json::Value::Object(map) => {
                    if let Some(v) = map.get(*part) {
                        v
                    } else {
                        found = false;
                        break;
                    }
                }
                _ => {
                    found = false;
                    break;
                }
            };
        }

        if found {
            let replacement = match current {
                serde_json::Value::String(s) => s.clone(),
                _ => current.to_string(),
            };
            result = result.replace(&placeholder, &replacement);
        }
    }

    result
}

/// 模板渲染（JSON Value 版本）
fn render_template_value(template: &serde_json::Value, input: &serde_json::Value) -> serde_json::Value {
    match template {
        serde_json::Value::String(s) => {
            serde_json::Value::String(render_template(s, input))
        }
        serde_json::Value::Object(map) => {
            let mut result = serde_json::Map::new();
            for (k, v) in map {
                result.insert(k.clone(), render_template_value(v, input));
            }
            serde_json::Value::Object(result)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|v| render_template_value(v, input)).collect())
        }
        _ => template.clone(),
    }
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 real_call_count = call_count - auto_test_count
    #[test]
    fn test_real_call_count() {
        let mut f = FitnessGene::default();
        assert_eq!(f.real_call_count(), 0);

        // 自测试 3 次
        for _ in 0..3 {
            f.record_auto_test(true, 50.0);
        }
        assert_eq!(f.call_count, 3);
        assert_eq!(f.auto_test_count, 3);
        assert_eq!(f.real_call_count(), 0, "自测试不应计入真实调用");

        // 真实调用 2 次
        for _ in 0..2 {
            f.record_real_call(true, 100.0);
        }
        assert_eq!(f.call_count, 5);
        assert_eq!(f.auto_test_count, 3);
        assert_eq!(f.real_call_count(), 2);
    }

    /// 验证自测试不清零 rounds_dormant，真实调用清零
    #[test]
    fn test_dormant_reset_only_on_real_call() {
        let mut f = FitnessGene::default();
        f.rounds_dormant = 5;

        // 自测试通过：rounds_dormant 不应清零
        f.record_auto_test(true, 100.0);
        assert_eq!(f.rounds_dormant, 5, "自测试不应清零 rounds_dormant");

        // 真实调用：rounds_dormant 应清零
        f.record_real_call(true, 100.0);
        assert_eq!(f.rounds_dormant, 0, "真实调用应清零 rounds_dormant");
    }

    /// 验证自测试分数低于真实调用分数（0.1 vs 1.0 系数）
    #[test]
    fn test_auto_test_score_lower_than_real_call() {
        let mut f1 = FitnessGene::default();
        f1.record_auto_test(true, 10.0);  // 快速通过

        let mut f2 = FitnessGene::default();
        f2.record_real_call(true, 10.0);  // 同样快速通过

        assert!(f1.score < f2.score, "自测试分数应低于真实调用分数");
        assert!(f1.score <= 0.1, "自测试分数不应超过 0.1");
        assert!(f2.score > 0.1, "真实调用分数应高于 0.1");
    }

    /// 验证成功率计算正确
    #[test]
    fn test_success_rate() {
        let mut f = FitnessGene::default();
        f.record_real_call(true, 100.0);
        f.record_real_call(true, 100.0);
        f.record_real_call(false, 100.0);
        assert_eq!(f.call_count, 3);
        assert_eq!(f.success_count, 2);
        assert_eq!(f.failure_count, 1);
        assert!((f.success_rate - 2.0 / 3.0).abs() < 0.001);
    }
}
