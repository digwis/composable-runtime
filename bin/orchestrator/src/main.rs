use clap::{Parser, Subcommand};
use capabilities::{
    CodeCapability, ComputeCapability, FsCapability, GreetCapability,
    HttpCapability, ShellCapability, StoreCapability, WebCapability,
};
use runtime::{
    Agent, LlmExecutor, MessageBus, McpServer, OrchestratorBuilder, Platform, RegistryBuilder, Workflow,
    Daemon, DaemonConfig, discover_llm_backends,
};
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "orch")]
#[command(about = "可组合能力编排引擎 — 统一运行时原型")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 运行工作流
    Run {
        /// 工作流 YAML 文件路径
        #[arg(short, long)]
        workflow: PathBuf,

        /// 详细输出
        #[arg(short, long)]
        verbose: bool,
    },
    /// 列出已注册能力
    List,
    /// 能力自省 — 显示所有能力的详细信息和动作
    Introspect,
    /// 交互模式 — 直接发送消息
    Send {
        /// 目标能力
        #[arg(short, long)]
        to: String,
        /// 动作
        #[arg(short, long)]
        action: String,
        /// JSON 负载
        #[arg(short, long, default_value = "{}")]
        payload: String,
    },
    /// 显示消息流转历史
    History {
        /// 工作流文件（可选，用于执行后查看历史）
        #[arg(short, long)]
        workflow: Option<PathBuf>,
    },
    /// 动态执行 — 从 JSON 指令直接执行单步（模拟 AI Agent 调用）
    Exec {
        /// JSON 指令（包含 name, capability, action, input）
        #[arg(short, long)]
        json: String,
    },
    /// AI Agent — 接入 LLM 自动编排能力完成任务
    Agent {
        /// 自然语言任务描述
        #[arg(short, long)]
        task: String,

        /// 最大迭代次数
        #[arg(short, long, default_value = "10")]
        max_iterations: usize,

        /// LLM 模型
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// API Base URL（支持代理）
        #[arg(long, default_value = "https://api.anthropic.com")]
        base_url: String,

        /// 启用进化引擎（AI 可创造/变异能力）
        #[arg(long, default_value_t = true)]
        evolve: bool,
    },
    /// 自主进化 — 不需要用户任务，系统自省并改进已有能力
    AutoEvolve {
        /// LLM 模型
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// API Base URL（支持代理）
        #[arg(long, default_value = "https://api.anthropic.com")]
        base_url: String,

        /// 运行轮数
        #[arg(short, long, default_value = "1")]
        rounds: u32,
    },
    /// 持续进化 — 无目标自创生模式，持续运行直到收敛或终止
    EvolveContinuous {
        /// LLM 模型
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// API Base URL（支持代理）
        #[arg(long, default_value = "https://api.anthropic.com")]
        base_url: String,

        /// 最大轮数
        #[arg(short, long, default_value = "100")]
        max_rounds: u32,

        /// 空闲阈值（连续 N 轮无动作则停止）
        #[arg(short, long, default_value = "3")]
        idle_threshold: u32,

        /// 轮间隔（秒）
        #[arg(long, default_value = "5")]
        interval: u64,
    },
    /// 定向进化 — 朝目标方向持续进化
    EvolveGoal {
        /// 进化目标
        #[arg(short, long)]
        goal: String,

        /// LLM 模型
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// API Base URL（支持代理）
        #[arg(long, default_value = "https://api.anthropic.com")]
        base_url: String,

        /// 最大轮数
        #[arg(short, long, default_value = "20")]
        max_rounds: u32,

        /// 轮间隔（秒）
        #[arg(long, default_value = "5")]
        interval: u64,
    },
    /// MCP Server — 通过 Model Context Protocol 暴露进化引擎给 Agent
    ///
    /// 启动 stdio JSON-RPC server，Agent（如 Claude Desktop）可调用 18 个原子 tool
    /// 进行自省、归因、变异、测试等进化操作。与 CLI 共享同一份 genomes.json。
    Mcp {
        /// LLM 模型
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// API Base URL（支持代理）
        #[arg(long, default_value = "https://api.anthropic.com")]
        base_url: String,

        /// 存储目录（默认使用平台标准目录，与 CLI 共享 genomes.json）
        #[arg(long)]
        storage: Option<PathBuf>,
    },
    /// Daemon — 系统级常驻进化运行时
    ///
    /// 启动 Unix socket server + PATH 注入 + 后台进化循环。
    /// 任何 AI 工具（Claude/Cursor/Devin）可通过 ~/.orch/bin/ 下的可执行文件
    /// 或 Unix socket 自动使用进化能力。
    Daemon {
        /// LLM 模型
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// API Base URL（支持代理）
        #[arg(long, default_value = "https://api.anthropic.com")]
        base_url: String,

        /// 存储目录（默认 ~/.orch/）
        #[arg(long)]
        storage: Option<PathBuf>,

        /// 进化循环间隔（秒）
        #[arg(long, default_value_t = 300)]
        interval: u64,

        /// 最大进化轮次
        #[arg(long, default_value_t = 100)]
        max_rounds: u32,
    },
    /// Autonomous — 自主模式（单次感知→目标→执行）
    ///
    /// 主动感知环境状态，自主生成目标，调用能力执行。
    /// 不启动 daemon，执行一轮后退出。
    Autonomous {
        /// API Base URL（支持代理，OpenAI 兼容接口如 https://api.iamhc.cn）
        #[arg(long, default_value = "claude")]
        base_url: String,

        /// LLM 模型名
        #[arg(long, default_value = "glm-5.1")]
        model: String,

        /// 存储目录（默认 ~/.orch/）
        #[arg(long)]
        storage: Option<PathBuf>,
    },
    /// 查看已发现的 LLM 后端
    Discover,
}

/// 获取 API key — devin 模式或未设置时返回空字符串
///
/// - base_url == "devin": 通过 `devin -p` CLI 调用 LLM，不需要 api_key
/// - MCP server: client-side 路径（get_*_context + register_genome）不需要 api_key
/// - 其他情况: 需要 ANTHROPIC_API_KEY 或 CLAUDE_API_KEY 环境变量
fn get_api_key(base_url: &str) -> String {
    if base_url == "devin" || base_url == "claude" {
        return String::new();
    }
    // 优先使用通用 ORCH_API_KEY
    if let Ok(key) = std::env::var("ORCH_API_KEY") {
        return key;
    }
    match std::env::var("ANTHROPIC_API_KEY").or_else(|_| std::env::var("CLAUDE_API_KEY")) {
        Ok(key) => key,
        Err(_) => {
            tracing::warn!(
                "未设置 ORCH_API_KEY/ANTHROPIC_API_KEY/CLAUDE_API_KEY — server-side LLM 工具将不可用，\
                 请使用 client-side 路径或设置 --base-url devin"
            );
            String::new()
        }
    }
}

fn build_registry() -> MessageBus {
    RegistryBuilder::new()
        .with(GreetCapability)
        .with(ComputeCapability)
        .with(StoreCapability::new())
        .with(FsCapability)
        .with(ShellCapability)
        .with(HttpCapability::new())
        .with(CodeCapability)
        .with(WebCapability::new())
        .build()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "orch=info,runtime=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { workflow, verbose } => {
            let wf = Workflow::from_file(workflow.to_str().unwrap())?;
            println!("\n📋 工作流: {}", wf.name);
            if !wf.description.is_empty() {
                println!("   描述: {}\n", wf.description);
            }

            let bus = build_registry();
            let orchestrator = OrchestratorBuilder::new()
                .with_bus(bus)
                .build();

            let result = orchestrator.run(&wf).await?;

            let status_icon = if result.success { "✅" } else { "⚠️" };
            println!("{} 执行完成: {} 步执行, {} 步跳过, {} 步失败, {} 步重试\n",
                status_icon, result.steps_executed, result.steps_skipped,
                result.steps_failed, result.steps_retried);

            if verbose {
                println!("── 步骤输出 ──");
                for output in &result.outputs {
                    let retry_info = if output.retries > 0 {
                        format!(" (重试 {} 次)", output.retries)
                    } else {
                        String::new()
                    };
                    match &output.result {
                        Ok(payload) => {
                            println!("  [{}] {}.{}{} → {}",
                                output.step, output.capability, output.action,
                                retry_info,
                                serde_json::to_string_pretty(payload)?);
                        }
                        Err(e) => {
                            println!("  [{}] {}.{}{} → ❌ {}",
                                output.step, output.capability, output.action, retry_info, e);
                        }
                    }
                }

                println!("\n── 最终上下文 ──");
                for (k, v) in &result.context {
                    println!("  {} = {}", k, serde_json::to_string_pretty(v)?);
                }
            }

            // 打印消息历史
            let history = orchestrator.bus().history().await;
            if !history.is_empty() {
                println!("\n── 消息流转 ──");
                for log in &history {
                    println!("  {} → {} ({}) [{}]",
                        log.message.from.as_deref().unwrap_or("?"),
                        log.message.to,
                        log.message.action,
                        log.result);
                }
            }
        }

        Commands::List => {
            let bus = build_registry();
            let caps = bus.list_capabilities().await;
            println!("\n📦 已注册能力:");
            for cap in caps {
                println!("  • {}", cap);
            }
            println!();
        }

        Commands::Introspect => {
            let bus = build_registry();
            let caps = bus.introspect().await;
            println!("\n🔍 能力自省:\n");
            for cap in &caps {
                println!("  ┌─ {} v{}", cap.name, cap.version);
                println!("  │  动作: {}", cap.actions.join(", "));
                println!("  │  描述: {}", cap.description);
                println!("  └─\n");
            }
        }

        Commands::Send { to, action, payload } => {
            let bus = build_registry();
            let payload: serde_json::Value = serde_json::from_str(&payload)?;
            let msg = runtime::Message::builder()
                .from("cli")
                .to(&to)
                .action(&action)
                .payload(payload)
                .build();

            let response = bus.send(msg).await?;
            println!("\n📨 响应:");
            println!("  from: {}", response.from.as_deref().unwrap_or("?"));
            println!("  action: {}", response.action);
            println!("  payload: {}\n", serde_json::to_string_pretty(&response.payload)?);
        }

        Commands::History { workflow: _ } => {
            println!("\n📜 消息历史需要先执行工作流。请使用 `orch run --workflow <file>` 然后查看输出。\n");
        }

        Commands::Exec { json } => {
            let bus = build_registry();
            let orchestrator = OrchestratorBuilder::new()
                .with_bus(bus)
                .build();

            let json_value: serde_json::Value = serde_json::from_str(&json)?;
            let context: HashMap<String, serde_json::Value> = HashMap::new();

            let (output, retries, failed) = orchestrator.execute_json(json_value, &context).await?;

            if failed > 0 {
                println!("\n❌ 动态执行失败 (重试 {} 次)", retries);
                if let Err(e) = &output.result {
                    println!("  错误: {}\n", e);
                }
            } else {
                println!("\n✅ 动态执行成功{}", if retries > 0 { format!(" (重试 {} 次)", retries) } else { String::new() });
                if let Ok(payload) = &output.result {
                    println!("  [{}] {}.{} → {}\n",
                        output.step, output.capability, output.action,
                        serde_json::to_string_pretty(payload)?);
                }
            }
        }

        Commands::Agent { task, max_iterations, model, base_url, evolve } => {
            let api_key = get_api_key(&base_url);

            let bus = build_registry();
            let orchestrator = OrchestratorBuilder::new()
                .with_bus(bus)
                .build();

            let mut agent = Agent::new(orchestrator, api_key)
                .with_max_iterations(max_iterations)
                .with_model(model)
                .with_base_url(base_url);

            if evolve {
                agent = agent.with_evolution();
            }

            let result = agent.run(&task).await?;

            println!("════════════════════════════════");
            println!("🤖 Agent 结果:");
            println!("   任务: {}", result.task);
            println!("   成功: {}", if result.success { "✅ 是" } else { "❌ 否" });
            println!("   迭代: {}", result.iterations);
            println!("   步骤: {}", result.outputs.len());
            println!("   学习: {}", if result.learned { "🧠 已保存工作流模板" } else { "无" });
            if !result.capabilities_created.is_empty() {
                println!("   🧬 创造/变异能力: {}", result.capabilities_created.join(", "));
            }
            if !result.summary.is_empty() {
                println!("   总结: {}", result.summary);
            }

            // 显示记忆
            let memory = agent.memory();
            if !memory.workflow_templates.is_empty() {
                println!("\n🧠 记忆中的工作流模板:");
                for w in &memory.workflow_templates {
                    println!("   • '{}' (成功 {} 次)", w.task, w.success_count);
                }
            }
            if !memory.failed_attempts.is_empty() {
                println!("\n⚠️  失败记录:");
                for f in &memory.failed_attempts {
                    println!("   • '{}': {}", f.step, f.error);
                }
            }

            // 显示进化报告
            if let Some(report) = agent.evolution_report() {
                println!("\n{}", report);
            }
            println!();
        }

        Commands::AutoEvolve { model, base_url, rounds } => {
            let api_key = get_api_key(&base_url);

            let bus = build_registry();
            let orchestrator = OrchestratorBuilder::new()
                .with_bus(bus)
                .build();

            let mut agent = Agent::new(orchestrator, api_key)
                .with_model(model)
                .with_base_url(base_url)
                .with_evolution();

            // 注册已有基因组到总线
            if let Some(evo) = agent.evolution() {
                if let Some(llm) = agent.llm_executor() {
                    let bus = agent.orchestrator().bus().clone();
                    let genomes: Vec<_> = evo.genomes().values().cloned().collect();
                    for genome in &genomes {
                        if genome.actions.is_empty() { continue; }
                        let cap = runtime::ScriptedCapability::from_genome(genome.clone())
                            .with_llm(llm.clone())
                            .with_bus(bus.clone());
                        agent.orchestrator().bus().register(std::sync::Arc::new(cap)).await;
                    }
                }
            }

            for round in 1..=rounds {
                println!("\n🧬 ═══ 自主进化 第 {} 轮 ═══", round);

                // 先 clone 需要的值，避免借用冲突
                let llm = agent.llm_executor().cloned();
                let bus = agent.orchestrator().bus().clone();
                let platform = agent.platform().clone();

                if let Some(evo) = agent.evolution_mut() {
                    if let Some(llm) = &llm {
                        let mut auto = runtime::AutoEvolver::new(
                            llm.clone(),
                            bus.clone(),
                            platform.clone(),
                        );
                        match auto.evolve_once(evo).await {
                            Ok(actions) => {
                                if actions.is_empty() {
                                    println!("  无需进化动作");
                                } else {
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
                        println!("\n{}", auto.report());
                    }
                }
            }

            // 显示最终进化报告
            if let Some(report) = agent.evolution_report() {
                println!("\n{}", report);
            }
            println!();
        }

        Commands::EvolveContinuous { model, base_url, max_rounds, idle_threshold, interval } => {
            let api_key = get_api_key(&base_url);

            let bus = build_registry();
            let orchestrator = OrchestratorBuilder::new()
                .with_bus(bus)
                .build();

            let mut agent = Agent::new(orchestrator, api_key)
                .with_model(model)
                .with_base_url(base_url)
                .with_evolution();

            // 注册已有基因组到总线
            if let Some(evo) = agent.evolution() {
                if let Some(llm) = agent.llm_executor() {
                    let bus = agent.orchestrator().bus().clone();
                    let genomes: Vec<_> = evo.genomes().values().cloned().collect();
                    for genome in &genomes {
                        if genome.actions.is_empty() { continue; }
                        let cap = runtime::ScriptedCapability::from_genome(genome.clone())
                            .with_llm(llm.clone())
                            .with_bus(bus.clone());
                        agent.orchestrator().bus().register(std::sync::Arc::new(cap)).await;
                    }
                }
            }

            let llm = agent.llm_executor().cloned();
            let bus = agent.orchestrator().bus().clone();
            let platform = agent.platform().clone();

            if let Some(evo) = agent.evolution_mut() {
                if let Some(llm) = &llm {
                    let mut auto = runtime::AutoEvolver::new(
                        llm.clone(),
                        bus,
                        platform,
                    );
                    auto.evolve_continuous(evo, max_rounds, idle_threshold, interval).await
                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                }
            }

            if let Some(report) = agent.evolution_report() {
                println!("\n{}", report);
            }
        }

        Commands::EvolveGoal { goal, model, base_url, max_rounds, interval } => {
            let api_key = get_api_key(&base_url);

            let bus = build_registry();
            let orchestrator = OrchestratorBuilder::new()
                .with_bus(bus)
                .build();

            let mut agent = Agent::new(orchestrator, api_key)
                .with_model(model)
                .with_base_url(base_url)
                .with_evolution();

            // 注册已有基因组到总线
            if let Some(evo) = agent.evolution() {
                if let Some(llm) = agent.llm_executor() {
                    let bus = agent.orchestrator().bus().clone();
                    let genomes: Vec<_> = evo.genomes().values().cloned().collect();
                    for genome in &genomes {
                        if genome.actions.is_empty() { continue; }
                        let cap = runtime::ScriptedCapability::from_genome(genome.clone())
                            .with_llm(llm.clone())
                            .with_bus(bus.clone());
                        agent.orchestrator().bus().register(std::sync::Arc::new(cap)).await;
                    }
                }
            }

            let llm = agent.llm_executor().cloned();
            let bus = agent.orchestrator().bus().clone();
            let platform = agent.platform().clone();

            if let Some(evo) = agent.evolution_mut() {
                if let Some(llm) = &llm {
                    let mut auto = runtime::AutoEvolver::new(
                        llm.clone(),
                        bus,
                        platform,
                    );
                    auto.evolve_towards(evo, &goal, max_rounds, interval).await
                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                }
            }

            if let Some(report) = agent.evolution_report() {
                println!("\n{}", report);
            }
        }

        Commands::Mcp { model, base_url, storage } => {
            let api_key = get_api_key(&base_url);

            // 平台探测 + 存储目录（与 CLI 共享 genomes.json）
            let platform = Platform::detect();
            let storage_dir = storage.unwrap_or_else(|| PathBuf::from(platform.storage_dir()));
            std::fs::create_dir_all(&storage_dir)
                .map_err(|e| anyhow::anyhow!("创建存储目录失败: {}: {}", storage_dir.display(), e))?;

            // 构建 LLM 执行器（行为与 CLI 完全一致）
            let llm = Arc::new(LlmExecutor::new(api_key, base_url));

            // 注册原生能力到总线（让 MCP 也能调用 greet/compute/fs/shell 等能力）
            let bus = Arc::new(build_registry());
            let native_count = bus.list_capabilities().await.len();

            tracing::info!(
                "MCP server 启动: storage={}, model={}, native capabilities={}",
                storage_dir.display(), model, native_count
            );

            let server = McpServer::new(
                llm,
                bus,
                platform,
                storage_dir,
            );

            server.run().await.map_err(|e| anyhow::anyhow!("MCP server 错误: {}", e))?;
        }

        Commands::Daemon { model, base_url, storage, interval, max_rounds } => {
            let api_key = get_api_key(&base_url);

            // 设置默认模型供运行时内部使用
            std::env::set_var("ORCH_MODEL", &model);

            // 多模型路由配置（可通过环境变量覆盖）
            let fast_model = std::env::var("ORCH_MODEL_FAST").unwrap_or_else(|_| model.clone());
            let smart_model = std::env::var("ORCH_MODEL_SMART").unwrap_or_else(|_| "MiniMax-M3".to_string());
            let coder_model = std::env::var("ORCH_MODEL_CODER").unwrap_or_else(|_| "MiniMax-M3".to_string());
            println!("🧬 多模型路由配置:");
            println!("  fast (测试输入/简单任务): {}", fast_model);
            println!("  smart (归因分析/目标生成): {}", smart_model);
            println!("  coder (代码生成/变异):    {}", coder_model);

            let platform = Platform::detect();
            let storage_dir = storage.unwrap_or_else(|| PathBuf::from(format!("{}/.orch", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))));
            std::fs::create_dir_all(&storage_dir)
                .map_err(|e| anyhow::anyhow!("创建存储目录失败: {}: {}", storage_dir.display(), e))?;

            // 构建 LLM 执行器
            let llm = Arc::new(LlmExecutor::new(api_key, base_url));

            // 构建进化引擎（加载已有 genomes.json）
            let evo_storage = storage_dir.join(".evolution");
            std::fs::create_dir_all(&evo_storage).ok();

            // 如果 ~/.orch/.evolution/genomes.json 不存在或为空，尝试从项目级复制
            let target_genomes = evo_storage.join("genomes.json");
            let need_copy = !target_genomes.exists()
                || std::fs::read_to_string(&target_genomes)
                    .map(|c| c.trim() == "[]" || c.trim().is_empty())
                    .unwrap_or(true);
            if need_copy {
                let local_genomes = PathBuf::from(".evolution/genomes.json");
                if local_genomes.exists() {
                    if let Ok(content) = std::fs::read_to_string(&local_genomes) {
                        if content.trim() != "[]" && !content.trim().is_empty() {
                            std::fs::write(&target_genomes, &content).ok();
                            println!("📦 从项目级复制 genomes.json 到 {}", target_genomes.display());
                        }
                    }
                }
            }

            let evolution = runtime::EvolutionEngine::new(evo_storage);

            // 构建 daemon 配置
            let config = DaemonConfig {
                socket_path: storage_dir.join("socket"),
                bin_dir: storage_dir.join("bin"),
                storage_dir: storage_dir.clone(),
                evolution_interval_secs: interval,
                max_rounds,
            };

            // 自动发现 LLM 后端
            let backends = discover_llm_backends();
            println!("🔍 已发现 LLM 后端:");
            for b in &backends {
                println!("   • {} ({:?}) {}", b.name, b.backend_type,
                    if !b.command.is_empty() { format!("→ {}", b.command) } else { String::new() });
            }

            // 构建消息总线 + 注册原生能力
            let bus = Arc::new(build_registry());

            println!("\n🧬 进化运行时 Daemon 启动");
            println!("   socket: {}", config.socket_path.display());
            println!("   bin:    {}", config.bin_dir.display());
            println!("   能力:   {} 个已注册", evolution.genomes().len());
            println!("   进化间隔: {}s, 最大轮次: {}", interval, max_rounds);
            println!("\n   按 Ctrl+C 停止\n");

            let mut daemon = Daemon::new(config, bus, evolution, Some(llm), platform);
            daemon.run().await.map_err(|e| anyhow::anyhow!("Daemon 错误: {}", e))?;
        }

        Commands::Autonomous { base_url, model, storage } => {
            let api_key = get_api_key(&base_url);
            let platform = Platform::detect();
            let storage_dir = storage.unwrap_or_else(|| PathBuf::from(format!("{}/.orch", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))));

            // 设置默认模型供运行时内部使用（LlmExecutor execute_openai 会读取 ORCH_MODEL）
            std::env::set_var("ORCH_MODEL", &model);
            let evo_storage = storage_dir.join(".evolution");
            std::fs::create_dir_all(&evo_storage).ok();

            // 如果 ~/.orch/.evolution/genomes.json 不存在或为空，尝试从项目级复制
            let target_genomes = evo_storage.join("genomes.json");
            let need_copy = !target_genomes.exists()
                || std::fs::read_to_string(&target_genomes)
                    .map(|c| c.trim() == "[]" || c.trim().is_empty())
                    .unwrap_or(true);
            if need_copy {
                let local_genomes = PathBuf::from(".evolution/genomes.json");
                if local_genomes.exists() {
                    if let Ok(content) = std::fs::read_to_string(&local_genomes) {
                        if content.trim() != "[]" && !content.trim().is_empty() {
                            std::fs::write(&target_genomes, &content).ok();
                            println!("📦 从项目级复制 genomes.json 到 {}", target_genomes.display());
                        }
                    }
                }
            }

            let mut evolution = runtime::EvolutionEngine::new(evo_storage);

            let llm = Arc::new(LlmExecutor::new(api_key, base_url));
            let bus = Arc::new(build_registry());

            // 注册已有能力
            let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
            for genome in &genomes {
                if genome.actions.is_empty() || !platform.is_compatible(genome) {
                    continue;
                }
                let cap = runtime::ScriptedCapability::from_genome(genome.clone())
                    .with_llm(llm.clone())
                    .with_bus(bus.clone());
                bus.register(Arc::new(cap)).await;
            }

            println!("\n🤖 自主模式启动");
            println!("   能力: {} 个已注册", evolution.genomes().len());
            println!("   平台: {} ({})\n", platform.os, platform.arch);

            let mut auto = runtime::AutonomousRuntime::new(llm, bus, platform);

            // 1. 感知环境
            println!("👁️  感知环境...");
            let report = auto.perceive().await;
            println!("   磁盘: {:.0}%", report.disk_usage_pct);
            println!("   监听端口: {} 个", report.network_listening.len());
            println!("   进程: {} 个", report.running_processes.len());
            if let Some(git) = &report.git_status {
                println!("   Git: 有变更\n{}", git);
            }
            if !report.recent_files.is_empty() {
                println!("   最近修改文件: {} 个", report.recent_files.len());
            }

            // 2. 生成目标
            let caps: Vec<String> = evolution.genomes().keys().cloned().collect();
            println!("\n🎯 生成目标 (可用能力 {} 个)...", caps.len());
            let goals = auto.generate_goals(&report, &caps).await;

            if goals.is_empty() {
                println!("   未生成目标，环境正常或无匹配能力。");
            } else {
                for (i, goal) in goals.iter().enumerate() {
                    println!("   {}. [{}] {}", i + 1, goal.priority, goal.description);
                    println!("      原因: {}", goal.reason);
                }

                // 3. 执行目标
                println!("\n⚙️  执行目标...");
                for goal in &goals {
                    println!("\n   ▶ [{}] {}", goal.priority, goal.description);
                    let result = auto.execute_goal(goal, &mut evolution).await;
                    if result.success {
                        println!("   ✅ 成功 ({}ms)", result.elapsed_ms);
                        if let Some(output) = result.output.get("result") {
                            println!("   结果: {}", serde_json::to_string_pretty(output).unwrap_or_default());
                        }
                    } else {
                        println!("   ❌ 失败: {}", result.error.as_deref().unwrap_or("未知"));
                    }
                }
            }

            let (successes, failures) = auto.stats();
            println!("\n📊 自主循环完成: {} 成功, {} 失败", successes, failures);
        }

        Commands::Discover => {
            let backends = discover_llm_backends();
            println!("\n🔍 已发现的 LLM 后端:\n");
            if backends.is_empty() {
                println!("  未发现任何 LLM 后端。");
                println!("  安装 claude CLI: npm install -g @anthropic-ai/claude-code");
                println!("  或设置 ANTHROPIC_API_KEY 环境变量\n");
            } else {
                for b in &backends {
                    let type_str = match b.backend_type {
                        runtime::BackendType::Cli => "CLI",
                        runtime::BackendType::Http => "HTTP API",
                    };
                    let cmd_str = if !b.command.is_empty() {
                        format!(" → {}", b.command)
                    } else {
                        String::new()
                    };
                    println!("  • {:20} [{}]{}", b.name, type_str, cmd_str);
                }
                println!();
            }
        }
    }

    Ok(())
}
