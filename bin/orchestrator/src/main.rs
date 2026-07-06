use clap::{Parser, Subcommand};
use capabilities::{
    CodeCapability, ComputeCapability, FsCapability, GreetCapability,
    HttpCapability, ShellCapability, StoreCapability, WebCapability,
};
use runtime::{Agent, MessageBus, OrchestratorBuilder, RegistryBuilder, Workflow};
use std::path::PathBuf;
use std::collections::HashMap;

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
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("CLAUDE_API_KEY"))
                .map_err(|_| anyhow::anyhow!(
                    "请设置 ANTHROPIC_API_KEY 环境变量\n"
                ))?;

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
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("CLAUDE_API_KEY"))
                .map_err(|_| anyhow::anyhow!(
                    "请设置 ANTHROPIC_API_KEY 环境变量\n"
                ))?;

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
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("CLAUDE_API_KEY"))
                .map_err(|_| anyhow::anyhow!(
                    "请设置 ANTHROPIC_API_KEY 环境变量\n"
                ))?;

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
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("CLAUDE_API_KEY"))
                .map_err(|_| anyhow::anyhow!(
                    "请设置 ANTHROPIC_API_KEY 环境变量\n"
                ))?;

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
    }

    Ok(())
}
