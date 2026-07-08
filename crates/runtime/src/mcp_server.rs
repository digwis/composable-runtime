//! MCP Server — 通过 Model Context Protocol 暴露进化引擎的原子操作
//!
//! 架构：
//! - stdio JSON-RPC 2.0 通信
//! - 18 个原子 tool，覆盖进化引擎的所有操作
//! - 与 CLI 共享同一份 genomes.json 存储
//! - server 内部用 LlmExecutor 调 LLM，行为与 CLI 完全一致

use crate::auto_evolve::{
    AutoEvolver, IntrospectionReport, MutationPlan, WeakCapability,
};
use crate::evolution::EvolutionEngine;
use crate::genome::{CapabilityGenome, LlmExecutor, ScriptedCapability};
use crate::message_bus::MessageBus;
use crate::meta_evolve::ExecutorRegistry;
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

/// MCP Server — 持有进化引擎状态，处理 JSON-RPC 请求
pub struct McpServer {
    state: Arc<McpEvolutionState>,
}

/// 进化引擎的共享状态
///
/// Mutex 保护并发访问（单用户场景下竞争极低）。
/// CLI 和 MCP 通过共享 genomes.json 交替使用。
struct McpEvolutionState {
    auto_evolver: Mutex<AutoEvolver>,
    evolution: Mutex<EvolutionEngine>,
    bus: Arc<MessageBus>,
    llm: Arc<LlmExecutor>,
    executor_registry: Arc<ExecutorRegistry>,
    /// 异步任务注册表（evolve_continuous 后台运行）
    tasks: Mutex<HashMap<String, TaskStatus>>,
}

/// 异步任务状态
#[derive(Debug, Clone, Serialize)]
struct TaskStatus {
    status: String,       // "running" | "completed" | "failed"
    round: u32,
    last_actions: Vec<String>,
    error: Option<String>,
}

impl McpServer {
    /// 创建 MCP server
    ///
    /// 与 CLI 共享同一份 genomes.json（通过 storage_dir 指定）。
    /// 调用方需要先注册原生能力到 bus，再调用此函数。
    pub fn new(
        llm: Arc<LlmExecutor>,
        bus: Arc<MessageBus>,
        platform: Platform,
        storage_dir: std::path::PathBuf,
    ) -> Self {
        let registry = Arc::new(ExecutorRegistry::new(&storage_dir));
        let evolution = EvolutionEngine::new(storage_dir)
            .with_llm_executor(llm.clone())
            .with_executor_registry(registry.clone());
        let auto_evolver = AutoEvolver::new(llm.clone(), bus.clone(), platform)
            .with_executor_registry(registry.clone());
        Self {
            state: Arc::new(McpEvolutionState {
                auto_evolver: Mutex::new(auto_evolver),
                evolution: Mutex::new(evolution),
                bus,
                llm,
                executor_registry: registry,
                tasks: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// 启动 MCP server（stdio JSON-RPC）
    pub async fn run(&self) -> Result<(), String> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        tracing::info!("MCP server 启动，等待 JSON-RPC 请求...");

        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            let response = self.handle_request(&line).await;
            if let Some(resp) = response {
                let serialized = serde_json::to_string(&resp)
                    .map_err(|e| format!("序列化响应失败: {}", e))?;
                stdout.write_all(serialized.as_bytes()).await
                    .map_err(|e| format!("写入 stdout 失败: {}", e))?;
                stdout.write_all(b"\n").await
                    .map_err(|e| format!("写入 stdout 失败: {}", e))?;
                stdout.flush().await
                    .map_err(|e| format!("flush stdout 失败: {}", e))?;
            }
        }

        tracing::info!("MCP server stdin 关闭，退出");
        Ok(())
    }

    /// 处理单个 JSON-RPC 请求
    async fn handle_request(&self, line: &str) -> Option<Value> {
        let req: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                return Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": format!("解析错误: {}", e) }
                }));
            }
        };

        let id = req.id.clone();
        // tools/call 的结果需要包装成 MCP 协议要求的 { content: [{ type: "text", text }] } 格式，
        // 其他方法（initialize、tools/list）直接返回原始 result。
        let is_tool_call = req.method == "tools/call";
        let result = match req.method.as_str() {
            "initialize" => Ok(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "evolution-mcp", "version": "0.1.0" }
            })),
            "tools/list" => Ok(self.list_tools()),
            "tools/call" => {
                match serde_json::from_value::<ToolsCallParams>(req.params.unwrap_or_default()) {
                    Ok(params) => self.call_tool(&params).await,
                    Err(e) => Err(format!("无效的 params: {}", e)),
                }
            }
            _ => Err(format!("未知方法: {}", req.method)),
        };

        match result {
            Ok(value) => {
                // tools/call 的结果包装成 MCP content 格式
                let result_value = if is_tool_call {
                    let text = serde_json::to_string_pretty(&value)
                        .unwrap_or_else(|_| value.to_string());
                    serde_json::json!({
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    })
                } else {
                    value
                };
                Some(serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": result_value
                }))
            }
            Err(e) => {
                // tool 调用错误：返回 { content, isError: true } 作为 result
                // 其他方法错误：返回 JSON-RPC error 字段
                if is_tool_call {
                    Some(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [{ "type": "text", "text": e }],
                            "isError": true
                        }
                    }))
                } else {
                    Some(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32603, "message": e }
                    }))
                }
            }
        }
    }

    /// 返回 tool 清单
    fn list_tools(&self) -> Value {
        serde_json::json!({
            "tools": [
                // 只读 (5)
                { "name": "get_introspection_report", "description": "自省：分析能力图谱，返回弱能力、休眠能力、图谱密度", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "get_capability_genome", "description": "获取指定能力的基因组", "inputSchema": { "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] } },
                { "name": "get_evolution_stats", "description": "获取进化统计", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "list_capabilities", "description": "列出所有已注册能力", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "get_genome_lineage", "description": "获取能力的谱系（父代、变异历史、代数）", "inputSchema": { "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] } },

                // 单步进化 (9)
                { "name": "sync_fitness", "description": "同步运行时适应度到进化引擎", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "attribute_failure", "description": "用 LLM 分析弱能力为什么失败，生成变异方案", "inputSchema": { "type": "object", "properties": { "capability": { "type": "string" }, "action": { "type": "string" } }, "required": ["capability", "action"] } },
                { "name": "test_capability", "description": "自测试一个能力", "inputSchema": { "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] } },
                { "name": "eliminate_dormant", "description": "淘汰长期无真实业务调用的能力", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "detect_capability_gaps", "description": "检测环境中的能力缺口", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "fill_capability_gap", "description": "用 LLM 创造新能力填补缺口", "inputSchema": { "type": "object", "properties": { "gap_description": { "type": "string" } }, "required": ["gap_description"] } },
                { "name": "explore_new_capability", "description": "好奇心驱动的探索，创造新能力", "inputSchema": { "type": "object", "properties": { "paradigm_shift": { "type": "boolean", "default": false } } } },
                { "name": "crossover_capabilities", "description": "交叉重组：组合现有能力产生新能力", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "apply_mutation", "description": "应用变异方案（MutationPlan JSON）", "inputSchema": { "type": "object", "properties": { "plan": { "type": "object" } }, "required": ["plan"] } },

                // 组合 (3)
                { "name": "evolve_one_round", "description": "跑一轮完整进化（自省→归因→变异→测试→淘汰）", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "evolve_continuous", "description": "持续进化（异步，返回 task_id）", "inputSchema": { "type": "object", "properties": { "max_rounds": { "type": "integer", "default": 100 }, "idle_threshold": { "type": "integer", "default": 3 }, "interval_secs": { "type": "integer", "default": 5 } } } },
                { "name": "get_task_status", "description": "查询异步任务状态", "inputSchema": { "type": "object", "properties": { "task_id": { "type": "string" } }, "required": ["task_id"] } },

                // 管理 (1)
                { "name": "register_existing_genomes", "description": "把 genomes.json 中的基因组注册到总线", "inputSchema": { "type": "object", "properties": {} } },

                // MCP 原生路径（无 LLM，客户端推理）(5)
                { "name": "get_failure_context", "description": "获取失败归因上下文（弱能力基因组+表现数据+MutationPlan schema），客户端推理后调 apply_mutation", "inputSchema": { "type": "object", "properties": { "capability": { "type": "string" }, "action": { "type": "string" } }, "required": ["capability", "action"] } },
                { "name": "get_gap_context", "description": "获取缺口填补上下文（已有能力+平台信息+基因组模板），客户端设计基因组后调 register_genome", "inputSchema": { "type": "object", "properties": { "gap_description": { "type": "string" } }, "required": ["gap_description"] } },
                { "name": "get_exploration_context", "description": "获取探索上下文（能力摘要+可用工具+基因组模板），客户端设计新能力后调 register_genome", "inputSchema": { "type": "object", "properties": { "paradigm_shift": { "type": "boolean", "default": false } } } },
                { "name": "get_crossover_candidates", "description": "获取交叉重组候选（top2 适应度能力），客户端组合后调 register_genome", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "register_genome", "description": "直接注册一个 CapabilityGenome JSON 到进化引擎和总线（不经过 LLM）", "inputSchema": { "type": "object", "properties": { "genome": { "type": "object" } }, "required": ["genome"] } }
            ]
        })
    }

    /// 分发 tool 调用
    async fn call_tool(&self, params: &ToolsCallParams) -> Result<Value, String> {
        match params.name.as_str() {
            // 只读
            "get_introspection_report" => self.tool_get_introspection_report().await,
            "get_capability_genome" => self.tool_get_capability_genome(params).await,
            "get_evolution_stats" => self.tool_get_evolution_stats().await,
            "list_capabilities" => self.tool_list_capabilities().await,
            "get_genome_lineage" => self.tool_get_genome_lineage(params).await,

            // 单步进化
            "sync_fitness" => self.tool_sync_fitness().await,
            "attribute_failure" => self.tool_attribute_failure(params).await,
            "test_capability" => self.tool_test_capability(params).await,
            "eliminate_dormant" => self.tool_eliminate_dormant().await,
            "detect_capability_gaps" => self.tool_detect_capability_gaps().await,
            "fill_capability_gap" => self.tool_fill_capability_gap(params).await,
            "explore_new_capability" => self.tool_explore_new_capability(params).await,
            "crossover_capabilities" => self.tool_crossover_capabilities().await,
            "apply_mutation" => self.tool_apply_mutation(params).await,

            // 组合
            "evolve_one_round" => self.tool_evolve_one_round().await,
            "evolve_continuous" => self.tool_evolve_continuous(params).await,
            "get_task_status" => self.tool_get_task_status(params).await,

            // 管理
            "register_existing_genomes" => self.tool_register_existing_genomes().await,

            // MCP 原生路径（无 LLM）
            "get_failure_context" => self.tool_get_failure_context(params).await,
            "get_gap_context" => self.tool_get_gap_context(params).await,
            "get_exploration_context" => self.tool_get_exploration_context(params).await,
            "get_crossover_candidates" => self.tool_get_crossover_candidates().await,
            "register_genome" => self.tool_register_genome(params).await,

            _ => Err(format!("未知 tool: {}", params.name)),
        }
    }

    // ===== 只读工具 =====

    /// 检查是否有可用的 LLM 后端，没有则返回错误提示
    fn require_llm(&self) -> Result<(), String> {
        if self.state.llm.has_llm_backend() {
            Ok(())
        } else {
            Err("此工具需要 LLM 后端，但未配置 api_key 且未使用 devin 模式。\n\
                 请使用 client-side 路径替代：\n  \
                 - get_failure_context + apply_mutation\n  \
                 - get_gap_context + register_genome\n  \
                 - get_exploration_context + register_genome\n  \
                 - get_crossover_candidates + register_genome".into())
        }
    }

    async fn tool_get_introspection_report(&self) -> Result<Value, String> {
        let evolver = self.state.auto_evolver.lock().await;
        let evolution = self.state.evolution.lock().await;
        let report: IntrospectionReport = evolver.introspect(&evolution);
        Ok(serde_json::to_value(&report).map_err(|e| e.to_string())?)
    }

    async fn tool_get_capability_genome(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let name = params.get_string("name")?;
        let evolution = self.state.evolution.lock().await;
        let genome = evolution.genomes().get(&name)
            .ok_or_else(|| format!("能力 '{}' 不存在", name))?;
        Ok(serde_json::to_value(genome).map_err(|e| e.to_string())?)
    }

    async fn tool_get_evolution_stats(&self) -> Result<Value, String> {
        let evolver = self.state.auto_evolver.lock().await;
        Ok(serde_json::to_value(evolver.stats()).map_err(|e| e.to_string())?)
    }

    async fn tool_list_capabilities(&self) -> Result<Value, String> {
        let names = self.state.bus.list_capabilities().await;
        let mut caps = Vec::new();
        for name in &names {
            if let Some(cap) = self.state.bus.get_capability(name).await {
                caps.push(serde_json::json!({
                    "name": name,
                    "version": cap.version(),
                    "actions": cap.actions(),
                    "is_native": cap.is_native(),
                }));
            }
        }
        Ok(serde_json::json!({ "capabilities": caps }))
    }

    async fn tool_get_genome_lineage(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let name = params.get_string("name")?;
        let evolution = self.state.evolution.lock().await;
        let genome = evolution.genomes().get(&name)
            .ok_or_else(|| format!("能力 '{}' 不存在", name))?;
        Ok(serde_json::to_value(&genome.lineage).map_err(|e| e.to_string())?)
    }

    // ===== 单步进化工具 =====

    async fn tool_sync_fitness(&self) -> Result<Value, String> {
        let evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;
        evolver.sync_fitness(&mut evolution).await?;
        Ok(serde_json::json!({ "status": "ok" }))
    }

    async fn tool_attribute_failure(&self, params: &ToolsCallParams) -> Result<Value, String> {
        self.require_llm()?;
        let capability = params.get_string("capability")?;
        let _action = params.get_string("action")?;

        // 构造 WeakCapability（从 genome 中查找）
        let evolution = self.state.evolution.lock().await;
        let genome = evolution.genomes().get(&capability)
            .ok_or_else(|| format!("能力 '{}' 不存在", capability))?;
        let weak = WeakCapability {
            name: capability.clone(),
            success_rate: genome.fitness.success_rate,
            call_count: genome.fitness.call_count,
            failure_count: genome.fitness.failure_count,
            avg_latency_ms: genome.fitness.avg_latency_ms,
            actions: genome.action_names(),
        };

        // attribute_failure 需要只读访问 evolution，锁可以继续持有
        let evolver = self.state.auto_evolver.lock().await;
        let result = evolver.attribute_failure(&evolution, &weak).await;
        drop(evolver);
        drop(evolution);

        match result {
            Some(attr) => Ok(serde_json::to_value(&attr).map_err(|e| e.to_string())?),
            None => Ok(serde_json::json!({ "result": null, "message": "归因未产生结果" })),
        }
    }

    /// 注意：attribute_failure 内部使用 `?` on Result，但签名返回 Option
    /// 这是 auto_evolve.rs 的原始设计，MCP 层不改动它，直接传播 Option

    async fn tool_test_capability(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let name = params.get_string("name")?;
        let evolver = self.state.auto_evolver.lock().await;
        let evolution = self.state.evolution.lock().await;
        let (pass, test_input) = evolver.test_capability(&evolution, &name).await;
        Ok(serde_json::json!({ "pass": pass, "test_input": test_input }))
    }

    async fn tool_eliminate_dormant(&self) -> Result<Value, String> {
        // 调用 evolve_once 的淘汰部分（简化版：直接跑一轮但只关心淘汰）
        let evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;

        // 同步适应度，更新 dormant 计数
        evolver.sync_fitness(&mut evolution).await?;

        // 淘汰逻辑（与 auto_evolve.rs 中的逻辑一致）
        const NEW_CAP_THRESHOLD: u32 = 20;
        const FAILED_CAP_THRESHOLD: u32 = 5;
        let mut eliminated = Vec::new();
        let to_eliminate: Vec<String> = evolution.genomes().iter()
            .filter(|(_, g)| {
                let real_calls = g.fitness.real_call_count();
                let threshold = if real_calls == 0 {
                    NEW_CAP_THRESHOLD
                } else if g.fitness.score < 0.01 {
                    FAILED_CAP_THRESHOLD
                } else {
                    return false;
                };
                g.fitness.rounds_dormant >= threshold
            })
            .map(|(name, _)| name.clone())
            .collect();

        for name in &to_eliminate {
            eliminated.push(name.clone());
            evolution.genomes_mut().remove(name);
        }
        evolution.save();

        Ok(serde_json::json!({ "eliminated": eliminated }))
    }

    async fn tool_detect_capability_gaps(&self) -> Result<Value, String> {
        self.require_llm()?;
        let evolver = self.state.auto_evolver.lock().await;
        let evolution = self.state.evolution.lock().await;
        let gaps = evolver.detect_capability_gaps(&evolution).await;
        Ok(serde_json::json!({ "gaps": gaps }))
    }

    async fn tool_fill_capability_gap(&self, params: &ToolsCallParams) -> Result<Value, String> {
        self.require_llm()?;
        let gap = params.get_string("gap_description")?;
        let evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;
        let created = evolver.fill_gap(&mut evolution, &gap).await;
        Ok(serde_json::json!({ "created": created }))
    }

    async fn tool_explore_new_capability(&self, params: &ToolsCallParams) -> Result<Value, String> {
        self.require_llm()?;
        let paradigm_shift = params.get_bool("paradigm_shift").unwrap_or(false);
        let evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;
        let created = evolver.explore_new_capability(&mut evolution, paradigm_shift).await;
        Ok(serde_json::json!({ "created": created }))
    }

    async fn tool_crossover_capabilities(&self) -> Result<Value, String> {
        self.require_llm()?;
        let evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;
        let created = evolver.crossover_capabilities(&mut evolution).await;
        Ok(serde_json::json!({ "created": created }))
    }

    async fn tool_apply_mutation(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let plan_json = params.arguments.get("plan")
            .ok_or("缺少 plan 参数")?;
        let plan: MutationPlan = serde_json::from_value(plan_json.clone())
            .map_err(|e| format!("MutationPlan 反序列化失败: {}", e))?;
        let evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;
        let new_name = evolver.apply_mutation(&mut evolution, &plan).await?;
        Ok(serde_json::json!({ "new_name": new_name }))
    }

    // ===== 组合工具 =====

    async fn tool_evolve_one_round(&self) -> Result<Value, String> {
        self.require_llm()?;
        let mut evolver = self.state.auto_evolver.lock().await;
        let mut evolution = self.state.evolution.lock().await;
        let actions = evolver.evolve_once(&mut evolution).await?;
        let stats = evolver.stats().clone();
        Ok(serde_json::json!({ "actions": actions, "stats": stats }))
    }

    async fn tool_evolve_continuous(&self, params: &ToolsCallParams) -> Result<Value, String> {
        self.require_llm()?;
        let max_rounds = params.get_u32("max_rounds").unwrap_or(100);
        let idle_threshold = params.get_u32("idle_threshold").unwrap_or(3);
        let interval_secs = params.get_u64("interval_secs").unwrap_or(5);

        let task_id = uuid::Uuid::new_v4().to_string();

        // 初始化任务状态
        {
            let mut tasks = self.state.tasks.lock().await;
            tasks.insert(task_id.clone(), TaskStatus {
                status: "running".into(),
                round: 0,
                last_actions: vec![],
                error: None,
            });
        }

        // 启动后台任务
        let state = self.state.clone();
        let task_id_clone = task_id.clone();
        let state_for_error = state.clone();
        tokio::spawn(async move {
            let result = run_continuous_task(
                state, task_id_clone.clone(),
                max_rounds, idle_threshold, interval_secs,
            ).await;
            if let Err(e) = result {
                let mut tasks = state_for_error.tasks.lock().await;
                if let Some(status) = tasks.get_mut(&task_id_clone) {
                    status.status = "failed".into();
                    status.error = Some(e);
                }
            }
        });

        Ok(serde_json::json!({ "task_id": task_id }))
    }

    async fn tool_get_task_status(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let task_id = params.get_string("task_id")?;
        let tasks = self.state.tasks.lock().await;
        let status = tasks.get(&task_id)
            .ok_or_else(|| format!("任务 '{}' 不存在", task_id))?;
        Ok(serde_json::to_value(status).map_err(|e| e.to_string())?)
    }

    // ===== 管理工具 =====

    async fn tool_register_existing_genomes(&self) -> Result<Value, String> {
        let evolution = self.state.evolution.lock().await;
        let llm = self.state.llm.clone();
        let bus = self.state.bus.clone();
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        let mut count = 0;
        for genome in &genomes {
            if genome.actions.is_empty() {
                continue;
            }
            let cap = ScriptedCapability::from_genome(genome.clone())
                .with_llm(llm.clone())
                .with_bus(bus.clone())
                .with_executor_registry(self.state.executor_registry.clone());
            bus.register(Arc::new(cap)).await;
            count += 1;
        }
        Ok(serde_json::json!({ "registered": count }))
    }

    // ===== MCP 原生路径工具（无 LLM，由客户端推理） =====
    //
    // 设计哲学：MCP server 只提供数据读写工具，LLM 推理由客户端（AI agent）完成。
    // 这样 server 不需要 API key，避免 LLM 被双重调用。
    //
    // 工作流：
    //   get_*_context → 客户端推理构造方案 → apply_mutation / register_genome

    /// 获取失败归因上下文：返回弱能力的完整基因组、表现数据、MutationPlan schema 说明
    ///
    /// 客户端（AI agent）拿到上下文后自行分析失败原因，构造 MutationPlan JSON，
    /// 然后调用 apply_mutation 应用变异。
    async fn tool_get_failure_context(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let capability = params.get_string("capability")?;
        let action = params.get_string("action")?;

        let evolution = self.state.evolution.lock().await;
        let genome = evolution.genomes().get(&capability)
            .ok_or_else(|| format!("能力 '{}' 不存在", capability))?;

        // 构造表现数据（与 auto_evolve.rs::attribute_failure 一致）
        let performance = serde_json::json!({
            "success_rate": genome.fitness.success_rate,
            "call_count": genome.fitness.call_count,
            "failure_count": genome.fitness.failure_count,
            "avg_latency_ms": genome.fitness.avg_latency_ms,
            "real_call_count": genome.fitness.real_call_count(),
            "auto_test_count": genome.fitness.auto_test_count,
        });

        // MutationPlan schema 说明（让客户端知道如何构造合法的 plan）
        let mutation_plan_schema = serde_json::json!({
            "description": "变异方案，mutation_type 决定携带的字段，不要携带无关字段",
            "variants": {
                "fix_script": {
                    "mutation_type": "fix_script",
                    "capability": "能力名",
                    "action": "动作名",
                    "new_code": "改进后的完整代码",
                    "expected_improvement": "预期改进效果"
                },
                "fix_composite": {
                    "mutation_type": "fix_composite",
                    "capability": "能力名",
                    "action": "动作名",
                    "new_steps": "CompositeStep 数组（JSON）",
                    "expected_improvement": "预期改进效果"
                },
                "fix_prompt": {
                    "mutation_type": "fix_prompt",
                    "capability": "能力名",
                    "action": "动作名",
                    "new_prompt": "新的提示模板",
                    "expected_improvement": "预期改进效果"
                }
            },
            "constraint": "mutation_type 必须与目标动作的 implementation 类型匹配：Script→fix_script, Composite→fix_composite, Llm→fix_prompt"
        });

        Ok(serde_json::json!({
            "capability": capability,
            "action": action,
            "genome": genome,
            "performance": performance,
            "mutation_plan_schema": mutation_plan_schema,
            "next_step": "分析失败原因，构造 MutationPlan JSON，调用 apply_mutation"
        }))
    }

    /// 获取缺口填补上下文：返回已有能力列表、平台信息、基因组模板
    ///
    /// 客户端设计新能力基因组后调用 register_genome 注册。
    async fn tool_get_gap_context(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let gap = params.get_string("gap_description")?;

        // 锁顺序统一：auto_evolver → evolution（与 tool_test_capability 一致，避免死锁）
        let (os, arch, tools) = {
            let evolver = self.state.auto_evolver.lock().await;
            let platform = evolver.platform();
            let tools: Vec<String> = platform.env.iter()
                .filter(|(k, v)| k.starts_with("has_") && v.as_str() == "true")
                .map(|(k, _)| k.strip_prefix("has_").unwrap_or(k).to_string())
                .collect();
            (platform.os.clone(), platform.arch.clone(), tools)
        };
        let existing: Vec<String> = {
            let evolution = self.state.evolution.lock().await;
            evolution.genomes().keys().cloned().collect()
        };

        // 基因组模板（与 auto_evolve.rs::fill_gap 一致）
        let genome_template = serde_json::json!({
            "name": "能力名",
            "version": "0.1.0",
            "description": "一句话描述",
            "actions": [{
                "name": "动作名",
                "description": "描述",
                "input_schema": { "properties": {} },
                "implementation": {
                    "type": "Script",
                    "language": "python",
                    "code": "简短Python代码",
                    "timeout_secs": 30
                }
            }],
            "fitness": {},
            "lineage": {}
        });

        Ok(serde_json::json!({
            "gap": gap,
            "existing_capabilities": existing,
            "platform": {
                "os": os,
                "arch": arch,
                "available_tools": tools,
            },
            "genome_template": genome_template,
            "next_step": "设计一个新能力基因组填补缺口，调用 register_genome"
        }))
    }

    /// 获取探索上下文：返回能力摘要、可用工具、基因组模板
    ///
    /// 客户端分析当前能力库的认知边界后，提出新能力方向并构造基因组，
    /// 然后调用 register_genome 注册。
    async fn tool_get_exploration_context(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let paradigm_shift = params.get_bool("paradigm_shift").unwrap_or(false);

        // 锁顺序统一：auto_evolver → evolution
        let (all_tools, cap_summary) = {
            let evolver = self.state.auto_evolver.lock().await;
            let platform = evolver.platform();
            let all_tools: Vec<String> = platform.env.iter()
                .filter(|(k, v)| (k.starts_with("has_") || k.starts_with("has_py_")) && v.as_str() == "true")
                .map(|(k, _)| k.strip_prefix("has_").or_else(|| k.strip_prefix("has_py_")).unwrap_or(k).to_string())
                .collect();
            let evolution = self.state.evolution.lock().await;
            let cap_summary: Vec<String> = evolution.genomes().values()
                .map(|g| format!("{}: {} [{}]", g.name, g.description, g.action_names().join(",")))
                .collect();
            (all_tools, cap_summary)
        };

        let genome_template = serde_json::json!({
            "name": "能力名",
            "version": "0.1.0",
            "description": "一句话描述",
            "actions": [{
                "name": "动作名",
                "description": "描述",
                "input_schema": { "properties": {} },
                "implementation": {
                    "type": "Script",
                    "language": "python",
                    "code": "简短Python代码",
                    "timeout_secs": 30
                }
            }],
            "fitness": {},
            "lineage": {}
        });

        Ok(serde_json::json!({
            "capabilities": cap_summary,
            "available_tools": all_tools,
            "paradigm_shift": paradigm_shift,
            "paradigm_shift_note": if paradigm_shift {
                "系统已在当前领域停留太久，请提出一个与现有能力完全不同的方向"
            } else {
                ""
            },
            "genome_template": genome_template,
            "next_step": "分析当前能力库的认知边界，提出新能力方向，构造基因组后调用 register_genome"
        }))
    }

    /// 获取交叉重组候选：返回适应度最高的两个能力
    ///
    /// 客户端组合两个父代能力产生新能力基因组，然后调用 register_genome 注册。
    async fn tool_get_crossover_candidates(&self) -> Result<Value, String> {
        let evolution = self.state.evolution.lock().await;
        let mut sorted: Vec<_> = evolution.genomes().values().cloned().collect();
        if sorted.len() < 2 {
            return Err("能力库不足 2 个，无法交叉重组".into());
        }
        sorted.sort_by(|a, b| b.fitness.score.partial_cmp(&a.fitness.score).unwrap_or(std::cmp::Ordering::Equal));
        let parent1 = &sorted[0];
        let parent2 = &sorted[1];

        let genome_template = serde_json::json!({
            "name": "新能力名",
            "version": "0.1.0",
            "description": "一句话描述",
            "actions": [{
                "name": "动作名",
                "description": "描述",
                "input_schema": { "properties": {} },
                "implementation": {
                    "type": "Script",
                    "language": "python",
                    "code": "简短Python代码",
                    "timeout_secs": 30
                }
            }],
            "fitness": {},
            "lineage": {}
        });

        Ok(serde_json::json!({
            "parent1": {
                "name": parent1.name,
                "description": parent1.description,
                "actions": parent1.action_names(),
                "fitness_score": parent1.fitness.score,
            },
            "parent2": {
                "name": parent2.name,
                "description": parent2.description,
                "actions": parent2.action_names(),
                "fitness_score": parent2.fitness.score,
            },
            "genome_template": genome_template,
            "next_step": "组合两个父代能力产生新能力基因组，调用 register_genome"
        }))
    }

    /// 直接注册一个 CapabilityGenome JSON 到进化引擎和总线（不经过 LLM）
    ///
    /// 用于 MCP 原生路径：客户端设计好基因组后直接注册。
    /// 与 fill_gap/explore_new_capability/crossover_capabilities 的注册逻辑一致。
    async fn tool_register_genome(&self, params: &ToolsCallParams) -> Result<Value, String> {
        let genome_json = params.arguments.get("genome")
            .ok_or("缺少 genome 参数")?;
        let genome: CapabilityGenome = serde_json::from_value(genome_json.clone())
            .map_err(|e| format!("CapabilityGenome 反序列化失败: {}", e))?;

        let name = genome.name.clone();
        let mut evolution = self.state.evolution.lock().await;
        evolution.register_genome(genome);
        drop(evolution);

        // 注册到总线（与 fill_gap 等一致）
        let evolution = self.state.evolution.lock().await;
        if let Some(genome) = evolution.genomes().get(&name) {
            let cap = ScriptedCapability::from_genome(genome.clone())
                .with_llm(self.state.llm.clone())
                .with_bus(self.state.bus.clone())
                .with_executor_registry(self.state.executor_registry.clone());
            self.state.bus.register(Arc::new(cap)).await;
        }

        Ok(serde_json::json!({ "registered": name }))
    }
}

/// 后台运行持续进化任务
async fn run_continuous_task(
    state: Arc<McpEvolutionState>,
    task_id: String,
    max_rounds: u32,
    idle_threshold: u32,
    interval_secs: u64,
) -> Result<(), String> {
    let mut idle_count = 0u32;
    let mut round = 0u32;

    while round < max_rounds {
        round += 1;

        // 执行一轮
        let actions = {
            let mut evolver = state.auto_evolver.lock().await;
            let mut evolution = state.evolution.lock().await;
            evolver.evolve_once(&mut evolution).await?
        };

        // 更新任务状态
        {
            let mut tasks = state.tasks.lock().await;
            if let Some(status) = tasks.get_mut(&task_id) {
                status.round = round;
                status.last_actions = actions.clone();
            }
        }

        // 判断是否有进化动作（与 CLI 逻辑一致）
        let has_evolution_action = actions.iter().any(|a| {
            !a.starts_with("自测试")
                && !a.starts_with("无")
                && !a.starts_with("发现缺口")
                && !a.contains("(测试失败)")
                && !a.contains("(未能自动填补)")
        });

        if actions.is_empty() || !has_evolution_action {
            idle_count += 1;
        } else {
            idle_count = 0;
        }

        if idle_count >= idle_threshold {
            break;
        }

        if round < max_rounds && idle_count < idle_threshold {
            tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)).await;
        }
    }

    // 标记完成
    let mut tasks = state.tasks.lock().await;
    if let Some(status) = tasks.get_mut(&task_id) {
        status.status = "completed".into();
    }

    Ok(())
}

// ===== JSON-RPC 类型 =====

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Value,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ToolsCallParams {
    name: String,
    #[serde(default)]
    arguments: Value,
}

impl ToolsCallParams {
    fn get_string(&self, key: &str) -> Result<String, String> {
        self.arguments.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("缺少参数: {}", key))
    }

    fn get_bool(&self, key: &str) -> Option<bool> {
        self.arguments.get(key)
            .and_then(|v| v.as_bool())
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        self.arguments.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    }

    fn get_u64(&self, key: &str) -> Option<u64> {
        self.arguments.get(key)
            .and_then(|v| v.as_u64())
    }
}
