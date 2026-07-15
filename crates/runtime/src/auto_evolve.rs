use crate::driver::EvolutionDriver;
use crate::evolution::EvolutionEngine;
use crate::genome::{ActionGene, ActionImpl, CapabilityGenome, CompositeStep, ScriptedCapability};
use crate::message_bus::MessageBus;
use crate::meta_evolve::ExecutorRegistry;
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 自主进化引擎 — 不依赖用户任务的后台进化循环
///
/// 实现"有认知的自创生"的阶段 3：
/// 1. 自省：分析能力图谱，发现弱能力和缺口
/// 2. 归因：LLM 分析失败原因
/// 3. 变异：自动生成改进方案
/// 4. 测试：自动验证变异效果
/// 5. 选择：保留更优的变异体
///
/// ## 反馈机制设计
///
/// ### 正反馈（自增强）
/// - 创造新能力 → 能力库增长 → 更多交叉重组机会
/// - 高适应度能力被选为父代 → 产生更多变异后代
///
/// ### 负反馈（耗散/纠正）
/// - 淘汰机制：长期无真实业务调用的能力被移除
/// - tried_gaps 滑动窗口：避免反复填补同一缺口
/// - 收敛判断：连续空闲 N 轮则停止
/// - LLM 成本预算：每轮调用上限
///
/// 当负反馈失效时（如自测试过的能力逃出淘汰池），
/// 正反馈失控，能力库无限膨胀 → 系统相变。
pub struct AutoEvolver {
    /// LLM 执行器（用于归因分析和变异方案生成）
    llm: Arc<dyn EvolutionDriver>,
    /// 消息总线（用于测试能力）
    bus: Arc<MessageBus>,
    /// 平台信息
    platform: Platform,
    /// 执行器注册表（用于 Custom 类型能力 — 元进化产物）
    executor_registry: Option<Arc<ExecutorRegistry>>,
    /// 进化统计
    stats: AutoEvolveStats,
    /// 进化轮次计数
    round_count: u32,
    /// 已尝试填补的工具缺口：工具名 -> 最后尝试的轮次
    ///
    /// 滑动窗口机制：GAP_RETRY_ROUNDS 轮后自动过期，
    /// 允许能力被淘汰后重新填补（恢复自我修复能力）。
    tried_gaps: std::collections::HashMap<String, u32>,
    /// 能力连续变异失败次数：能力名 -> 失败次数
    /// 连续失败 3 次后跳过，避免卡环
    mutation_failures: std::collections::HashMap<String, u32>,
    /// 能力最近一次变异失败的轮次：能力名 -> 轮次
    ///
    /// 滑动窗口机制：MUTATION_RETRY_ROUNDS 轮后自动过期，
    /// 允许失败能力在冷却期后重新被选为变异目标（渐进修复）。
    mutation_failures_round: std::collections::HashMap<String, u32>,
    /// 环境验证器注册表 — 把"能力自报成功"升级为"环境证明成功"
    ///
    /// 真实世界压力的注入点：validate_in_real_project 执行能力后，
    /// 用匹配的验证器追加一次环境校验（cargo build / git status / 退出码），
    /// 结果覆盖原 success 判定，并喂给 fitness.real_validation_passes。
    validators: Arc<crate::validator::ValidatorRegistry>,
}

/// 自主进化统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutoEvolveStats {
    /// 自省次数
    pub introspections: u32,
    /// 归因次数
    pub attributions: u32,
    /// 自主变异次数
    pub mutations: u32,
    /// 变异成功次数
    pub mutation_successes: u32,
    /// 自动发现缺口次数
    pub gaps_found: u32,
    /// 缺口填补次数
    pub gaps_filled: u32,
    /// 好奇心探索次数
    pub explorations: u32,
    /// 探索创造能力数
    pub explored_created: u32,
    /// 自动淘汰次数
    pub eliminations: u32,
    /// 交叉重组次数
    pub crossovers: u32,
    /// 自测试次数
    pub auto_tests: u32,
    /// 自测试通过次数
    pub auto_test_passes: u32,
    /// 结晶化尝试次数
    pub crystallizations: u32,
    /// 结晶化成功次数
    pub crystallization_successes: u32,
}

/// 真实项目验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealValidationResult {
    /// 能力名
    pub capability: String,
    /// 执行的 action
    pub action: String,
    /// 是否成功
    pub success: bool,
    /// 耗时（ms）
    pub elapsed_ms: u64,
    /// 输出内容
    pub output: String,
    /// 环境验证证据（验证器追加的真实世界信号摘要，用于归因和审计）
    #[serde(default)]
    pub evidence: String,
    /// 真实信号强度（透传自 RealWorldSignal.strength，供回写 fitness）
    #[serde(default)]
    pub strength: crate::validator::SignalStrength,
}

/// 自省报告 — 系统自我分析的产物
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionReport {
    /// 弱能力（成功率 < 0.8 且调用次数 > 2）
    pub weak_capabilities: Vec<WeakCapability>,
    /// 未使用能力（调用次数为 0 且存在时间较长）
    pub dormant_capabilities: Vec<String>,
    /// 能力总数
    pub total_capabilities: usize,
    /// 平均适应度
    pub avg_fitness: f64,
    /// 能力图谱密度（能力间引用关系数 / 能力数²）
    pub graph_density: f64,
    /// P5: 多样性评分（0.0~1.0，越高越好）
    #[serde(default)]
    pub diversity_score: f64,
    /// P5: 重复能力组（基础名, [版本列表]）
    #[serde(default)]
    pub duplicate_groups: Vec<(String, Vec<String>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakCapability {
    pub name: String,
    pub success_rate: f64,
    pub call_count: u32,
    pub failure_count: u32,
    pub avg_latency_ms: f64,
    pub actions: Vec<String>,
}

/// 归因结果 — LLM 分析失败原因后的产物
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionResult {
    /// 失败原因分析
    pub analysis: String,
    /// 建议的变异方案
    pub mutation_plan: MutationPlan,
}

/// 归因快照 — 弱能力的只读副本 + 相关教训，用于无锁归因
///
/// 归因只需读弱能力的 genome + 相关 lessons。把它们克隆出来做成快照后，
/// 归因可在不持有 shared 锁的情况下并行执行，消除进化循环持锁跨 LLM 调用
/// 导致 socket 命令阻塞的根因。见 `AutoEvolver::snapshot_for_attribution`
/// 和 `AutoEvolver::attribute_failure_snapshot`。
#[derive(Debug, Clone, Default)]
pub struct AttributionSnapshot {
    /// 弱能力的完整 genome 副本（name → 克隆）
    pub genomes: std::collections::HashMap<String, CapabilityGenome>,
    /// 与本次归因能力相关的跨代教训副本
    pub lessons: Vec<crate::evolution::EvolutionLesson>,
}

/// 变异测试目标 — 锁内取出的 genome 快照，用于锁外测试。
///
/// 三阶段重构的核心载体：
/// 1. `prepare_mutations`（锁内）：自省 + 变异应用 + 取快照 → 返回 Vec<MutationTestTarget>
/// 2. `test_and_select_unlocked`（锁外）：用快照版方法测试 + 回归 + AB → 返回 Vec<TestOutcome>
/// 3. `commit_test_results`（锁内）：根据 TestOutcome 写回选择结果
#[derive(Debug, Clone)]
pub struct MutationTestTarget {
    /// 父代名
    pub parent_name: String,
    /// 变异体名
    pub child_name: String,
    /// 父代 genome 快照
    pub parent_genome: CapabilityGenome,
    /// 变异体 genome 快照
    pub child_genome: CapabilityGenome,
}

/// 测试结果 — 锁外测试的产出，用于锁内写回。
#[derive(Debug, Clone)]
pub enum TestOutcome {
    /// 测试 + 回归 + AB 全部通过，应推广变异体
    Promote {
        parent_name: String,
        child_name: String,
        test_input: Option<serde_json::Value>,
    },
    /// 测试失败，应淘汰变异体
    TestFailed {
        parent_name: String,
        child_name: String,
        test_input: Option<serde_json::Value>,
    },
    /// 回归测试失败，应淘汰变异体
    RegressionFailed {
        parent_name: String,
        child_name: String,
    },
    /// AB 对比失败（父代更优），应回滚变异体
    AbRolledBack {
        parent_name: String,
        child_name: String,
    },
}

/// 第一阶段结果 — 锁内执行自省+变异+取快照后的产出
///
/// daemon 三阶段编排：
/// 1. `prepare_phase`（锁内）→ Phase1Result
/// 2. `test_and_select_unlocked`（锁外）→ Vec<TestOutcome>
/// 3. `commit_phase2`（锁内）→ 写回结果 + 后续步骤
pub struct Phase1Result {
    /// 第一阶段产生的 actions（多样性合并、归因等）
    pub actions: Vec<String>,
    /// 变异测试目标快照（空表示无变异或能力库为空）
    pub test_targets: Vec<MutationTestTarget>,
    /// 变异应用失败的列表
    pub mutation_failures: Vec<(String, String)>,
    /// 自省报告（供后续阶段使用）
    pub report: IntrospectionReport,
    /// 是否跳过（能力库为空或 sync_fitness 失败时为 true）
    pub skipped: bool,
}

/// 第二阶段中间结果 — 写回变异测试结果后，收集自测试和真实验证的 genome 快照
pub struct CommitIntermediate {
    /// 变异阶段的 actions
    pub actions: Vec<String>,
    /// 自省报告
    pub report: IntrospectionReport,
    /// 自测试目标快照（genome 副本）
    pub self_test_targets: Vec<(String, CapabilityGenome)>,
    /// 真实验证目标快照（genome 副本）
    pub validation_targets: Vec<(String, CapabilityGenome)>,
    /// 是否跳过后续步骤
    pub skipped: bool,
}

/// 自测试结果（锁外执行产出）
pub struct SelfTestResult {
    pub name: String,
    pub pass: bool,
    pub test_input: Option<serde_json::Value>,
}

/// 真实验证结果（锁外执行产出）
pub struct ValidationResult {
    pub name: String,
    pub success: bool,
    pub evidence: String,
    pub strength: crate::validator::SignalStrength,
    pub elapsed_ms: u64,
}

/// 变异方案 — 编译期保证 mutation_type 与携带字段匹配
///
/// 用 enum 替代原来的 struct + Option<String> + Option<Value> + String mutation_type：
/// - 原设计允许 "fix_script" + new_steps（无效状态）可表示
/// - 新设计让每种变异类型只携带自己需要的字段，无效状态不可表示
/// - 如果 LLM 返回不匹配的组合，反序列化直接失败（fail fast）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mutation_type", rename_all = "snake_case")]
pub enum MutationPlan {
    /// 修改 Script 实现的代码
    FixScript {
        capability: String,
        action: String,
        new_code: String,
        expected_improvement: String,
    },
    /// 补丁式修改 Script 实现 — 只改局部,不重写整个文件
    ///
    /// 适合小修(加 try/except、修单个 bug、加参数校验):LLM 只需给出
    /// "找到这段、换成那段",不用维持整个文件一致性 —— 认知负荷低、出错面小。
    /// 锚点 `find` 必须在现有代码里精确且唯一匹配,否则 fail fast。
    FixScriptPatch {
        capability: String,
        action: String,
        /// 要查找的原始代码片段(必须与现有代码精确匹配,作为锚点)
        find: String,
        /// 替换成的代码
        replace: String,
        expected_improvement: String,
    },
    /// 修改 Composite 实现的步骤
    FixComposite {
        capability: String,
        action: String,
        new_steps: serde_json::Value,
        expected_improvement: String,
    },
    /// 修改 Llm 实现的提示模板
    FixPrompt {
        capability: String,
        action: String,
        new_prompt: String,
        expected_improvement: String,
    },
    /// 修改 Custom 执行器的参数（元进化产物）
    FixCustomParams {
        capability: String,
        action: String,
        new_params: serde_json::Value,
        expected_improvement: String,
    },
}

impl MutationPlan {
    /// 目标能力名
    fn capability(&self) -> &str {
        match self {
            MutationPlan::FixScript { capability, .. } => capability,
            MutationPlan::FixScriptPatch { capability, .. } => capability,
            MutationPlan::FixComposite { capability, .. } => capability,
            MutationPlan::FixPrompt { capability, .. } => capability,
            MutationPlan::FixCustomParams { capability, .. } => capability,
        }
    }

    /// 目标动作名
    fn action(&self) -> &str {
        match self {
            MutationPlan::FixScript { action, .. } => action,
            MutationPlan::FixScriptPatch { action, .. } => action,
            MutationPlan::FixComposite { action, .. } => action,
            MutationPlan::FixPrompt { action, .. } => action,
            MutationPlan::FixCustomParams { action, .. } => action,
        }
    }

    /// 预期改进描述
    fn expected_improvement(&self) -> &str {
        match self {
            MutationPlan::FixScript {
                expected_improvement,
                ..
            } => expected_improvement,
            MutationPlan::FixScriptPatch {
                expected_improvement,
                ..
            } => expected_improvement,
            MutationPlan::FixComposite {
                expected_improvement,
                ..
            } => expected_improvement,
            MutationPlan::FixPrompt {
                expected_improvement,
                ..
            } => expected_improvement,
            MutationPlan::FixCustomParams {
                expected_improvement,
                ..
            } => expected_improvement,
        }
    }

    /// 变异类型字符串（用于记录到谱系）
    fn mutation_type_str(&self) -> &'static str {
        match self {
            MutationPlan::FixScript { .. } => "fix_script",
            MutationPlan::FixScriptPatch { .. } => "fix_script_patch",
            MutationPlan::FixComposite { .. } => "fix_composite",
            MutationPlan::FixPrompt { .. } => "fix_prompt",
            MutationPlan::FixCustomParams { .. } => "fix_custom_params",
        }
    }
}

/// 工具缺口重试间隔：N 轮后允许重新尝试填补被淘汰的缺口
const GAP_RETRY_ROUNDS: u32 = 20;

/// 范式跃迁触发阈值：连续 N 轮没有创造新能力时触发
const PARADIGM_SHIFT_IDLE_ROUNDS: u32 = 5;

/// 好奇心探索触发阈值：连续 N 轮没有创造新能力时主动探索
const EXPLORATION_IDLE_ROUNDS: u32 = 1;

impl AutoEvolver {
    pub fn new(llm: Arc<dyn EvolutionDriver>, bus: Arc<MessageBus>, platform: Platform) -> Self {
        Self {
            llm,
            bus,
            platform,
            executor_registry: None,
            stats: AutoEvolveStats::default(),
            round_count: 0,
            tried_gaps: std::collections::HashMap::new(),
            mutation_failures: std::collections::HashMap::new(),
            mutation_failures_round: std::collections::HashMap::new(),
            validators: crate::validator::default_registry(),
        }
    }

    /// P0-1: Python 语法预检 — 用 ast.parse 快速过滤语法错误的代码
    pub async fn validate_python_syntax(code: &str) -> Result<(), String> {
        let check = tokio::process::Command::new("python3")
            .arg("-c")
            .arg("import ast; ast.parse(open('/dev/stdin').read())")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn();
        let mut child = check.map_err(|e| format!("启动 python3 失败: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(code.as_bytes())
                .await
                .map_err(|e| format!("写入 stdin 失败: {}", e))?;
        }
        let output = child
            .wait_with_output()
            .await
            .map_err(|e| format!("等待 python3 失败: {}", e))?;
        if output.status.success() {
            Ok(())
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            // 不截断到 200 字符 —— Python traceback 的有用部分(行号、错误类型)常在后半段,
            // 截断会让 LLM 归因时看不到真正的语法错误位置。保留完整 stderr(上限 2000 防失控)。
            let limit = err.len().min(2000);
            Err(format!("语法错误: {}", &err[..limit]))
        }
    }

    /// P0-1: 对 genome 中所有 Script action 做语法预检
    pub async fn validate_genome_scripts(genome: &CapabilityGenome) -> Result<(), String> {
        for action in &genome.actions {
            if let ActionImpl::Script { code, language, .. } = &action.implementation {
                if language == "python" {
                    Self::validate_python_syntax(code).await?;
                }
            }
        }
        Ok(())
    }

    /// 设置执行器注册表（元进化产物）
    pub fn with_executor_registry(mut self, registry: Arc<ExecutorRegistry>) -> Self {
        self.executor_registry = Some(registry);
        self
    }

    /// 设置环境验证器注册表（真实世界硬信号注入）
    pub fn with_validators(mut self, registry: Arc<crate::validator::ValidatorRegistry>) -> Self {
        self.validators = registry;
        self
    }

    /// 创建带执行器注册表的 ScriptedCapability
    fn build_capability(&self, genome: crate::genome::CapabilityGenome) -> ScriptedCapability {
        let mut cap = ScriptedCapability::from_genome(genome)
            .with_llm(self.llm.clone())
            .with_bus(self.bus.clone());
        if let Some(registry) = &self.executor_registry {
            cap = cap.with_executor_registry(registry.clone());
        }
        cap
    }

    /// 运行一轮自主进化循环
    ///
    /// 自省 → 归因 → 变异 → 测试 → 选择
    pub async fn evolve_once(
        &mut self,
        evolution: &mut EvolutionEngine,
    ) -> Result<Vec<String>, String> {
        // 普通调用在进入本轮前同步一次；daemon 的预计算路径已在取快照前同步。
        self.sync_fitness(evolution).await?;
        // 兼容旧调用：无预计算归因，走内联归因（持锁）
        self.evolve_once_core(evolution, None).await
    }

    /// 内联归因（持锁）— 无 daemon 编排时用，如 mcp/测试路径
    ///
    /// 错峰启动第 i 个归因延迟 i*90s。返回与 weak_list 一一对应的结果。
    async fn attribute_weak_caps_inline(
        &mut self,
        evolution: &EvolutionEngine,
        weak_list: &[WeakCapability],
    ) -> Vec<Option<AttributionResult>> {
        let mut attr_futures = Vec::new();
        for (i, weak) in weak_list.iter().enumerate() {
            let delay_secs = i as u64 * 90;
            let attr_fut = self.attribute_failure(evolution, weak);
            attr_futures.push(Box::pin(async move {
                if delay_secs > 0 {
                    tracing::info!("归因错峰: {} 延迟 {}s 启动", weak.name, delay_secs);
                    tokio::time::sleep(tokio::time::Duration::from_secs(delay_secs)).await;
                }
                attr_fut.await
            }));
        }
        futures::future::join_all(attr_futures).await
    }

    /// 无锁归因（daemon 编排用）— 用快照在锁外跑，错峰全程不占 shared 锁
    ///
    /// daemon 流程：锁内 introspect + snapshot_for_attribution → 释放锁 →
    /// 调本方法跑归因 → 锁内 evolve_once_with_attribution 写回。
    /// 这消除了"持锁跨 LLM 归因（3 分钟）"导致 socket/HTTP 全 503 的根因。
    pub async fn attribute_weak_caps_snapshot(
        &self,
        snapshot: &AttributionSnapshot,
        weak_list: &[WeakCapability],
    ) -> Vec<Option<AttributionResult>> {
        let mut attr_futures = Vec::new();
        for (i, weak) in weak_list.iter().enumerate() {
            let delay_secs = i as u64 * 90;
            let attr_fut = self.attribute_failure_snapshot(snapshot, weak);
            attr_futures.push(Box::pin(async move {
                if delay_secs > 0 {
                    tracing::info!("归因错峰: {} 延迟 {}s 启动", weak.name, delay_secs);
                    tokio::time::sleep(tokio::time::Duration::from_secs(delay_secs)).await;
                }
                attr_fut.await
            }));
        }
        futures::future::join_all(attr_futures).await
    }

    /// daemon 编排辅助：锁内调用，返回 (weak_list, snapshot) 供锁外归因用
    ///
    /// 弱能力筛选逻辑与 evolve_once 内联版一致（take 3 + 跳过连续失败≥3）。
    pub fn prepare_attribution(
        &self,
        evolution: &EvolutionEngine,
        report: &IntrospectionReport,
    ) -> (Vec<WeakCapability>, AttributionSnapshot) {
        let weak_list: Vec<WeakCapability> = report
            .weak_capabilities
            .iter()
            .take(3)
            .filter(|weak| {
                // P3a-fix: 用滑动窗口冷却替代永久封禁（只读判断，清除延后到 &mut self 上下文）
                if self.is_mutation_in_cooldown(&weak.name) {
                    let fc = *self.mutation_failures.get(&weak.name).unwrap_or(&0);
                    println!("  ⏭️  跳过 {} (连续变异失败 {} 次, 冷却中)", weak.name, fc);
                    false
                } else {
                    if *self.mutation_failures.get(&weak.name).unwrap_or(&0) >= 3 {
                        println!("  ♻️  变异冷却期结束, 重新尝试: {}", weak.name);
                    }
                    true
                }
            })
            .cloned()
            .collect();
        let snapshot = Self::snapshot_for_attribution(evolution, &weak_list);
        (weak_list, snapshot)
    }

    /// 运行一轮进化，可传入无锁预计算的归因结果
    ///
    /// `precomputed_attribution = Some((weak_list, attr_results, snapshot))` 时跳过内联归因
    /// （daemon 在锁外用快照跑完归因后传入），消除"持锁跨 LLM 归因"导致的 socket/HTTP 阻塞。
    /// 调用方必须已经同步 fitness；None 时退化为内联归因，供 `evolve_once` 使用。
    pub async fn evolve_once_with_attribution(
        &mut self,
        evolution: &mut EvolutionEngine,
        precomputed_attribution: Option<(
            Vec<WeakCapability>,
            Vec<Option<AttributionResult>>,
            AttributionSnapshot,
        )>,
    ) -> Result<Vec<String>, String> {
        // 锁外归因可能持续数分钟；写回前刷新期间发生的真实调用，但不重复推进 dormant。
        self.refresh_runtime_fitness(evolution).await?;
        self.evolve_once_core(evolution, precomputed_attribution)
            .await
    }

    /// 三阶段编排 — 第一阶段（锁内）：自省 + 多样性合并 + 归因写回 + 变异应用 + 取快照
    ///
    /// daemon 调用此方法后释放锁，在锁外调用 `test_and_select_unlocked`，
    /// 再持锁调用 `commit_phase2` 写回结果。
    pub async fn prepare_phase(
        &mut self,
        evolution: &mut EvolutionEngine,
        precomputed_attribution: Option<(
            Vec<WeakCapability>,
            Vec<Option<AttributionResult>>,
            AttributionSnapshot,
        )>,
    ) -> Result<Phase1Result, String> {
        self.refresh_runtime_fitness(evolution).await?;

        let mut actions = Vec::new();
        let mut mutation_failures: Vec<(String, String)> = Vec::new();
        self.round_count += 1;

        let report = self.introspect(evolution);
        self.stats.introspections += 1;

        if report.total_capabilities == 0 {
            println!("  🔍 自省: 能力库为空，从零开始...");
            return Ok(Phase1Result {
                actions,
                test_targets: Vec::new(),
                mutation_failures,
                report,
                skipped: true,
            });
        }

        println!(
            "  🔍 自省: {} 个能力, 平均适应度 {:.2}, {} 个弱能力, {} 个休眠能力, 多样性 {:.0}%",
            report.total_capabilities,
            report.avg_fitness,
            report.weak_capabilities.len(),
            report.dormant_capabilities.len(),
            report.diversity_score * 100.0,
        );

        // P5: 多样性合并
        if !report.duplicate_groups.is_empty() {
            println!(
                "  ⚠️  多样性警告: {} 个重复组: {}",
                report.duplicate_groups.len(),
                report
                    .duplicate_groups
                    .iter()
                    .map(|(base, versions)| format!("{}({})", base, versions.len()))
                    .collect::<Vec<_>>()
                    .join(", "),
            );

            for (_base, versions) in &report.duplicate_groups {
                if versions.len() <= 1 {
                    continue;
                }
                let best = versions.iter().max_by(|a, b| {
                    let fa = evolution
                        .genomes()
                        .get(*a)
                        .map(|g| g.fitness.score)
                        .unwrap_or(0.0);
                    let fb = evolution
                        .genomes()
                        .get(*b)
                        .map(|g| g.fitness.score)
                        .unwrap_or(0.0);
                    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(best_name) = best {
                    for ver in versions {
                        if ver == best_name {
                            continue;
                        }
                        let dependents = evolution.find_dependents(ver);
                        if !dependents.is_empty() {
                            continue;
                        }
                        if let Some(g) = evolution.genomes().get(ver) {
                            if g.fitness.score
                                < evolution
                                    .genomes()
                                    .get(best_name)
                                    .map(|bg| bg.fitness.score)
                                    .unwrap_or(0.0)
                            {
                                println!(
                                    "  🔄 多样性合并: 淘汰 {} (适应度 {:.2} < {} 的 {:.2})",
                                    ver,
                                    g.fitness.score,
                                    best_name,
                                    evolution
                                        .genomes()
                                        .get(best_name)
                                        .map(|bg| bg.fitness.score)
                                        .unwrap_or(0.0)
                                );
                                if let Err(e) = evolution.remove_genome(ver) {
                                    tracing::warn!("remove_genome 保存失败: {}", e);
                                }
                                self.bus.unregister(ver).await;
                                self.stats.eliminations += 1;
                                actions.push(format!("多样性合并: 淘汰 {}", ver));
                            }
                        }
                    }
                }
            }
        }

        // 归因 + 变异
        let mut precomputed_attribution = precomputed_attribution;
        let (weak_list, attr_results, attribution_snapshot): (
            Vec<WeakCapability>,
            Vec<Option<AttributionResult>>,
            Option<AttributionSnapshot>,
        ) = if let Some((wl, pre, snapshot)) = precomputed_attribution.take() {
            (wl, pre, Some(snapshot))
        } else {
            let wl: Vec<WeakCapability> = report
                .weak_capabilities
                .iter()
                .take(3)
                .filter(|weak| *self.mutation_failures.get(&weak.name).unwrap_or(&0) < 3)
                .cloned()
                .collect();
            let attrs = self.attribute_weak_caps_inline(evolution, &wl).await;
            (wl, attrs, None)
        };

        // 串行变异
        let mut mutation_results: Vec<(String, Result<String, String>)> = Vec::new();
        for (weak, attr) in weak_list.iter().zip(attr_results.into_iter()) {
            if let Some(attr) = attr {
                if let Some(snapshot) = &attribution_snapshot {
                    if !Self::attribution_snapshot_is_current(evolution, &report, snapshot, weak) {
                        tracing::info!(
                            "归因结果已过期，跳过写回: {}（归因期间能力或证据发生变化）",
                            weak.name
                        );
                        actions.push(format!("归因过期: {} (跳过变异)", weak.name));
                        continue;
                    }
                }
                if attr.mutation_plan.capability() != weak.name {
                    tracing::warn!(
                        "归因目标不一致，跳过: 弱能力={} 方案目标={}",
                        weak.name,
                        attr.mutation_plan.capability()
                    );
                    actions.push(format!("归因目标不一致: {} (跳过变异)", weak.name));
                    continue;
                }
                self.stats.attributions += 1;
                println!("  🧠 归因: {} → {}", weak.name, attr.analysis);

                evolution.record_thought_chain(crate::evolution::ThoughtChain {
                    chain_type: "attribution".to_string(),
                    reasoning: safe_truncate(&attr.analysis, 2000).to_string(),
                    conclusion: format!("变异方案: {:?}", attr.mutation_plan),
                    related_capabilities: vec![weak.name.clone()],
                    related_goal: None,
                    success: true,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                });

                let result = self.apply_mutation(evolution, &attr.mutation_plan).await;
                self.stats.mutations += 1;
                mutation_results.push((weak.name.clone(), result));
            }
        }

        // 取 genome 快照
        let test_targets: Vec<MutationTestTarget> = mutation_results
            .iter()
            .filter_map(|(parent, result)| {
                if let Ok(new_name) = result {
                    let parent_genome = evolution.genomes().get(parent)?.clone();
                    let child_genome = evolution.genomes().get(new_name)?.clone();
                    Some(MutationTestTarget {
                        parent_name: parent.clone(),
                        child_name: new_name.clone(),
                        parent_genome,
                        child_genome,
                    })
                } else {
                    None
                }
            })
            .collect();

        // 收集变异应用失败
        for (parent, result) in &mutation_results {
            if let Err(e) = result {
                mutation_failures.push((parent.clone(), e.clone()));
            }
        }

        Ok(Phase1Result {
            actions,
            test_targets,
            mutation_failures,
            report,
            skipped: false,
        })
    }

    /// 三阶段中间步骤（锁内）：写回变异测试结果 + 收集自测试/真实验证快照
    ///
    /// daemon 调用此方法后释放锁，在锁外执行 `run_self_tests_unlocked` + `run_validations_unlocked`，
    /// 再持锁调用 `commit_final` 完成后续步骤。
    pub async fn commit_test_results(
        &mut self,
        evolution: &mut EvolutionEngine,
        phase1: Phase1Result,
        test_outcomes: Vec<TestOutcome>,
    ) -> Result<CommitIntermediate, String> {
        let report = phase1.report.clone();
        let mut actions = phase1.actions;

        if phase1.skipped {
            return Ok(CommitIntermediate {
                actions,
                report,
                self_test_targets: Vec::new(),
                validation_targets: Vec::new(),
                skipped: true,
            });
        }

        // 写回测试结果（与 commit_phase2 逻辑一致）
        for outcome in &test_outcomes {
            match outcome {
                TestOutcome::Promote {
                    parent_name,
                    child_name,
                    test_input,
                } => {
                    self.stats.mutation_successes += 1;
                    // P1-fix: 变异成功计数（修复 total_mutation_successes 永远为 0 的断裂）
                    evolution.memory_mut().global_stats.total_mutation_successes += 1;
                    self.record_tried_mutation(
                        evolution,
                        parent_name,
                        Some(child_name),
                        true,
                        "mutation",
                        "变异成功",
                    );
                    self.mutation_failures.remove(parent_name);
                    println!(
                        "  ✅ 变异成功: {} → {} (测试+回归+AB 通过)",
                        parent_name, child_name
                    );
                    actions.push(format!("变异 {} → {} (成功)", parent_name, child_name));

                    let is_mutated = evolution
                        .genomes()
                        .get(parent_name)
                        .map(|g| g.lineage.origin == crate::genome::Origin::Mutated)
                        .unwrap_or(false);
                    if is_mutated {
                        if let Err(e) = evolution.remove_genome(parent_name) {
                            tracing::warn!("remove_genome 保存失败: {}", e);
                        }
                        self.bus.unregister(parent_name).await;
                        println!("  🗑️  淘汰旧版本: {}", parent_name);
                    }

                    if let Some(input) = test_input {
                        if let Some(g) = evolution.genomes_mut().get_mut(child_name.as_str()) {
                            g.add_test_case(input.clone(), true, "mutation_test");
                        }
                    }
                }
                TestOutcome::TestFailed {
                    parent_name,
                    child_name,
                    test_input,
                } => {
                    self.record_mutation_failure(parent_name);
                    self.record_tried_mutation(
                        evolution,
                        parent_name,
                        Some(child_name),
                        false,
                        "mutation_test_failure",
                        "变异测试失败",
                    );
                    println!("  ❌ 变异测试失败: {} → {}", parent_name, child_name);
                    actions.push(format!("变异 {} → {} (测试失败)", parent_name, child_name));

                    // P3b-fix: 在删除子代前提取失败代码，存入 lesson 供下轮归因做增量修复
                    let failed_code = evolution
                        .genomes()
                        .get(child_name.as_str())
                        .map(|g| {
                            g.actions
                                .iter()
                                .filter_map(|a| {
                                    a.implementation
                                        .code_string()
                                        .map(|c| format!("--- {} ---\n{}", a.name, c))
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    let mutation_desc = evolution
                        .genomes()
                        .get(child_name.as_str())
                        .and_then(|g| g.lineage.mutations.last().map(|m| m.description.clone()))
                        .unwrap_or_default();
                    let test_input_str = test_input
                        .as_ref()
                        .map(|i| serde_json::to_string(i).unwrap_or_default())
                        .unwrap_or_default();

                    if let Err(e) = evolution.remove_genome(child_name) {
                        tracing::warn!("remove_genome 保存失败: {}", e);
                    }
                    self.bus.unregister(child_name).await;
                    println!("  🗑️  淘汰失败变体: {}", child_name);

                    let lesson_text = if failed_code.is_empty() {
                        format!(
                            "变异 {} → {} 测试失败，变异方案可能不正确",
                            parent_name, child_name
                        )
                    } else {
                        format!(
                            "变异 {} → {} 测试失败。\n变异方案: {}\n测试输入: {}\n失败子代代码(下轮应在修复此代码基础上迭代):\n{}",
                            parent_name, child_name,
                            if mutation_desc.is_empty() { "未知".into() } else { mutation_desc },
                            test_input_str,
                            safe_truncate(&failed_code, 2000),
                        )
                    };
                    evolution.record_lesson(crate::evolution::EvolutionLesson {
                        lesson: lesson_text,
                        capability: parent_name.clone(),
                        failure_type: "mutation_test_failure".into(),
                        learned_at: format!(
                            "{}",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0)
                        ),
                        referenced_count: 0,
                    });
                }
                TestOutcome::RegressionFailed {
                    parent_name,
                    child_name,
                } => {
                    self.record_mutation_failure(parent_name);
                    self.record_tried_mutation(
                        evolution,
                        parent_name,
                        Some(child_name),
                        false,
                        "regression_failure",
                        "回归测试失败",
                    );
                    println!("  ❌ 回归测试失败: {} → {}", parent_name, child_name);
                    actions.push(format!(
                        "变异 {} → {} (回归测试失败)",
                        parent_name, child_name
                    ));
                    if let Err(e) = evolution.remove_genome(child_name) {
                        tracing::warn!("remove_genome 保存失败: {}", e);
                    }
                    self.bus.unregister(child_name).await;
                    println!("  🗑️  淘汰回归失败变体: {}", child_name);
                }
                TestOutcome::AbRolledBack {
                    parent_name,
                    child_name,
                } => {
                    self.record_mutation_failure(parent_name);
                    self.record_tried_mutation(
                        evolution,
                        parent_name,
                        Some(child_name),
                        false,
                        "ab_rollback",
                        "AB 对比失败",
                    );
                    println!(
                        "  ❌ AB 对比失败: {} → {} (父代更优)",
                        parent_name, child_name
                    );
                    actions.push(format!("变异 {} → {} (AB 回滚)", parent_name, child_name));
                    if let Err(e) = evolution.remove_genome(child_name) {
                        tracing::warn!("remove_genome 保存失败: {}", e);
                    }
                    self.bus.unregister(child_name).await;
                    println!("  🗑️  淘汰 AB 回滚变体: {}", child_name);
                }
            }
        }

        // 处理变异应用失败
        for (parent, err) in &phase1.mutation_failures {
            self.record_mutation_failure(parent);
            self.record_tried_mutation(
                evolution,
                parent,
                None,
                false,
                "mutation_apply_failure",
                err,
            );
            println!("  ❌ 变异应用失败: {}", err);
        }

        // 收集 4.5 自测试快照（不执行，仅取 genome 副本）
        let self_test_targets: Vec<(String, CapabilityGenome)> = evolution
            .genomes()
            .iter()
            .filter(|(_, g)| g.fitness.call_count == 0)
            .take(5)
            .map(|(name, g)| (name.clone(), g.clone()))
            .collect();

        // 收集 4.6 真实验证快照
        let op_keywords = [
            "git", "cargo", "make", "shell", "fs", "file", "ssh", "curl", "http", "npm", "pip",
            "brew", "rg", "jq", "sqlite", "rustc", "wasm",
        ];
        let validation_targets: Vec<(String, CapabilityGenome)> = evolution
            .genomes()
            .iter()
            .filter(|(_, g)| g.fitness.call_count > 0 && g.fitness.success_rate > 0.0)
            .filter(|(name, _)| op_keywords.iter().any(|k| name.contains(k)))
            .take(3)
            .map(|(name, g)| (name.clone(), g.clone()))
            .collect();

        Ok(CommitIntermediate {
            actions,
            report,
            self_test_targets,
            validation_targets,
            skipped: false,
        })
    }

    /// 锁外自测试 — 使用 genome 快照，不访问 EvolutionEngine
    pub async fn run_self_tests_unlocked(
        &self,
        targets: &[(String, CapabilityGenome)],
    ) -> Vec<SelfTestResult> {
        if targets.is_empty() {
            return Vec::new();
        }
        println!("  🧪 自测试: {} 个能力待测试 (锁外并行)", targets.len());

        let mut futures = Vec::new();
        for (_, genome) in targets {
            futures.push(self.test_capability_snapshot(genome));
        }
        let results = futures::future::join_all(futures).await;

        targets
            .iter()
            .zip(results.into_iter())
            .map(|((name, _), (pass, test_input))| SelfTestResult {
                name: name.clone(),
                pass,
                test_input,
            })
            .collect()
    }

    /// 锁外真实验证 — 使用 genome 快照，不访问 EvolutionEngine
    pub async fn run_validations_unlocked(
        &self,
        targets: &[(String, CapabilityGenome)],
    ) -> Vec<ValidationResult> {
        if targets.is_empty() {
            return Vec::new();
        }
        println!("  🔨 真实验证: {} 个能力待验证 (锁外并行)", targets.len());

        let mut results = Vec::new();
        for (name, genome) in targets {
            if genome.actions.is_empty() {
                continue;
            }
            let action_name = genome.actions[0].name.clone();
            let real_input = match Self::build_real_test_input(name, &action_name) {
                Some(input) => input,
                None => continue,
            };

            let cap = self.build_capability(genome.clone());
            let msg = crate::message::Message::builder()
                .from("real_validator")
                .to(name)
                .action(&action_name)
                .payload(real_input)
                .metadata(
                    crate::message::FITNESS_CLASS_METADATA,
                    crate::message::FITNESS_CLASS_AUTO_TEST,
                )
                .build();

            self.bus.register(Arc::new(cap)).await;
            let start = std::time::Instant::now();
            let result = self.bus.send(msg).await;
            let elapsed_ms = start.elapsed().as_millis() as u64;

            match result {
                Ok(resp) => {
                    let success = resp
                        .payload
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let evidence = if success {
                        format!(
                            "exit_code={}",
                            resp.payload
                                .get("exit_code")
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0)
                        )
                    } else {
                        resp.payload
                            .get("stderr")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string()
                    };
                    let strength = if success {
                        crate::validator::SignalStrength::RealTask
                    } else {
                        crate::validator::SignalStrength::ExitCode
                    };
                    results.push(ValidationResult {
                        name: name.clone(),
                        success,
                        evidence,
                        strength,
                        elapsed_ms,
                    });
                }
                Err(e) => {
                    results.push(ValidationResult {
                        name: name.clone(),
                        success: false,
                        evidence: format!("调用失败: {}", e),
                        strength: crate::validator::SignalStrength::ExitCode,
                        elapsed_ms,
                    });
                }
            }
        }
        results
    }

    /// 三阶段最终步骤（锁内）：写回自测试/真实验证结果 + 结晶化 + 淘汰 + 缺口 + 探索 + 持久化
    pub async fn commit_final(
        &mut self,
        evolution: &mut EvolutionEngine,
        intermediate: CommitIntermediate,
        self_test_results: Vec<SelfTestResult>,
        validation_results: Vec<ValidationResult>,
    ) -> Result<Vec<String>, String> {
        let report = intermediate.report;
        let mut actions = intermediate.actions;

        if intermediate.skipped {
            return self
                .run_post_mutation_steps(evolution, &report, actions)
                .await;
        }

        // 写回自测试结果
        for result in &self_test_results {
            self.stats.auto_tests += 1;
            if let Some(g) = evolution.genomes_mut().get_mut(&result.name) {
                g.fitness.record_auto_test(result.pass, 100.0);
                if let Some(input) = &result.test_input {
                    g.add_test_case(input.clone(), result.pass, "auto_test");
                }
            }
            if result.pass {
                self.stats.auto_test_passes += 1;
                println!("  ✅ 自测试通过: {}", result.name);
                actions.push(format!("自测试: {} (通过)", result.name));
            } else {
                println!("  ❌ 自测试失败: {}", result.name);
                actions.push(format!("自测试: {} (失败)", result.name));
            }
        }

        // 写回真实验证结果
        for result in &validation_results {
            let signal = crate::validator::RealWorldSignal {
                success: result.success,
                evidence: result.evidence.clone(),
                strength: result.strength,
            };
            if result.success {
                println!(
                    "  ✅ 真实验证通过: {} ({}ms, {:?})",
                    result.name, result.elapsed_ms, result.strength
                );
                actions.push(format!(
                    "真实验证: {} (通过, {:?})",
                    result.name, result.strength
                ));
                if let Some(g) = evolution.genomes_mut().get_mut(&result.name) {
                    g.fitness.record_auto_test(true, result.elapsed_ms as f64);
                    crate::validator::record_validation(g, &signal);
                    g.fitness.recompute_score();
                }
            } else {
                println!("  ❌ 真实验证失败: {} ({:?})", result.name, result.strength);
                actions.push(format!(
                    "真实验证: {} (失败, {:?})",
                    result.name, result.strength
                ));
                if let Some(g) = evolution.genomes_mut().get_mut(&result.name) {
                    g.fitness.record_auto_test(false, result.elapsed_ms as f64);
                    crate::validator::record_validation(g, &signal);
                    g.fitness.recompute_score();
                }
            }
        }

        // 运行后续步骤（4.7 结晶化 ~ 9 持久化）
        self.run_post_mutation_steps(evolution, &report, actions)
            .await
    }

    /// 三阶段编排 — 第三阶段（锁内）：写回测试结果 + 自测试 + 真实验证 + 结晶化 + 持久化
    ///
    /// 保留为非 daemon 调用方的便捷方法（一步到位，不拆 4.5/4.6 锁外）。
    pub async fn commit_phase2(
        &mut self,
        evolution: &mut EvolutionEngine,
        phase1: Phase1Result,
        test_outcomes: Vec<TestOutcome>,
    ) -> Result<Vec<String>, String> {
        let report = phase1.report;
        let mut actions = phase1.actions;

        if phase1.skipped {
            // 能力库为空时的后续步骤（缺口检测、好奇心探索等）
            return self
                .run_post_mutation_steps(evolution, &report, actions)
                .await;
        }

        // 写回测试结果
        for outcome in &test_outcomes {
            match outcome {
                TestOutcome::Promote {
                    parent_name,
                    child_name,
                    test_input,
                } => {
                    self.stats.mutation_successes += 1;
                    // P1-fix: 变异成功计数（修复 total_mutation_successes 永远为 0 的断裂）
                    evolution.memory_mut().global_stats.total_mutation_successes += 1;
                    self.record_tried_mutation(
                        evolution,
                        parent_name,
                        Some(child_name),
                        true,
                        "mutation",
                        "变异成功",
                    );
                    self.mutation_failures.remove(parent_name);
                    println!(
                        "  ✅ 变异成功: {} → {} (测试+回归+AB 通过)",
                        parent_name, child_name
                    );
                    actions.push(format!("变异 {} → {} (成功)", parent_name, child_name));

                    let is_mutated = evolution
                        .genomes()
                        .get(parent_name)
                        .map(|g| g.lineage.origin == crate::genome::Origin::Mutated)
                        .unwrap_or(false);
                    if is_mutated {
                        if let Err(e) = evolution.remove_genome(parent_name) {
                            tracing::warn!("remove_genome 保存失败: {}", e);
                        }
                        self.bus.unregister(parent_name).await;
                        println!("  🗑️  淘汰旧版本: {}", parent_name);
                    }

                    if let Some(input) = test_input {
                        if let Some(g) = evolution.genomes_mut().get_mut(child_name.as_str()) {
                            g.add_test_case(input.clone(), true, "mutation_test");
                        }
                    }
                }
                TestOutcome::TestFailed {
                    parent_name,
                    child_name,
                    test_input,
                } => {
                    self.record_mutation_failure(parent_name);
                    println!("  ❌ 变异测试失败: {} → {}", parent_name, child_name);
                    actions.push(format!("变异 {} → {} (测试失败)", parent_name, child_name));

                    // P3b-fix: 在删除子代前提取失败代码，存入 lesson 供下轮归因做增量修复
                    let failed_code = evolution
                        .genomes()
                        .get(child_name.as_str())
                        .map(|g| {
                            g.actions
                                .iter()
                                .filter_map(|a| {
                                    a.implementation
                                        .code_string()
                                        .map(|c| format!("--- {} ---\n{}", a.name, c))
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    let mutation_desc = evolution
                        .genomes()
                        .get(child_name.as_str())
                        .and_then(|g| g.lineage.mutations.last().map(|m| m.description.clone()))
                        .unwrap_or_default();
                    let test_input_str = test_input
                        .as_ref()
                        .map(|i| serde_json::to_string(i).unwrap_or_default())
                        .unwrap_or_default();

                    if let Err(e) = evolution.remove_genome(child_name) {
                        tracing::warn!("remove_genome 保存失败: {}", e);
                    }
                    self.bus.unregister(child_name).await;
                    println!("  🗑️  淘汰失败变体: {}", child_name);

                    let lesson_text = if failed_code.is_empty() {
                        format!(
                            "变异 {} → {} 测试失败，变异方案可能不正确",
                            parent_name, child_name
                        )
                    } else {
                        format!(
                            "变异 {} → {} 测试失败。\n变异方案: {}\n测试输入: {}\n失败子代代码(下轮应在修复此代码基础上迭代):\n{}",
                            parent_name, child_name,
                            if mutation_desc.is_empty() { "未知".into() } else { mutation_desc },
                            test_input_str,
                            safe_truncate(&failed_code, 2000),
                        )
                    };
                    evolution.record_lesson(crate::evolution::EvolutionLesson {
                        lesson: lesson_text,
                        capability: parent_name.clone(),
                        failure_type: "mutation_test_failure".into(),
                        learned_at: format!(
                            "{}",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0)
                        ),
                        referenced_count: 0,
                    });
                }
                TestOutcome::RegressionFailed {
                    parent_name,
                    child_name,
                } => {
                    self.record_mutation_failure(parent_name);
                    println!("  ❌ 回归测试失败: {} → {}", parent_name, child_name);
                    actions.push(format!(
                        "变异 {} → {} (回归测试失败)",
                        parent_name, child_name
                    ));
                    if let Err(e) = evolution.remove_genome(child_name) {
                        tracing::warn!("remove_genome 保存失败: {}", e);
                    }
                    self.bus.unregister(child_name).await;
                    println!("  🗑️  淘汰回归失败变体: {}", child_name);
                }
                TestOutcome::AbRolledBack {
                    parent_name,
                    child_name,
                } => {
                    self.record_mutation_failure(parent_name);
                    println!(
                        "  ❌ AB 对比失败: {} → {} (父代更优)",
                        parent_name, child_name
                    );
                    actions.push(format!("变异 {} → {} (AB 回滚)", parent_name, child_name));
                    if let Err(e) = evolution.remove_genome(child_name) {
                        tracing::warn!("remove_genome 保存失败: {}", e);
                    }
                    self.bus.unregister(child_name).await;
                    println!("  🗑️  淘汰 AB 回滚变体: {}", child_name);
                }
            }
        }

        // 处理变异应用失败
        for (parent, err) in &phase1.mutation_failures {
            self.record_mutation_failure(parent);
            println!("  ❌ 变异应用失败: {}", err);
        }

        // 运行后续步骤（4.5 自测试 ~ 9 持久化）
        self.run_post_mutation_steps(evolution, &report, actions)
            .await
    }

    /// 后续步骤：4.5 自测试 → 4.6 真实验证 → 4.7 结晶化 → 交叉重组 → 缺口填补 → 组合能力 → 评估 → 持久化
    async fn run_post_mutation_steps(
        &mut self,
        evolution: &mut EvolutionEngine,
        report: &IntrospectionReport,
        mut actions: Vec<String>,
    ) -> Result<Vec<String>, String> {
        // 4.5 自测试
        let untested: Vec<String> = evolution
            .genomes()
            .iter()
            .filter(|(_, g)| g.fitness.call_count == 0)
            .map(|(name, _)| name.clone())
            .collect();
        if !untested.is_empty() {
            println!("  🧪 自测试: {} 个能力待测试 (并行)", untested.len());
        }
        let to_test: Vec<String> = untested.iter().take(5).cloned().collect();
        if !to_test.is_empty() {
            let mut test_futures = Vec::new();
            for name in &to_test {
                test_futures.push(self.test_capability(evolution, name));
            }
            let results = futures::future::join_all(test_futures).await;

            for (name, (pass, test_input)) in to_test.iter().zip(results.into_iter()) {
                self.stats.auto_tests += 1;
                if let Some(g) = evolution.genomes_mut().get_mut(name) {
                    g.fitness.record_auto_test(pass, 100.0);
                    if let Some(input) = test_input {
                        g.add_test_case(input, pass, "auto_test");
                    }
                }
                if pass {
                    self.stats.auto_test_passes += 1;
                    println!("  ✅ 自测试通过: {}", name);
                    actions.push(format!("自测试: {} (通过)", name));
                } else {
                    println!("  ❌ 自测试失败: {}", name);
                    actions.push(format!("自测试: {} (失败)", name));
                }
            }
        }

        // 4.6 真实项目验证
        let to_validate: Vec<String> = evolution
            .genomes()
            .iter()
            .filter(|(_, g)| g.fitness.call_count > 0 && g.fitness.success_rate > 0.0)
            .filter(|(name, _)| {
                let op_keywords = [
                    "git", "cargo", "make", "shell", "fs", "file", "ssh", "curl", "http", "npm",
                    "pip", "brew", "rg", "jq", "sqlite", "rustc", "wasm",
                ];
                op_keywords.iter().any(|k| name.contains(k))
            })
            .map(|(name, _)| name.clone())
            .collect();
        if !to_validate.is_empty() {
            println!(
                "  🔨 真实验证: {} 个操作类能力待验证 (并行)",
                to_validate.len().min(3)
            );
        }
        let to_validate_3: Vec<String> = to_validate.iter().take(3).cloned().collect();
        if !to_validate_3.is_empty() {
            let mut validate_futures = Vec::new();
            for name in &to_validate_3 {
                validate_futures.push(self.validate_in_real_project(evolution, name));
            }
            let validate_results = futures::future::join_all(validate_futures).await;

            for (name, result) in to_validate_3.iter().zip(validate_results.into_iter()) {
                if let Some(result) = result {
                    let signal = crate::validator::RealWorldSignal {
                        success: result.success,
                        evidence: result.evidence.clone(),
                        strength: result.strength,
                    };
                    if result.success {
                        println!(
                            "  ✅ 真实验证通过: {} ({}ms, {:?})",
                            name, result.elapsed_ms, result.strength
                        );
                        actions.push(format!("真实验证: {} (通过, {:?})", name, result.strength));
                        if let Some(g) = evolution.genomes_mut().get_mut(name) {
                            g.fitness.record_auto_test(true, result.elapsed_ms as f64);
                            crate::validator::record_validation(g, &signal);
                            g.fitness.recompute_score();
                        }
                    } else {
                        println!(
                            "  ❌ 真实验证失败: {} ({:?}) — {}",
                            name,
                            result.strength,
                            &result.evidence[..100.min(result.evidence.len())]
                        );
                        actions.push(format!("真实验证: {} (失败, {:?})", name, result.strength));
                        if let Some(g) = evolution.genomes_mut().get_mut(name) {
                            g.fitness.record_auto_test(false, result.elapsed_ms as f64);
                            crate::validator::record_validation(g, &signal);
                            g.fitness.recompute_score();
                        }
                    }
                }
            }
        }

        // 4.7 结晶化：把高频 LLM 推理"冻结"为直接执行
        let crystallize_candidates = evolution.crystallize_candidates(3);
        if !crystallize_candidates.is_empty() {
            println!(
                "  💎 结晶化扫描: {} 个 LLM 动作在烧 token (top: {} = {} tokens)",
                crystallize_candidates.len(),
                crystallize_candidates[0].capability,
                crystallize_candidates[0].token_cost
            );
        }
        if let Some(candidate) = crystallize_candidates.first() {
            self.stats.crystallizations += 1;
            println!(
                "  💎 尝试结晶化: {}.{} (已消耗 {} tokens, {} 次调用)",
                candidate.capability, candidate.action, candidate.token_cost, candidate.call_count
            );
            match evolution
                .crystallize_action(&candidate.capability, &candidate.action)
                .await
            {
                Ok(new_genome) => {
                    println!(
                        "  ✅ 结晶化成功: {} → {} (LLM→Script, token 成本归零)",
                        candidate.capability, new_genome.name
                    );
                    actions.push(format!(
                        "结晶化: {}.{} → {} (利润率提升)",
                        candidate.capability, candidate.action, new_genome.name
                    ));
                    let mut cap = ScriptedCapability::from_genome(new_genome.clone());
                    cap = cap.with_llm(self.llm.clone()).with_bus(self.bus.clone());
                    if let Some(reg) = &self.executor_registry {
                        cap = cap.with_executor_registry(reg.clone());
                    }
                    self.bus.register(Arc::new(cap)).await;
                    self.stats.crystallization_successes += 1;
                }
                Err(e) => {
                    println!(
                        "  ⏭️  结晶化跳过: {}.{} — {}",
                        candidate.capability, candidate.action, e
                    );
                    actions.push(format!(
                        "结晶化尝试: {}.{} (跳过: {})",
                        candidate.capability, candidate.action, e
                    ));
                }
            }
        }

        // 5. 选择：淘汰长期无真实业务调用的能力
        const NEW_CAP_THRESHOLD: u32 = 20;
        const FAILED_CAP_THRESHOLD: u32 = 5;
        let mut to_eliminate = Vec::new();
        for (name, g) in evolution.genomes() {
            let real_calls = g.fitness.real_call_count();
            // P2-fix: 从未被真实调用 + 自测已失败的能力，不应享 20 轮宽限
            let threshold =
                if real_calls == 0 && g.fitness.auto_test_count > 0 && g.fitness.success_rate < 0.5
                {
                    FAILED_CAP_THRESHOLD // 自测都失败，快速淘汰
                } else if real_calls == 0 {
                    NEW_CAP_THRESHOLD // 从未被调用也没自测过，给充分宽限
                } else if g.fitness.score < 0.01 {
                    FAILED_CAP_THRESHOLD
                } else {
                    continue;
                };
            if g.fitness.rounds_dormant >= threshold {
                let dependents = evolution.find_dependents(name);
                if !dependents.is_empty() {
                    continue;
                }
                println!(
                    "  🗑️  自动淘汰: {} (连续 {} 轮无真实调用, 真实调用 {} 次, 自测试 {} 次)",
                    name, g.fitness.rounds_dormant, real_calls, g.fitness.auto_test_count
                );
                self.stats.eliminations += 1;
                actions.push(format!("淘汰 {}", name));
                to_eliminate.push(name.clone());
            }
        }
        for name in &to_eliminate {
            evolution.genomes_mut().remove(name);
            self.bus.unregister(name).await;
        }

        // 6. 环境感知：检测环境变化，发现能力缺口
        let gaps = self.detect_capability_gaps(evolution).await;
        let gaps_to_fill: Vec<String> = gaps.into_iter().take(3).collect();
        for gap in &gaps_to_fill {
            self.stats.gaps_found += 1;
            println!("  💡 发现能力缺口: {}", gap);
            if let Some(tool_name) = gap.split_whitespace().next() {
                self.tried_gaps
                    .insert(tool_name.to_string(), self.round_count);
            }
            if let Some(created) = self.fill_gap(evolution, gap).await {
                self.stats.gaps_filled += 1;
                evolution.reset_rounds_since_last_creation();
                println!("  🔧 自动填补缺口: 创造能力 {}", created);
                actions.push(format!("填补缺口: {} → {}", gap, created));
            } else {
                actions.push(format!("发现缺口: {} (未能自动填补)", gap));
            }
        }

        // 7. 好奇心探索
        evolution.increment_rounds_since_last_creation();
        let rounds_idle = evolution.rounds_since_last_creation();
        const PARADIGM_SHIFT_IDLE_ROUNDS: u32 = 15;
        const EXPLORATION_IDLE_ROUNDS: u32 = 5;
        let paradigm_shift =
            rounds_idle >= PARADIGM_SHIFT_IDLE_ROUNDS && report.total_capabilities > 0;
        let need_explore = (report.weak_capabilities.is_empty()
            && gaps_to_fill.is_empty()
            && rounds_idle >= EXPLORATION_IDLE_ROUNDS)
            || paradigm_shift;
        if need_explore {
            self.stats.explorations += 1;
            if paradigm_shift {
                println!(
                    "  ⚡ 范式跃迁: 连续 {} 轮无新能力，强制跳出当前领域探索...",
                    rounds_idle
                );
            } else if report.total_capabilities == 0 {
                println!("  🔬 好奇心驱动探索: 能力库为空，从零开始创造...");
            } else {
                println!("  🔬 好奇心驱动探索: 系统健康，主动寻找新能力方向...");
            }
            if let Some(created) = self.explore_new_capability(evolution, paradigm_shift).await {
                self.stats.explored_created += 1;
                evolution.reset_rounds_since_last_creation();
                println!("  🌱 探索创造新能力: {}", created);
                actions.push(format!("探索创造: {}", created));
            } else {
                println!("  💤 探索未产生新能力");
            }
        }

        // 8. 交叉重组
        if report.total_capabilities >= 2 && (self.stats.introspections % 3 == 0) {
            if let Some(created) = self.crossover_capabilities(evolution).await {
                self.stats.crossovers += 1;
                evolution.reset_rounds_since_last_creation();
                println!("  🧪 交叉重组: {}", created);
                actions.push(format!("交叉重组: {}", created));
            }
        }

        // 9.5 组合能力
        if report.total_capabilities >= 5 && (self.stats.introspections % 5 == 0) {
            if let Some(created) = self.create_composite_capability(evolution).await {
                evolution.reset_rounds_since_last_creation();
                println!("  🔗 组合能力: {}", created);
                actions.push(format!("组合能力: {}", created));
            }
        }

        // 9.6 LLM 语义评估
        if self.stats.introspections % 3 == 0 && report.total_capabilities > 0 {
            self.llm_evaluate_capabilities(evolution).await;
        }

        // 9. 持久化适应度
        evolution.save_fitness()?;

        Ok(actions)
    }

    async fn evolve_once_core(
        &mut self,
        evolution: &mut EvolutionEngine,
        mut precomputed_attribution: Option<(
            Vec<WeakCapability>,
            Vec<Option<AttributionResult>>,
            AttributionSnapshot,
        )>,
    ) -> Result<Vec<String>, String> {
        let mut actions = Vec::new();

        // 递增轮次计数
        self.round_count += 1;

        // 1. 自省：分析能力图谱
        let report = self.introspect(evolution);
        self.stats.introspections += 1;

        if report.total_capabilities == 0 {
            println!("  🔍 自省: 能力库为空，从零开始...");
            // 不直接返回，继续到缺口检测和好奇心探索
            // 跳过归因和变异（没有能力可归因）
        } else {
            println!(
                "  🔍 自省: {} 个能力, 平均适应度 {:.2}, {} 个弱能力, {} 个休眠能力, 多样性 {:.0}%",
                report.total_capabilities,
                report.avg_fitness,
                report.weak_capabilities.len(),
                report.dormant_capabilities.len(),
                report.diversity_score * 100.0,
            );
            // P5: 多样性低时输出重复组
            if !report.duplicate_groups.is_empty() {
                println!(
                    "  ⚠️  多样性警告: {} 个重复组: {}",
                    report.duplicate_groups.len(),
                    report
                        .duplicate_groups
                        .iter()
                        .map(|(base, versions)| format!("{}({})", base, versions.len()))
                        .collect::<Vec<_>>()
                        .join(", "),
                );

                // P5: 自动合并 — 淘汰重复组中适应度最低的版本（保留最高）
                for (_base, versions) in &report.duplicate_groups {
                    if versions.len() <= 1 {
                        continue;
                    }
                    // 找到适应度最高的版本
                    let best = versions.iter().max_by(|a, b| {
                        let fa = evolution
                            .genomes()
                            .get(*a)
                            .map(|g| g.fitness.score)
                            .unwrap_or(0.0);
                        let fb = evolution
                            .genomes()
                            .get(*b)
                            .map(|g| g.fitness.score)
                            .unwrap_or(0.0);
                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if let Some(best_name) = best {
                        for ver in versions {
                            if ver == best_name {
                                continue;
                            }
                            // 检查是否有依赖
                            let dependents = evolution.find_dependents(ver);
                            if !dependents.is_empty() {
                                continue;
                            }
                            // 淘汰低适应度版本
                            if let Some(g) = evolution.genomes().get(ver) {
                                if g.fitness.score
                                    < evolution
                                        .genomes()
                                        .get(best_name)
                                        .map(|bg| bg.fitness.score)
                                        .unwrap_or(0.0)
                                {
                                    println!(
                                        "  🔄 多样性合并: 淘汰 {} (适应度 {:.2} < {} 的 {:.2})",
                                        ver,
                                        g.fitness.score,
                                        best_name,
                                        evolution
                                            .genomes()
                                            .get(best_name)
                                            .map(|bg| bg.fitness.score)
                                            .unwrap_or(0.0)
                                    );
                                    if let Err(e) = evolution.remove_genome(ver) {
                                        tracing::warn!("remove_genome 保存失败: {}", e);
                                    }
                                    self.bus.unregister(ver).await;
                                    self.stats.eliminations += 1;
                                    actions.push(format!("多样性合并: 淘汰 {}", ver));
                                }
                            }
                        }
                    }
                }
            }

            // 2. 归因 + 变异：对每个弱能力分析原因并尝试改进
            // 并行归因（LLM 调用慢），串行变异（只替换代码，快），并行测试
            // 预计算模式携带生成归因时的完整 genome 快照；写回前会校验快照，
            // 避免锁外归因期间到达的人类反馈或其它变更被旧结论覆盖。
            let (weak_list, attr_results, attribution_snapshot): (
                Vec<WeakCapability>,
                Vec<Option<AttributionResult>>,
                Option<AttributionSnapshot>,
            ) = if let Some((wl, pre, snapshot)) = precomputed_attribution.take() {
                (wl, pre, Some(snapshot))
            } else {
                let wl: Vec<WeakCapability> = report
                    .weak_capabilities
                    .iter()
                    .take(3)
                    .filter(|weak| {
                        // P3a-fix: 用滑动窗口冷却替代永久封禁（只读判断，清除延后到 &mut self 上下文）
                        if self.is_mutation_in_cooldown(&weak.name) {
                            let fail_count = *self.mutation_failures.get(&weak.name).unwrap_or(&0);
                            println!(
                                "  ⏭️  跳过 {} (连续变异失败 {} 次, 冷却中)",
                                weak.name, fail_count
                            );
                            false
                        } else {
                            if *self.mutation_failures.get(&weak.name).unwrap_or(&0) >= 3 {
                                println!("  ♻️  变异冷却期结束, 重新尝试: {}", weak.name);
                            }
                            true
                        }
                    })
                    .cloned()
                    .collect();
                // 内联归因（无 daemon 编排时，如 mcp/测试调用）：仍持锁跑
                let attrs = self.attribute_weak_caps_inline(evolution, &wl).await;
                (wl, attrs, None)
            };

            // 2b. 串行变异（快，只替换代码 + 语法预检）
            let mut mutation_results: Vec<(String, Result<String, String>)> = Vec::new();
            for (weak, attr) in weak_list.iter().zip(attr_results.into_iter()) {
                if let Some(attr) = attr {
                    if let Some(snapshot) = &attribution_snapshot {
                        if !Self::attribution_snapshot_is_current(
                            evolution, &report, snapshot, weak,
                        ) {
                            tracing::info!(
                                "归因结果已过期，跳过写回: {}（归因期间能力或证据发生变化）",
                                weak.name
                            );
                            actions.push(format!("归因过期: {} (跳过变异)", weak.name));
                            continue;
                        }
                    }
                    if attr.mutation_plan.capability() != weak.name {
                        tracing::warn!(
                            "归因目标不一致，跳过: 弱能力={} 方案目标={}",
                            weak.name,
                            attr.mutation_plan.capability()
                        );
                        actions.push(format!("归因目标不一致: {} (跳过变异)", weak.name));
                        continue;
                    }
                    self.stats.attributions += 1;
                    println!("  🧠 归因: {} → {}", weak.name, attr.analysis);

                    // 记录归因思维链
                    evolution.record_thought_chain(crate::evolution::ThoughtChain {
                        chain_type: "attribution".to_string(),
                        reasoning: safe_truncate(&attr.analysis, 2000).to_string(),
                        conclusion: format!("变异方案: {:?}", attr.mutation_plan),
                        related_capabilities: vec![weak.name.clone()],
                        related_goal: None,
                        success: true,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0),
                    });

                    let result = self.apply_mutation(evolution, &attr.mutation_plan).await;
                    self.stats.mutations += 1;
                    mutation_results.push((weak.name.clone(), result));
                }
            }

            // 2c. 取 genome 快照（为锁外测试做准备）
            let test_targets: Vec<(String, String)> = mutation_results
                .iter()
                .filter_map(|(parent, result)| {
                    if let Ok(new_name) = result {
                        Some((parent.clone(), new_name.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            // 构建 MutationTestTarget 快照 — 之后可在锁外执行测试
            let test_snapshots: Vec<MutationTestTarget> = test_targets
                .iter()
                .filter_map(|(parent, child)| {
                    let parent_genome = evolution.genomes().get(parent)?.clone();
                    let child_genome = evolution.genomes().get(child)?.clone();
                    Some(MutationTestTarget {
                        parent_name: parent.clone(),
                        child_name: child.clone(),
                        parent_genome,
                        child_genome,
                    })
                })
                .collect();

            // 2d. 锁外测试 + 回归 + AB（使用快照版本，不访问 evolution 引用）
            let test_outcomes = self.test_and_select_unlocked(&test_snapshots).await;

            // 2e. 锁内写回测试结果
            for outcome in &test_outcomes {
                match outcome {
                    TestOutcome::Promote {
                        parent_name,
                        child_name,
                        test_input,
                    } => {
                        self.stats.mutation_successes += 1;
                        // P1-fix: 变异成功计数（修复 total_mutation_successes 永远为 0 的断裂）
                        evolution.memory_mut().global_stats.total_mutation_successes += 1;
                        self.mutation_failures.remove(parent_name);
                        println!(
                            "  ✅ 变异成功: {} → {} (测试+回归+AB 通过)",
                            parent_name, child_name
                        );
                        actions.push(format!("变异 {} → {} (成功)", parent_name, child_name));

                        // P0-2: 淘汰旧版本（仅变异体）
                        let is_mutated = evolution
                            .genomes()
                            .get(parent_name)
                            .map(|g| g.lineage.origin == crate::genome::Origin::Mutated)
                            .unwrap_or(false);
                        if is_mutated {
                            if let Err(e) = evolution.remove_genome(parent_name) {
                                tracing::warn!("remove_genome 保存失败: {}", e);
                            }
                            self.bus.unregister(parent_name).await;
                            println!("  🗑️  淘汰旧版本: {}", parent_name);
                        }

                        // P4: 保存测试用例
                        if let Some(input) = test_input {
                            if let Some(g) = evolution.genomes_mut().get_mut(child_name.as_str()) {
                                g.add_test_case(input.clone(), true, "mutation_test");
                            }
                        }
                    }
                    TestOutcome::TestFailed {
                        parent_name,
                        child_name,
                        test_input,
                    } => {
                        self.record_mutation_failure(parent_name);
                        println!("  ❌ 变异测试失败: {} → {}", parent_name, child_name);
                        actions.push(format!("变异 {} → {} (测试失败)", parent_name, child_name));

                        // P3b-fix: 在删除子代前提取失败代码，存入 lesson 供下轮归因做增量修复
                        let failed_code = evolution
                            .genomes()
                            .get(child_name.as_str())
                            .map(|g| {
                                g.actions
                                    .iter()
                                    .filter_map(|a| {
                                        a.implementation
                                            .code_string()
                                            .map(|c| format!("--- {} ---\n{}", a.name, c))
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            })
                            .unwrap_or_default();
                        let mutation_desc = evolution
                            .genomes()
                            .get(child_name.as_str())
                            .and_then(|g| g.lineage.mutations.last().map(|m| m.description.clone()))
                            .unwrap_or_default();
                        let test_input_str = test_input
                            .as_ref()
                            .map(|i| serde_json::to_string(i).unwrap_or_default())
                            .unwrap_or_default();

                        if let Err(e) = evolution.remove_genome(child_name) {
                            tracing::warn!("remove_genome 保存失败: {}", e);
                        }
                        self.bus.unregister(child_name).await;
                        println!("  🗑️  淘汰失败变体: {}", child_name);

                        let lesson_text = if failed_code.is_empty() {
                            format!(
                                "变异 {} → {} 测试失败，变异方案可能不正确",
                                parent_name, child_name
                            )
                        } else {
                            format!(
                                "变异 {} → {} 测试失败。\n变异方案: {}\n测试输入: {}\n失败子代代码(下轮应在修复此代码基础上迭代):\n{}",
                                parent_name, child_name,
                                if mutation_desc.is_empty() { "未知".into() } else { mutation_desc },
                                test_input_str,
                                safe_truncate(&failed_code, 2000),
                            )
                        };
                        evolution.record_lesson(crate::evolution::EvolutionLesson {
                            lesson: lesson_text,
                            capability: parent_name.clone(),
                            failure_type: "mutation_test_failure".into(),
                            learned_at: format!(
                                "{}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_secs())
                                    .unwrap_or(0)
                            ),
                            referenced_count: 0,
                        });
                    }
                    TestOutcome::RegressionFailed {
                        parent_name,
                        child_name,
                    } => {
                        self.record_mutation_failure(parent_name);
                        println!("  ❌ 回归测试失败: {} → {}", parent_name, child_name);
                        actions.push(format!(
                            "变异 {} → {} (回归测试失败)",
                            parent_name, child_name
                        ));
                        if let Err(e) = evolution.remove_genome(child_name) {
                            tracing::warn!("remove_genome 保存失败: {}", e);
                        }
                        self.bus.unregister(child_name).await;
                        println!("  🗑️  淘汰回归失败变体: {}", child_name);
                    }
                    TestOutcome::AbRolledBack {
                        parent_name,
                        child_name,
                    } => {
                        self.record_mutation_failure(parent_name);
                        println!(
                            "  ❌ AB 对比失败: {} → {} (父代更优)",
                            parent_name, child_name
                        );
                        actions.push(format!("变异 {} → {} (AB 回滚)", parent_name, child_name));
                        if let Err(e) = evolution.remove_genome(child_name) {
                            tracing::warn!("remove_genome 保存失败: {}", e);
                        }
                        self.bus.unregister(child_name).await;
                        println!("  🗑️  淘汰 AB 回滚变体: {}", child_name);
                    }
                }
            }

            // 处理变异应用失败
            for (parent, result) in &mutation_results {
                if let Err(e) = result {
                    self.record_mutation_failure(parent);
                    self.record_tried_mutation(
                        evolution,
                        parent,
                        None,
                        false,
                        "mutation_apply_failure",
                        e,
                    );
                    println!("  ❌ 变异应用失败: {}", e);
                }
            }

            // 4.5 自测试：对所有从未被调用过的能力执行自动测试（并行）
            //
            // P1-1: 并行执行自测试，加速进化循环
            // 注意：自测试只证明能力"能跑通"，不能证明能力"有用"。
            let untested: Vec<String> = evolution
                .genomes()
                .iter()
                .filter(|(_, g)| g.fitness.call_count == 0)
                .map(|(name, _)| name.clone())
                .collect();
            if !untested.is_empty() {
                println!("  🧪 自测试: {} 个能力待测试 (并行)", untested.len());
            }
            let to_test: Vec<String> = untested.iter().take(5).cloned().collect();
            if !to_test.is_empty() {
                // 并行执行测试
                let mut test_futures = Vec::new();
                for name in &to_test {
                    test_futures.push(self.test_capability(evolution, name));
                }
                let results = futures::future::join_all(test_futures).await;

                for (name, (pass, test_input)) in to_test.iter().zip(results.into_iter()) {
                    self.stats.auto_tests += 1;
                    if let Some(g) = evolution.genomes_mut().get_mut(name) {
                        g.fitness.record_auto_test(pass, 100.0);
                        // P4: 保存测试用例到持久化测试套件
                        if let Some(input) = test_input {
                            g.add_test_case(input, pass, "auto_test");
                        }
                    }
                    if pass {
                        self.stats.auto_test_passes += 1;
                        println!("  ✅ 自测试通过: {}", name);
                        actions.push(format!("自测试: {} (通过)", name));
                    } else {
                        println!("  ❌ 自测试失败: {}", name);
                        actions.push(format!("自测试: {} (失败)", name));
                    }
                }
            }

            // 4.6 真实项目验证：对自测试通过的操作类能力做真实验证
            //
            // 自测试用 LLM 生成合成输入，真实验证用预设的真实场景输入，
            // 确保能力不仅自报成功，而且在目标环境中确实成立；这仍不是用户需求信号。
            let to_validate: Vec<String> = evolution
                .genomes()
                .iter()
                .filter(|(_, g)| g.fitness.call_count > 0 && g.fitness.success_rate > 0.0)
                .filter(|(name, _)| {
                    let op_keywords = [
                        "git", "cargo", "make", "shell", "fs", "file", "ssh", "curl", "http",
                        "npm", "pip", "brew", "rg", "jq", "sqlite", "rustc", "wasm",
                    ];
                    op_keywords.iter().any(|k| name.contains(k))
                })
                .map(|(name, _)| name.clone())
                .collect();
            if !to_validate.is_empty() {
                println!(
                    "  🔨 真实验证: {} 个操作类能力待验证 (并行)",
                    to_validate.len().min(3)
                );
            }
            let to_validate_3: Vec<String> = to_validate.iter().take(3).cloned().collect();
            if !to_validate_3.is_empty() {
                let mut validate_futures = Vec::new();
                for name in &to_validate_3 {
                    validate_futures.push(self.validate_in_real_project(evolution, name));
                }
                let validate_results = futures::future::join_all(validate_futures).await;

                for (name, result) in to_validate_3.iter().zip(validate_results.into_iter()) {
                    if let Some(result) = result {
                        // 重建 RealWorldSignal 供 record_validation 记录正/负反馈 + 最强信号
                        let signal = crate::validator::RealWorldSignal {
                            success: result.success,
                            evidence: result.evidence.clone(),
                            strength: result.strength,
                        };
                        if result.success {
                            println!(
                                "  ✅ 真实验证通过: {} ({}ms, {:?})",
                                name, result.elapsed_ms, result.strength
                            );
                            actions
                                .push(format!("真实验证: {} (通过, {:?})", name, result.strength));
                            if let Some(g) = evolution.genomes_mut().get_mut(name) {
                                // 环境探针属于自动评测，不得伪造真实用户调用或清零 dormant。
                                g.fitness.record_auto_test(true, result.elapsed_ms as f64);
                                // 环境验证器背书 → 记正反馈 + 升级最强信号
                                crate::validator::record_validation(g, &signal);
                                g.fitness.recompute_score();
                            }
                        } else {
                            println!(
                                "  ❌ 真实验证失败: {} ({:?}) — {}",
                                name,
                                result.strength,
                                &result.evidence[..100.min(result.evidence.len())]
                            );
                            actions
                                .push(format!("真实验证: {} (失败, {:?})", name, result.strength));
                            if let Some(g) = evolution.genomes_mut().get_mut(name) {
                                // 失败探针仍属自动评测；强负证据单独写 real_validation_failures。
                                g.fitness.record_auto_test(false, result.elapsed_ms as f64);
                                // 负反馈:记 real_validation_failures,压低真实轨通过比
                                crate::validator::record_validation(g, &signal);
                                g.fitness.recompute_score();
                            }
                        }
                    }
                }
            }

            // 4.7 结晶化：把高频 LLM 推理"冻结"为直接执行
            //
            // 能量维度的核心进化步骤：找出烧 token 最多的 LLM 能力，
            // 尝试将其编译为等价的 Script 实现。
            // 成功后该能力不再消耗 token，利润率飙升。
            let crystallize_candidates = evolution.crystallize_candidates(3);
            if !crystallize_candidates.is_empty() {
                println!(
                    "  💎 结晶化扫描: {} 个 LLM 动作在烧 token (top: {} = {} tokens)",
                    crystallize_candidates.len(),
                    crystallize_candidates[0].capability,
                    crystallize_candidates[0].token_cost
                );
            }
            // 每轮最多结晶 1 个（最耗能的），避免 LLM 调用过多
            if let Some(candidate) = crystallize_candidates.first() {
                self.stats.crystallizations += 1;
                println!(
                    "  💎 尝试结晶化: {}.{} (已消耗 {} tokens, {} 次调用)",
                    candidate.capability,
                    candidate.action,
                    candidate.token_cost,
                    candidate.call_count
                );
                match evolution
                    .crystallize_action(&candidate.capability, &candidate.action)
                    .await
                {
                    Ok(new_genome) => {
                        println!(
                            "  ✅ 结晶化成功: {} → {} (LLM→Script, token 成本归零)",
                            candidate.capability, new_genome.name
                        );
                        actions.push(format!(
                            "结晶化: {}.{} → {} (利润率提升)",
                            candidate.capability, candidate.action, new_genome.name
                        ));
                        // 注册新能力到消息总线
                        let mut cap = ScriptedCapability::from_genome(new_genome.clone());
                        cap = cap.with_llm(self.llm.clone()).with_bus(self.bus.clone());
                        if let Some(reg) = &self.executor_registry {
                            cap = cap.with_executor_registry(reg.clone());
                        }
                        self.bus.register(Arc::new(cap)).await;
                        // 更新统计
                        self.stats.crystallization_successes += 1;
                    }
                    Err(e) => {
                        println!(
                            "  ⏭️  结晶化跳过: {}.{} — {}",
                            candidate.capability, candidate.action, e
                        );
                        actions.push(format!(
                            "结晶化尝试: {}.{} (跳过: {})",
                            candidate.capability, candidate.action, e
                        ));
                    }
                }
            }
            // 5. 选择：淘汰长期无真实业务调用的能力（负反馈/耗散机制）
            //
            // 关键修复：用 real_call_count（= call_count - auto_test_count）判断是否"被真正使用"，
            // 而非 call_count。这样自测试过但从未被真实业务调用的能力也能被淘汰，
            // 恢复耗散负反馈，防止能力库无限膨胀（正反馈失控 → 系统相变）。
            //
            // 淘汰规则：
            // - 从未被真实调用 + 自测失败（real_calls==0 && auto_test>0 && success_rate<0.5）→ 5 轮淘汰
            // - 从未被真实调用（real_calls == 0）+ rounds_dormant >= NEW_CAP_THRESHOLD → 淘汰
            // - 有真实调用但成功率极低（score < 0.01）+ rounds_dormant >= FAILED_CAP_THRESHOLD → 淘汰
            const NEW_CAP_THRESHOLD: u32 = 20; // 新能力 20 轮宽限期
            const FAILED_CAP_THRESHOLD: u32 = 5; // 失败能力 5 轮宽限期
            let mut to_eliminate = Vec::new();
            for (name, g) in evolution.genomes() {
                let real_calls = g.fitness.real_call_count();
                // P2-fix: 从未被真实调用 + 自测已失败的能力，不应享 20 轮宽限
                let threshold = if real_calls == 0
                    && g.fitness.auto_test_count > 0
                    && g.fitness.success_rate < 0.5
                {
                    FAILED_CAP_THRESHOLD // 自测都失败，快速淘汰
                } else if real_calls == 0 {
                    NEW_CAP_THRESHOLD // 从未被调用也没自测过，给充分宽限
                } else if g.fitness.score < 0.01 {
                    FAILED_CAP_THRESHOLD
                } else {
                    continue;
                };
                if g.fitness.rounds_dormant >= threshold {
                    // P3-3: 检查是否有 Composite 能力依赖此能力
                    let dependents = evolution.find_dependents(name);
                    if !dependents.is_empty() {
                        println!(
                            "  ⏭️  跳过淘汰: {} (被 {} 个能力依赖: {})",
                            name,
                            dependents.len(),
                            dependents.join(", ")
                        );
                        continue;
                    }
                    println!(
                        "  🗑️  自动淘汰: {} (连续 {} 轮无真实调用, 真实调用 {} 次, 自测试 {} 次)",
                        name, g.fitness.rounds_dormant, real_calls, g.fitness.auto_test_count
                    );
                    self.stats.eliminations += 1;
                    actions.push(format!("淘汰 {}", name));
                    to_eliminate.push(name.clone());
                }
            }
            // 实际从进化引擎中移除，并从总线注销（防止幽灵能力）
            for name in &to_eliminate {
                evolution.genomes_mut().remove(name);
                self.bus.unregister(name).await;
            }
        }

        // 6. 环境感知：检测环境变化，发现能力缺口
        let gaps = self.detect_capability_gaps(evolution).await;
        let gaps_to_fill: Vec<String> = gaps.into_iter().take(3).collect();
        for gap in &gaps_to_fill {
            self.stats.gaps_found += 1;
            println!("  💡 发现能力缺口: {}", gap);
            if let Some(tool_name) = gap.split_whitespace().next() {
                self.tried_gaps
                    .insert(tool_name.to_string(), self.round_count);
            }

            if let Some(created) = self.fill_gap(evolution, gap).await {
                self.stats.gaps_filled += 1;
                evolution.reset_rounds_since_last_creation();
                println!("  🔧 自动填补缺口: 创造能力 {}", created);
                actions.push(format!("填补缺口: {} → {}", gap, created));
            } else {
                actions.push(format!("发现缺口: {} (未能自动填补)", gap));
            }
        }

        // 7. 好奇心探索：如果本轮没有弱能力和缺口，主动探索新能力方向
        //    范式跃迁：连续 PARADIGM_SHIFT_IDLE_ROUNDS 轮无新能力创造时触发
        //    （软触发，而非每 10 轮强制，避免与收敛机制矛盾）
        evolution.increment_rounds_since_last_creation();
        let rounds_idle = evolution.rounds_since_last_creation();
        let paradigm_shift =
            rounds_idle >= PARADIGM_SHIFT_IDLE_ROUNDS && report.total_capabilities > 0;
        let need_explore = (report.weak_capabilities.is_empty()
            && gaps_to_fill.is_empty()
            && rounds_idle >= EXPLORATION_IDLE_ROUNDS)  // 至少 N 轮无创造才探索
            || paradigm_shift;
        if need_explore {
            self.stats.explorations += 1;
            if paradigm_shift {
                println!(
                    "  ⚡ 范式跃迁: 连续 {} 轮无新能力，强制跳出当前领域探索...",
                    rounds_idle
                );
            } else if report.total_capabilities == 0 {
                println!("  🔬 好奇心驱动探索: 能力库为空，从零开始创造...");
            } else {
                println!("  🔬 好奇心驱动探索: 系统健康，主动寻找新能力方向...");
            }
            if let Some(created) = self.explore_new_capability(evolution, paradigm_shift).await {
                self.stats.explored_created += 1;
                evolution.reset_rounds_since_last_creation();
                println!("  🌱 探索创造新能力: {}", created);
                actions.push(format!("探索创造: {}", created));
            } else {
                println!("  💤 探索未产生新能力");
            }
        }

        // 8. 交叉重组：偶尔尝试组合现有能力产生新能力
        if report.total_capabilities >= 2 && (self.stats.introspections % 3 == 0) {
            if let Some(created) = self.crossover_capabilities(evolution).await {
                self.stats.crossovers += 1;
                evolution.reset_rounds_since_last_creation();
                println!("  🧪 交叉重组: {}", created);
                actions.push(format!("交叉重组: {}", created));
            }
        }

        // P2-1: 9.5 组合能力：偶尔生成 Composite 类型能力（编排现有能力）
        if report.total_capabilities >= 5 && (self.stats.introspections % 5 == 0) {
            if let Some(created) = self.create_composite_capability(evolution).await {
                evolution.reset_rounds_since_last_creation();
                println!("  🔗 组合能力: {}", created);
                actions.push(format!("组合能力: {}", created));
            }
        }

        // 9.6 LLM 语义评估：只生成探索元数据，不作为自我晋升证据
        if self.stats.introspections % 3 == 0 && report.total_capabilities > 0 {
            self.llm_evaluate_capabilities(evolution).await;
        }

        // 9. 持久化适应度
        evolution.save_fitness()?;

        Ok(actions)
    }

    /// LLM 语义评估 — 估计能力的创新性和潜在实用性
    ///
    /// 传统适应度只看调用次数和成功率，但有些能力可能从未被调用过
    /// 却很有价值（休眠能力），或者调用很多但功能重复（低创新性）。
    /// 这些分数只用于探索/展示，不能直接改写 fitness：让提出能力的 LLM
    /// 同时给自己打分并晋升，会形成没有外部压力的自证循环。选择仍由运行结果、
    /// 环境验证和人类反馈决定。
    async fn llm_evaluate_capabilities(&self, evolution: &mut EvolutionEngine) {
        // 选择需要评估的能力：从未被 LLM 评估过的，或评估时间超过 1 小时的
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let to_evaluate: Vec<String> = evolution
            .genomes()
            .iter()
            .filter(|(_, g)| {
                g.fitness.llm_evaluated_at == 0
                    || now.saturating_sub(g.fitness.llm_evaluated_at) > 3600
            })
            .take(5) // 每次最多评估 5 个，控制 LLM 成本
            .map(|(name, _)| name.clone())
            .collect();

        if to_evaluate.is_empty() {
            return;
        }

        println!("  🧠 LLM 评估: {} 个能力待评估", to_evaluate.len());

        // 构建能力摘要
        let cap_summaries: Vec<String> = to_evaluate
            .iter()
            .filter_map(|name| {
                evolution.genomes().get(name).map(|g| {
                    format!(
                        "- {} : {} (调用 {} 次, 成功率 {:.0}%)",
                        g.name,
                        g.description,
                        g.fitness.call_count,
                        g.fitness.success_rate * 100.0
                    )
                })
            })
            .collect();

        let all_caps: Vec<String> = evolution.genomes().keys().cloned().collect();

        let prompt = format!(
            r#"你是能力评估专家。请评估以下能力的创新性和实用性。

当前能力库共 {} 个能力: {}

待评估能力:
{}

请对每个待评估能力打分（0.0~1.0）：
- innovation_score: 创新性（是否提供独特功能，与已有能力的差异化程度）
- utility_score: 实用性（在实际开发和运维场景中的有用程度）

返回严格 JSON 数组:
[
  {{
    "capability": "能力名",
    "innovation_score": 0.8,
    "utility_score": 0.7,
    "reason": "简短评估理由"
  }}
]
只返回 JSON。"#,
            all_caps.len(),
            all_caps.join(", "),
            cap_summaries.join("\n")
        );

        let response = match self.llm.execute(&prompt, "smart:eval", None).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM 能力评估失败: {}", e);
                return;
            }
        };

        // 解析评估结果
        let json_str = extract_json_array(&response);
        let evaluations: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "LLM 评估 JSON 解析失败: {} | 原始: {}",
                    e,
                    &response[..200.min(response.len())]
                );
                return;
            }
        };

        let mut evaluated = 0u32;
        for eval in &evaluations {
            let cap_name = eval
                .get("capability")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            let innovation = eval
                .get("innovation_score")
                .and_then(|s| s.as_f64())
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let utility = eval
                .get("utility_score")
                .and_then(|s| s.as_f64())
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            if let Some(g) = evolution.genomes_mut().get_mut(cap_name) {
                let observed_score = g.fitness.score;
                g.fitness.innovation_score = innovation;
                g.fitness.utility_score = utility;
                g.fitness.llm_evaluated_at = now;

                println!(
                    "  📊 语义先验: {} — 创新性 {:.1}, 潜在实用性 {:.1}, 观测适应度保持 {:.2}",
                    cap_name, innovation, utility, observed_score
                );
                evaluated += 1;
            }
        }

        if evaluated > 0 {
            println!("  ✅ LLM 评估完成: {} 个能力已更新", evaluated);
        }
    }

    /// 持续进化 — 无目标自创生模式
    ///
    /// 系统持续自省、归因、变异、测试，直到：
    /// - 连续 `idle_threshold` 轮无进化动作（系统已收敛）
    /// - 收到终止信号（Ctrl+C）
    /// - 达到最大轮数
    pub async fn evolve_continuous(
        &mut self,
        evolution: &mut EvolutionEngine,
        max_rounds: u32,
        idle_threshold: u32,
        interval_secs: u64,
    ) -> Result<AutoEvolveStats, String> {
        let mut idle_count = 0u32;
        let mut round = 0u32;

        println!("🧬 ═══ 持续进化模式启动 ═══");
        println!("  最大轮数: {}", max_rounds);
        println!("  空闲阈值: {} 轮无动作则停止", idle_threshold);
        println!("  轮间隔: {}s", interval_secs);
        println!("  按 Ctrl+C 终止\n");

        // Ctrl+C 信号处理：graceful shutdown
        // 收到信号后保存进化记忆再退出，避免适应度数据丢失
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        let mut should_stop = false;

        while round < max_rounds && !should_stop {
            round += 1;
            println!("🧬 ── 第 {} 轮 ──", round);

            let actions = self.evolve_once(evolution).await?;

            // 只有真正的进化动作才算非空闲：
            // - 变异、淘汰、填补缺口、探索创造、交叉重组
            // 不算的动作：
            // - 自测试（只是验证，不是进化）
            // - 发现缺口但未能填补（失败不算进化）
            // - 变异测试失败（失败不算进化）
            let has_evolution_action = actions.iter().any(|a| {
                !a.starts_with("自测试")
                    && !a.starts_with("无")
                    && !a.starts_with("发现缺口")
                    && !a.contains("(测试失败)")
                    && !a.contains("(未能自动填补)")
            });

            if actions.is_empty() || !has_evolution_action {
                idle_count += 1;
                println!(
                    "  💤 无进化动作 (连续空闲 {} / {})",
                    idle_count, idle_threshold
                );
            } else {
                idle_count = 0;
                println!("  自主进化动作:");
                for a in &actions {
                    println!("    • {}", a);
                }
            }

            // 检查是否收敛
            if idle_count >= idle_threshold {
                println!("\n🧬 系统已收敛 — 连续 {} 轮无进化动作", idle_threshold);
                break;
            }

            // 打印当前状态
            let genomes = evolution.genomes();
            println!(
                "  📊 能力数: {} | 平均适应度: {:.2}",
                genomes.len(),
                genomes.values().map(|g| g.fitness.score).sum::<f64>()
                    / genomes.len().max(1) as f64,
            );

            // 等待下一轮（同时监听 Ctrl+C）
            if round < max_rounds && idle_count < idle_threshold {
                tokio::select! {
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {}
                    _ = &mut ctrl_c => {
                        println!("\n⚠️  收到 Ctrl+C 信号，正在保存进化记忆...");
                        should_stop = true;
                    }
                }
            }

            println!();
        }

        // 确保最终状态被持久化
        evolution.save_fitness()?;

        println!("🧬 ═══ 持续进化结束 (共 {} 轮) ═══\n", round);
        println!("{}", self.report());

        Ok(self.stats.clone())
    }

    /// 有目标进化 — 定向进化模式
    ///
    /// 用户给出进化目标，系统朝目标方向进化，直到：
    /// - 目标达成（LLM 判断目标已满足）
    /// - 达到最大轮数
    /// - 收到终止信号（Ctrl+C）
    pub async fn evolve_towards(
        &mut self,
        evolution: &mut EvolutionEngine,
        goal: &str,
        max_rounds: u32,
        interval_secs: u64,
    ) -> Result<AutoEvolveStats, String> {
        let mut round = 0u32;

        println!("🧬 ═══ 定向进化模式启动 ═══");
        println!("  目标: {}", goal);
        println!("  最大轮数: {}", max_rounds);
        println!("  按 Ctrl+C 终止\n");

        // Ctrl+C 信号处理
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        let mut should_stop = false;

        while round < max_rounds && !should_stop {
            round += 1;
            println!("🧬 ── 第 {} 轮 (目标: {}) ──", round, goal);

            // 先运行一轮无目标进化（修复弱能力）
            let actions = self.evolve_once(evolution).await?;
            if !actions.is_empty() {
                println!("  自主进化动作:");
                for a in &actions {
                    println!("    • {}", a);
                }
            }

            // 评估目标达成度
            let (achieved, assessment) = self.evaluate_goal(evolution, goal).await;

            println!("  🎯 目标评估: {}", assessment);

            if achieved {
                evolution.save_fitness()?;
                println!("\n🧬 ✅ 目标达成！定向进化结束 (共 {} 轮)\n", round);
                println!("{}", self.report());
                return Ok(self.stats.clone());
            }

            // 如果目标未达成，让 LLM 生成朝目标方向的新能力
            println!("  🧠 思考朝目标方向的进化策略...");
            let created = self.evolve_towards_goal(evolution, goal, &assessment).await;
            if let Some(name) = created {
                evolution.reset_rounds_since_last_creation();
                println!("  🧬 为目标创造新能力: {}", name);
            }

            // 等待下一轮（同时监听 Ctrl+C）
            if round < max_rounds {
                tokio::select! {
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {}
                    _ = &mut ctrl_c => {
                        println!("\n⚠️  收到 Ctrl+C 信号，正在保存进化记忆...");
                        should_stop = true;
                    }
                }
            }

            println!();
        }

        // 确保最终状态被持久化
        evolution.save_fitness()?;

        println!("🧬 ═══ 达到最大轮数，定向进化结束 (共 {} 轮) ═══\n", round);
        println!("{}", self.report());

        Ok(self.stats.clone())
    }

    /// 评估目标达成度
    async fn evaluate_goal(&self, evolution: &EvolutionEngine, goal: &str) -> (bool, String) {
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        let genomes_summary: Vec<String> = genomes
            .iter()
            .map(|g| {
                format!(
                    "{} (适应度:{:.2}, 动作:{})",
                    g.name,
                    g.fitness.score,
                    g.action_names().join(",")
                )
            })
            .collect();

        let prompt = format!(
            r#"你是一个进化目标评估器。判断当前能力库是否已经达成进化目标。

进化目标: {goal}

当前能力库:
{}

请判断目标是否已达成，返回严格 JSON:
{{
  "achieved": true或false,
  "assessment": "评估说明",
  "missing": "如果未达成，还缺什么能力或改进"
}}"#,
            genomes_summary.join("\n")
        );

        let result = self.llm.execute(&prompt, "smart:assess", None).await;
        match result {
            Ok(text) => {
                let json_str = extract_json(&text);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let achieved = v.get("achieved").and_then(|v| v.as_bool()).unwrap_or(false);
                    let assessment = v
                        .get("assessment")
                        .and_then(|v| v.as_str())
                        .unwrap_or("未知")
                        .to_string();
                    (achieved, assessment)
                } else {
                    (false, "目标评估解析失败".to_string())
                }
            }
            Err(e) => (false, format!("目标评估失败: {}", e)),
        }
    }

    /// 朝目标方向创造新能力
    async fn evolve_towards_goal(
        &self,
        evolution: &mut EvolutionEngine,
        goal: &str,
        assessment: &str,
    ) -> Option<String> {
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        let genomes_summary: Vec<String> = genomes
            .iter()
            .map(|g| {
                format!(
                    "{}: {} ({})",
                    g.name,
                    g.description,
                    g.action_names().join(",")
                )
            })
            .collect();

        let prompt = format!(
            r#"你是一个能力进化引擎。根据进化目标和当前评估，创造一个新能力来推进目标。

进化目标: {goal}
当前评估: {assessment}

当前能力库:
{}

平台: {} ({})

请创造一个新能力基因组来推进目标。返回严格 JSON（基因组格式）:
{{
  "name": "能力名",
  "version": "0.1.0",
  "description": "能力描述",
  "actions": [
    {{
      "name": "动作名",
      "description": "动作描述",
      "input_schema": {{"properties": {{}}}},
      "implementation": {{
        "type": "Script",
        "language": "python",
        "code": "Python代码",
        "timeout_secs": 30
      }}
    }}
  ],
  "fitness": {{}},
  "lineage": {{}}
}}"#,
            genomes_summary.join("\n"),
            self.platform.os,
            self.platform.arch
        );

        let result = self.llm.execute(&prompt, "coder:novel", None).await.ok()?;
        let json_str = extract_json(&result);
        let genome: CapabilityGenome = serde_json::from_str(json_str).ok()?;

        // P0-1: 语法预检
        if let Err(e) = Self::validate_genome_scripts(&genome).await {
            tracing::warn!("新能力语法预检失败: {}", e);
            return None;
        }

        let name = genome.name.clone();
        if let Err(e) = evolution.register_genome(genome) {
            tracing::warn!("register_genome 保存失败: {}", e);
        }

        // 注册到总线并测试
        let genome = evolution.genomes().get(&name)?.clone();
        let cap = self.build_capability(genome);
        self.bus.register(Arc::new(cap)).await;

        Some(name)
    }

    /// 自省：分析能力图谱
    ///
    /// 用 real_call_count 判断能力是否"真正被使用"：
    /// - 有真实业务调用 → 参与弱能力分析和适应度统计
    /// - 无真实业务调用（含从未调用、仅自测试） → 列为休眠
    pub fn introspect(&self, evolution: &EvolutionEngine) -> IntrospectionReport {
        let genomes = evolution.genomes();
        let total = genomes.len();

        let mut weak = Vec::new();
        let mut dormant = Vec::new();
        let mut total_score = 0.0;
        let mut scored_count = 0;

        for (name, genome) in genomes {
            let real_calls = genome.fitness.real_call_count();
            if real_calls > 0 {
                // 有真实业务调用，参与弱能力分析
                if genome.fitness.success_rate < 0.8 {
                    weak.push(WeakCapability {
                        name: name.clone(),
                        success_rate: genome.fitness.success_rate,
                        call_count: genome.fitness.call_count,
                        failure_count: genome.fitness.failure_count,
                        avg_latency_ms: genome.fitness.avg_latency_ms,
                        actions: genome.action_names(),
                    });
                }
                total_score += genome.fitness.score;
                scored_count += 1;
            } else if genome.fitness.auto_test_count > 0 {
                // P-变异门槛修复: 被自测过但无真实调用 —— 这正是"该被变异改进但当前卡死"的一群。
                // 旧逻辑把这类扔进 dormant 等死，导致 weak 恒空 → 0 变异(死锁门槛)。
                // 现在纳入 weak，让变异管线能触及它们。success_rate=0 排序时自动排到最前(最该处理)。
                weak.push(WeakCapability {
                    name: name.clone(),
                    success_rate: genome.fitness.success_rate,
                    call_count: genome.fitness.call_count,
                    failure_count: genome.fitness.failure_count,
                    avg_latency_ms: genome.fitness.avg_latency_ms,
                    actions: genome.action_names(),
                });
            } else {
                // 纯新生能力(从未被调用、也未被自测) → 仍列为休眠，给几轮先被自测的机会
                dormant.push(name.clone());
            }
        }

        // 按成功率排序，最差的优先处理
        weak.sort_by(|a, b| {
            a.success_rate
                .partial_cmp(&b.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let avg_fitness = if scored_count > 0 {
            total_score / scored_count as f64
        } else {
            0.0
        };

        // 计算能力图谱密度（简化版：统计 Composite 步骤中引用其他能力的数量）
        let mut references = 0;
        for genome in genomes.values() {
            for action in &genome.actions {
                if let ActionImpl::Composite { steps } = &action.implementation {
                    references += steps.len();
                }
            }
        }
        let graph_density = if total > 0 {
            references as f64 / (total * total) as f64
        } else {
            0.0
        };

        // P5: 多样性度量
        let (diversity_score, duplicate_groups) = evolution.diversity_metrics();

        IntrospectionReport {
            weak_capabilities: weak,
            dormant_capabilities: dormant,
            total_capabilities: total,
            avg_fitness,
            graph_density,
            diversity_score,
            duplicate_groups,
        }
    }

    /// 判断锁外生成的归因是否仍适用于当前能力。
    fn attribution_snapshot_is_current(
        evolution: &EvolutionEngine,
        report: &IntrospectionReport,
        snapshot: &AttributionSnapshot,
        weak: &WeakCapability,
    ) -> bool {
        if !report
            .weak_capabilities
            .iter()
            .any(|current| current.name == weak.name)
        {
            return false;
        }

        let Some(original) = snapshot.genomes.get(&weak.name) else {
            return false;
        };
        let Some(current) = evolution.genomes().get(&weak.name) else {
            return false;
        };

        match (
            serde_json::to_value(original),
            serde_json::to_value(current),
        ) {
            (Ok(original), Ok(current)) => original == current,
            _ => false,
        }
    }

    /// 归因快照 — 弱能力的只读副本 + 相关教训，用于无锁归因
    ///
    /// 归因（attribute_failure）只需读弱能力的 genome + 相关 lessons，不写 evolution。
    /// 把这些克隆出来做成快照后，归因可在不持有 shared 锁的情况下并行执行——
    /// 这消除了"进化循环持锁跨 LLM 调用"导致 socket 命令阻塞的根因。
    /// 快照是 KB 级，无内存压力。
    pub fn snapshot_for_attribution(
        evolution: &EvolutionEngine,
        weak_list: &[WeakCapability],
    ) -> AttributionSnapshot {
        let mut genomes = std::collections::HashMap::new();
        for w in weak_list {
            if let Some(g) = evolution.genomes().get(&w.name) {
                genomes.insert(w.name.clone(), g.clone());
            }
        }
        // 教训：只取与本次归因能力相关的
        let lessons: Vec<crate::evolution::EvolutionLesson> = evolution
            .memory()
            .lessons
            .iter()
            .filter(|l| {
                weak_list
                    .iter()
                    .any(|w| l.capability == w.name || l.failure_type == "mutation_test_failure")
            })
            .take(15)
            .cloned()
            .collect();
        AttributionSnapshot { genomes, lessons }
    }

    /// 无锁归因：用快照数据跑 LLM 归因，不访问 EvolutionEngine
    ///
    /// 与 attribute_failure 逻辑一致，但数据来自快照而非 &EvolutionEngine。
    /// 调用方在持有锁时 snapshot_for_attribution 取快照，释放锁后用本方法跑归因。
    pub async fn attribute_failure_snapshot(
        &self,
        snapshot: &AttributionSnapshot,
        weak: &WeakCapability,
    ) -> Option<AttributionResult> {
        let genome = snapshot.genomes.get(&weak.name)?;
        let action_summaries: Vec<String> = genome
            .actions
            .iter()
            .map(|a| {
                let impl_summary = match &a.implementation {
                    ActionImpl::Script { code, language, .. } => {
                        let truncated = if code.len() > 2000 {
                            format!(
                                "{}...（截断，共 {} 字符）",
                                safe_truncate(code, 2000),
                                code.len()
                            )
                        } else {
                            code.clone()
                        };
                        format!("Script({}): {}", language, truncated)
                    }
                    ActionImpl::Composite { steps } => {
                        format!(
                            "Composite({} steps): {:?}",
                            steps.len(),
                            steps.iter().map(|s| &s.capability).collect::<Vec<_>>()
                        )
                    }
                    ActionImpl::Llm { prompt, .. } => {
                        let truncated = if prompt.len() > 500 {
                            format!("{}...", safe_truncate(prompt, 500))
                        } else {
                            prompt.clone()
                        };
                        format!("Llm: {}", truncated)
                    }
                    ActionImpl::Rule { template } => format!("Rule: {}", template),
                    ActionImpl::Native { capability, action } => {
                        format!("Native: {} -> {}", capability, action)
                    }
                    ActionImpl::Shell { command, .. } => {
                        let truncated = if command.len() > 200 {
                            format!("{}...", safe_truncate(command, 200))
                        } else {
                            command.clone()
                        };
                        format!("Shell: {}", truncated)
                    }
                    ActionImpl::Custom {
                        executor_type,
                        params,
                    } => format!("Custom({}): {:?}", executor_type, params),
                };
                format!(
                    "  - action: {} | {} | input_schema: {} | impl: {}",
                    a.name,
                    a.description,
                    serde_json::to_string(&a.input_schema).unwrap_or_default(),
                    impl_summary
                )
            })
            .collect();

        let lessons_block = {
            let lessons: Vec<String> = snapshot
                .lessons
                .iter()
                .filter(|l| l.capability == weak.name || l.failure_type == "mutation_test_failure")
                .take(5)
                .map(|l| format!("  - {}", l.lesson))
                .collect();
            if lessons.is_empty() {
                String::new()
            } else {
                format!("历史教训（避免重复犯错）:\n{}\n", lessons.join("\n"))
            }
        };

        let prompt = format!(
            r#"你是一个能力进化分析器。以下能力表现不佳，请分析原因并给出变异方案。

能力: {} — {}
动作列表:
{}

表现数据:
- 成功率: {:.0}%
- 调用次数: {}
- 失败次数: {}
- 平均延迟: {:.0}ms

输入约定（Python 脚本运行时契约）:
- 输入通过预定义变量 `input`（dict 类型）访问，用 `input.get('field', default)` 或 `input['field']` 取值
- 不要用 sys.stdin、sys.argv 读取输入（运行时不通过这些通道传递）
- 脚本必须实际执行操作并 print JSON 输出，不要只定义函数而不调用
- 如果操作可能失败，用 try/except 捕获异常并 print json.dumps({{"error": str(e)}})，确保进程退出码为 0

{}
请分析失败原因，并给出具体的变异方案。
返回严格 JSON（mutation_type 决定携带的字段，不要携带无关字段）:
{{
  "analysis": "失败原因分析",
  "mutation_plan": {{
    "capability": "能力名",
    "action": "动作名",
    "mutation_type": "fix_script_patch",
    "find": "要定位的现有代码片段（必须精确存在于原代码）",
    "replace": "替换后的代码",
    "expected_improvement": "预期改进效果"
  }}
}}

mutation_type 必须是以下之一，且只携带对应字段：
- fix_script_patch（优先，适合小修）: {{ mutation_type, capability, action, find, replace, expected_improvement }}
  find 是原代码里要被替换的片段（必须精确且唯一存在），replace 是新代码。
  适合：加 try/except、修单个 bug、加参数校验、换一个函数调用。认知负荷低、出错面小。
  补丁示例（加异常兜底）:
    find: "result = df.describe()"
    replace: "try:\n    result = df.describe()\nexcept Exception as e:\n    print(json.dumps({{\"error\": str(e)}}))\n    return"
- fix_script（深度重构才用）: {{ mutation_type, capability, action, new_code, expected_improvement }}
  new_code 是完整重写后的代码。仅在需要大范围改动、无法用局部补丁表达时使用。
- fix_composite: {{ mutation_type, capability, action, new_steps, expected_improvement }}
- fix_prompt: {{ mutation_type, capability, action, new_prompt, expected_improvement }}"#,
            genome.name,
            genome.description,
            action_summaries.join("\n"),
            weak.success_rate * 100.0,
            weak.call_count,
            weak.failure_count,
            weak.avg_latency_ms,
            lessons_block,
        );

        let result = match self.llm.execute_conversation(
            &prompt,
            "smart:attribution",
            None,
            &[
                "请分析根因（代码逻辑错误/环境依赖缺失/参数处理不当）并给出最终的结构化归因结果。必须包含 analysis 和 mutation_plan 两个字段，mutation_type 和携带的字段必须匹配，只返回严格 JSON:\n{\"analysis\":\"根因分析总结\",\"mutation_plan\":{\"capability\":\"...\",\"action\":\"...\",\"mutation_type\":\"fix_script\",\"new_code\":\"...\",\"expected_improvement\":\"...\"}}",
            ],
        ).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("归因深度推理失败，回退到单次调用: {}", e);
                match self.llm.execute(&prompt, "smart:attribution", None).await {
                    Ok(text) => text,
                    Err(e) => {
                        tracing::warn!("归因 LLM 调用失败: {}", e);
                        return None;
                    }
                }
            }
        };
        let json_str = extract_json(&result);
        let parsed: AttributionResult = match serde_json::from_str(json_str) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "归因 JSON 解析失败: {} | 原始: {}",
                    e,
                    safe_truncate(&result, 200)
                );
                return None;
            }
        };
        Some(parsed)
    }

    /// 归因：用 LLM 分析弱能力为什么失败，并生成变异方案
    pub async fn attribute_failure(
        &self,
        evolution: &EvolutionEngine,
        weak: &WeakCapability,
    ) -> Option<AttributionResult> {
        let genome = evolution.genomes().get(&weak.name)?;

        // 精简 prompt：只传关键信息，不传整个 genome JSON
        let action_summaries: Vec<String> = genome
            .actions
            .iter()
            .map(|a| {
                let impl_summary = match &a.implementation {
                    ActionImpl::Script { code, language, .. } => {
                        let truncated = if code.len() > 2000 {
                            format!(
                                "{}...（截断，共 {} 字符）",
                                safe_truncate(code, 2000),
                                code.len()
                            )
                        } else {
                            code.clone()
                        };
                        format!("Script({}): {}", language, truncated)
                    }
                    ActionImpl::Composite { steps } => {
                        format!(
                            "Composite({} steps): {:?}",
                            steps.len(),
                            steps.iter().map(|s| &s.capability).collect::<Vec<_>>()
                        )
                    }
                    ActionImpl::Llm { prompt, .. } => {
                        let truncated = if prompt.len() > 500 {
                            format!("{}...", safe_truncate(prompt, 500))
                        } else {
                            prompt.clone()
                        };
                        format!("Llm: {}", truncated)
                    }
                    ActionImpl::Rule { template } => format!("Rule: {}", template),
                    ActionImpl::Native { capability, action } => {
                        format!("Native: {} -> {}", capability, action)
                    }
                    ActionImpl::Shell { command, .. } => {
                        let truncated = if command.len() > 200 {
                            format!("{}...", safe_truncate(command, 200))
                        } else {
                            command.clone()
                        };
                        format!("Shell: {}", truncated)
                    }
                    ActionImpl::Custom {
                        executor_type,
                        params,
                    } => format!("Custom({}): {:?}", executor_type, params),
                };
                format!(
                    "  - action: {} | {} | input_schema: {} | impl: {}",
                    a.name,
                    a.description,
                    serde_json::to_string(&a.input_schema).unwrap_or_default(),
                    impl_summary
                )
            })
            .collect();

        let prompt = format!(
            r#"你是一个能力进化分析器。以下能力表现不佳，请分析原因并给出变异方案。

能力: {} — {}
动作列表:
{}

表现数据:
- 成功率: {:.0}%
- 调用次数: {}
- 失败次数: {}
- 平均延迟: {:.0}ms

输入约定（Python 脚本运行时契约）:
- 输入通过预定义变量 `input`（dict 类型）访问，用 `input.get('field', default)` 或 `input['field']` 取值
- 不要用 sys.stdin、sys.argv 读取输入（运行时不通过这些通道传递）
- 脚本必须实际执行操作并 print JSON 输出，不要只定义函数而不调用
- 如果操作可能失败，用 try/except 捕获异常并 print json.dumps({{"error": str(e)}})，确保进程退出码为 0

{}
请分析失败原因，并给出具体的变异方案。
返回严格 JSON（mutation_type 决定携带的字段，不要携带无关字段）:
{{
  "analysis": "失败原因分析",
  "mutation_plan": {{
    "capability": "能力名",
    "action": "动作名",
    "mutation_type": "fix_script_patch",
    "find": "要定位的现有代码片段（必须精确存在于原代码）",
    "replace": "替换后的代码",
    "expected_improvement": "预期改进效果"
  }}
}}

mutation_type 必须是以下之一，且只携带对应字段：
- fix_script_patch（优先，适合小修）: {{ mutation_type, capability, action, find, replace, expected_improvement }}
  find 是原代码里要被替换的片段（必须精确且唯一存在），replace 是新代码。
  适合：加 try/except、修单个 bug、加参数校验、换一个函数调用。认知负荷低、出错面小。
  补丁示例（加异常兜底）:
    find: "result = df.describe()"
    replace: "try:\n    result = df.describe()\nexcept Exception as e:\n    print(json.dumps({{\"error\": str(e)}}))\n    return"
- fix_script（深度重构才用）: {{ mutation_type, capability, action, new_code, expected_improvement }}
  new_code 是完整重写后的代码。仅在需要大范围改动、无法用局部补丁表达时使用。
- fix_composite: {{ mutation_type, capability, action, new_steps, expected_improvement }}
- fix_prompt: {{ mutation_type, capability, action, new_prompt, expected_improvement }}"#,
            genome.name,
            genome.description,
            action_summaries.join("\n"),
            weak.success_rate * 100.0,
            weak.call_count,
            weak.failure_count,
            weak.avg_latency_ms,
            {
                // 注入跨代记忆中的进化教训
                let lessons: Vec<String> = evolution
                    .memory()
                    .lessons
                    .iter()
                    .filter(|l| {
                        l.capability == weak.name || l.failure_type == "mutation_test_failure"
                    })
                    .take(5)
                    .map(|l| format!("  - {}", l.lesson))
                    .collect();
                if lessons.is_empty() {
                    String::new()
                } else {
                    format!("历史教训（避免重复犯错）:\n{}\n", lessons.join("\n"))
                }
            },
        );

        // 使用 multi-turn 对话进行深度归因推理
        let result = match self.llm.execute_conversation(
            &prompt,
            "smart:attribution",
            None,
            &[
                "请分析根因（代码逻辑错误/环境依赖缺失/参数处理不当）并给出最终的结构化归因结果。必须包含 analysis 和 mutation_plan 两个字段，mutation_type 和携带的字段必须匹配，只返回严格 JSON:\n{\"analysis\":\"根因分析总结\",\"mutation_plan\":{\"capability\":\"...\",\"action\":\"...\",\"mutation_type\":\"fix_script\",\"new_code\":\"...\",\"expected_improvement\":\"...\"}}",
            ],
        ).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("归因深度推理失败，回退到单次调用: {}", e);
                // 回退到单次调用
                match self.llm.execute(&prompt, "smart:attribution", None).await {
                    Ok(text) => text,
                    Err(e) => {
                        tracing::warn!("归因 LLM 调用失败: {}", e);
                        return None;
                    }
                }
            }
        };
        let json_str = extract_json(&result);
        let parsed: AttributionResult = match serde_json::from_str(json_str) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "归因 JSON 解析失败: {} | 原始: {}",
                    e,
                    safe_truncate(&result, 200)
                );
                return None;
            }
        };

        // 记录归因思维链（通过 evolution 的不可变引用获取信息，思维链由调用方记录）
        // 这里通过返回值附带思维链信息
        Some(parsed)
    }

    /// P3a-fix: 检查能力是否在变异冷却期内（连续失败 3 次 + 未过冷却窗口）
    ///
    /// 与 tried_gaps 的 GAP_RETRY_ROUNDS 对称：失败能力在 MUTATION_RETRY_ROUNDS 轮后
    /// 自动解除封禁，允许重新被选为变异目标，实现渐进修复而非永久搁置。
    fn is_mutation_in_cooldown(&self, name: &str) -> bool {
        const MUTATION_RETRY_ROUNDS: u32 = 15;
        let fail_count = *self.mutation_failures.get(name).unwrap_or(&0);
        if fail_count < 3 {
            return false;
        }
        let last_fail_round = *self.mutation_failures_round.get(name).unwrap_or(&0);
        if self.round_count.saturating_sub(last_fail_round) >= MUTATION_RETRY_ROUNDS {
            // 冷却期已过，清除封禁，允许重试
            return false;
        }
        true
    }

    /// P3a-fix: 记录一次变异失败（同时更新失败次数和失败轮次）
    fn record_mutation_failure(&mut self, name: &str) {
        *self.mutation_failures.entry(name.to_string()).or_insert(0) += 1;
        self.mutation_failures_round
            .insert(name.to_string(), self.round_count);
    }

    /// 将一次变异尝试写入跨重启记忆，供 UI 时间线展示。
    fn record_tried_mutation(
        &self,
        evolution: &mut EvolutionEngine,
        parent_name: &str,
        child_name: Option<&str>,
        success: bool,
        fallback_type: &str,
        fallback_description: &str,
    ) {
        let (mutation_type, description) = child_name
            .and_then(|name| evolution.genomes().get(name))
            .and_then(|g| g.lineage.mutations.last())
            .map(|m| (m.mutation_type.clone(), m.description.clone()))
            .unwrap_or_else(|| (fallback_type.to_string(), fallback_description.to_string()));
        let tried_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());
        evolution.record_tried_mutation(crate::evolution::TriedMutation {
            capability: parent_name.to_string(),
            mutation_type,
            description,
            success,
            tried_at,
        });
    }

    /// 应用变异：根据变异方案修改基因组
    ///
    /// MutationPlan 是 enum，编译期保证每种变异类型只携带自己需要的字段。
    /// 这里通过 match 把 MutationPlan 变体与 ActionImpl 变体对齐，
    /// 类型不匹配会直接返回 Err（而非静默跳过）。
    pub async fn apply_mutation(
        &self,
        evolution: &mut EvolutionEngine,
        plan: &MutationPlan,
    ) -> Result<String, String> {
        let cap_name = plan.capability();
        let action_name = plan.action();
        let genome = evolution
            .genomes()
            .get(cap_name)
            .ok_or_else(|| format!("能力 '{}' 不存在", cap_name))?
            .clone();

        let mut new_genome = genome.clone();
        new_genome.lineage.parent = Some(cap_name.to_string());
        new_genome.lineage.origin = crate::genome::Origin::Mutated;
        new_genome.lineage.generation += 1;
        new_genome.record_mutation(plan.mutation_type_str(), plan.expected_improvement());

        // 找到目标动作
        let action = new_genome
            .actions
            .iter_mut()
            .find(|a| a.name == action_name)
            .ok_or_else(|| format!("动作 '{}' 不存在", action_name))?;

        // enum 对齐：MutationPlan 变体必须与 ActionImpl 变体匹配
        match (plan, &mut action.implementation) {
            (
                MutationPlan::FixScript { new_code, .. },
                ActionImpl::Script { code, language, .. },
            ) => {
                // P0-1: 语法预检 — Python 代码用 ast.parse 快速过滤语法错误
                if language == "python" {
                    Self::validate_python_syntax(new_code).await?;
                }
                *code = new_code.clone();
            }
            (
                MutationPlan::FixScriptPatch { find, replace, .. },
                ActionImpl::Script { code, language, .. },
            ) => {
                // 锚点必须精确且唯一匹配 —— 0 匹配或 >1 匹配都 fail fast(不猜)
                let count = code.matches(find.as_str()).count();
                if count == 0 {
                    return Err(format!(
                        "补丁锚点未找到(代码可能已变,或 find 与现有代码不一致): {}",
                        safe_truncate(find, 80)
                    ));
                }
                if count > 1 {
                    return Err(format!(
                        "补丁锚点匹配 {} 处,需更具体的上下文: {}",
                        count,
                        safe_truncate(find, 80)
                    ));
                }
                // 应用替换(只替换第一处,因为已确认唯一)
                let new_full = code.replacen(find.as_str(), replace.as_str(), 1);
                // 语法预检在"应用后的完整文件"上做 —— 片段单独可能不合法(如裸 except:)
                if language == "python" {
                    Self::validate_python_syntax(&new_full).await?;
                }
                *code = new_full;
            }
            (MutationPlan::FixComposite { new_steps, .. }, ActionImpl::Composite { steps }) => {
                let parsed_steps: Vec<crate::genome::CompositeStep> =
                    serde_json::from_value(new_steps.clone())
                        .map_err(|e| format!("new_steps 反序列化失败: {}", e))?;
                *steps = parsed_steps;
            }
            (MutationPlan::FixPrompt { new_prompt, .. }, ActionImpl::Llm { prompt, .. }) => {
                *prompt = new_prompt.clone();
            }
            (
                MutationPlan::FixCustomParams { new_params, .. },
                ActionImpl::Custom { params, .. },
            ) => {
                *params = new_params.clone();
            }
            // 类型不匹配：fail fast 而非静默跳过
            (plan, impl_kind) => {
                return Err(format!(
                    "变异类型 {:?} 不适用于动作 '{}' 的实现类型 {:?}",
                    plan.mutation_type_str(),
                    action_name,
                    impl_kind
                ));
            }
        }

        let new_name = format!("{}-v{}", cap_name, new_genome.lineage.generation);
        new_genome.name = new_name.clone();

        // 重置适应度（新变异体从零开始评估）
        new_genome.fitness = crate::genome::FitnessGene::default();

        if let Err(e) = evolution.register_genome(new_genome) {
            tracing::warn!("register_genome 保存失败: {}", e);
        }

        Ok(new_name)
    }

    /// 锁外测试 + 回归 + AB — 不访问 EvolutionEngine，可安全在锁外执行。
    ///
    /// 三阶段重构的核心：接受 genome 快照，使用快照版测试方法，
    /// 返回 TestOutcome 供锁内写回。
    pub async fn test_and_select_unlocked(
        &self,
        targets: &[MutationTestTarget],
    ) -> Vec<TestOutcome> {
        let mut outcomes = Vec::with_capacity(targets.len());

        // P3c-fix: 多样本测试（3 份输入取多数票），避免单次 LLM 生成不合理输入误杀变异体
        const TEST_SAMPLES: usize = 3;
        let mut test_futures = Vec::new();
        for target in targets {
            // 每个 target 跑 TEST_SAMPLES 次 test_capability_snapshot（每次生成不同输入）
            for _ in 0..TEST_SAMPLES {
                test_futures.push(self.test_capability_snapshot(&target.child_genome));
            }
        }
        let all_results = futures::future::join_all(test_futures).await;

        // 按 target 分组，每组 TEST_SAMPLES 个结果
        for (i, target) in targets.iter().enumerate() {
            let start = i * TEST_SAMPLES;
            let end = start + TEST_SAMPLES;
            let results: Vec<_> = all_results[start..end.min(all_results.len())].to_vec();
            let pass_count = results.iter().filter(|(p, _)| *p).count();
            let pass = pass_count >= 2; // 2/3 通过即算通过

            // 取首个通过的测试输入（用于后续回归和 AB），没有则取首个
            let test_input = results
                .iter()
                .find(|(p, _)| *p)
                .or_else(|| results.first())
                .and_then(|(_, i)| i.clone());

            if !pass {
                outcomes.push(TestOutcome::TestFailed {
                    parent_name: target.parent_name.clone(),
                    child_name: target.child_name.clone(),
                    test_input,
                });
                continue;
            }

            // 回归测试（快照版）
            let (regression_rate, regression_total) = self
                .run_regression_tests_snapshot(&target.parent_genome, &target.child_genome)
                .await;
            if regression_total > 0 && regression_rate < 0.5 {
                outcomes.push(TestOutcome::RegressionFailed {
                    parent_name: target.parent_name.clone(),
                    child_name: target.child_name.clone(),
                });
                continue;
            }

            // AB 对比（快照版）
            let ab_promote = if let Some(ref input) = test_input {
                self.ab_compare_snapshot(&target.parent_genome, &target.child_genome, input)
                    .await
            } else {
                true
            };

            if !ab_promote {
                outcomes.push(TestOutcome::AbRolledBack {
                    parent_name: target.parent_name.clone(),
                    child_name: target.child_name.clone(),
                });
                continue;
            }

            outcomes.push(TestOutcome::Promote {
                parent_name: target.parent_name.clone(),
                child_name: target.child_name.clone(),
                test_input,
            });
        }

        outcomes
    }

    /// 测试变异后的能力是否正常工作
    /// 返回 (是否通过, 测试输入)
    pub async fn test_capability(
        &self,
        evolution: &EvolutionEngine,
        capability_name: &str,
    ) -> (bool, Option<serde_json::Value>) {
        let genome = match evolution.genomes().get(capability_name) {
            Some(g) => g.clone(),
            None => return (false, None),
        };

        if genome.actions.is_empty() {
            return (false, None);
        }

        // 先 clone action 信息，因为 genome 会被 move
        let action_name = genome.actions[0].name.clone();
        let action_schema = genome.actions[0].input_schema.clone();
        let action_desc = genome.actions[0].description.clone();
        let cap_desc = genome.description.clone();

        // 用 LLM 生成真实测试数据（而非假数据）
        let test_input = self
            .generate_smart_test_input(
                capability_name,
                &cap_desc,
                &action_name,
                &action_desc,
                &action_schema,
            )
            .await;

        let cap = self.build_capability(genome);

        let msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(capability_name)
            .action(&action_name)
            .payload(test_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();

        // 先注册到总线
        self.bus.register(Arc::new(cap)).await;

        let result = self.bus.send(msg).await;

        match result {
            Ok(resp) => match resp.payload.get("success").and_then(|v| v.as_bool()) {
                Some(success) => (success, Some(test_input)),
                None => {
                    tracing::warn!(
                        "能力 '{}' 测试响应缺少 success 字段（协议违反），视为失败",
                        capability_name
                    );
                    (false, Some(test_input))
                }
            },
            Err(e) => {
                tracing::warn!("能力 '{}' 测试调用失败: {}", capability_name, e);
                (false, Some(test_input))
            }
        }
    }

    /// 测试能力（快照版本）— 不需要 EvolutionEngine 引用，可在锁外执行。
    ///
    /// 与 `test_capability` 功能相同，但接受 genome 快照而非从 evolution 引用获取。
    /// 锁内取快照 → 锁外调用此方法 → 锁内写回结果，消除长时间持锁。
    pub async fn test_capability_snapshot(
        &self,
        genome: &crate::genome::CapabilityGenome,
    ) -> (bool, Option<serde_json::Value>) {
        if genome.actions.is_empty() {
            return (false, None);
        }

        let capability_name = &genome.name;
        let action_name = genome.actions[0].name.clone();
        let action_schema = genome.actions[0].input_schema.clone();
        let action_desc = genome.actions[0].description.clone();
        let cap_desc = genome.description.clone();

        // LLM 调用在锁外执行
        let test_input = self
            .generate_smart_test_input(
                capability_name,
                &cap_desc,
                &action_name,
                &action_desc,
                &action_schema,
            )
            .await;

        let cap = self.build_capability(genome.clone());

        let msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(capability_name)
            .action(&action_name)
            .payload(test_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();

        self.bus.register(Arc::new(cap)).await;

        let result = self.bus.send(msg).await;

        match result {
            Ok(resp) => match resp.payload.get("success").and_then(|v| v.as_bool()) {
                Some(success) => (success, Some(test_input)),
                None => {
                    tracing::warn!(
                        "能力 '{}' 测试响应缺少 success 字段（协议违反），视为失败",
                        capability_name
                    );
                    (false, Some(test_input))
                }
            },
            Err(e) => {
                tracing::warn!("能力 '{}' 测试调用失败: {}", capability_name, e);
                (false, Some(test_input))
            }
        }
    }

    /// 回归测试（快照版本）— 不需要 EvolutionEngine 引用，可在锁外执行。
    pub async fn run_regression_tests_snapshot(
        &self,
        parent_genome: &crate::genome::CapabilityGenome,
        child_genome: &crate::genome::CapabilityGenome,
    ) -> (f64, usize) {
        if parent_genome.test_suite.is_empty() {
            return (1.0, 0);
        }

        if child_genome.actions.is_empty() {
            return (0.0, parent_genome.test_suite.len());
        }

        let child_name = &child_genome.name;
        let action_name = child_genome.actions[0].name.clone();
        let cap = self.build_capability(child_genome.clone());
        self.bus.register(Arc::new(cap)).await;

        let mut passed = 0usize;
        let total = parent_genome.test_suite.len();

        for tc in &parent_genome.test_suite {
            let msg = crate::message::Message::builder()
                .from("auto_evolver")
                .to(child_name)
                .action(&action_name)
                .payload(tc.input.clone())
                .metadata(
                    crate::message::FITNESS_CLASS_METADATA,
                    crate::message::FITNESS_CLASS_AUTO_TEST,
                )
                .build();

            match self.bus.send(msg).await {
                Ok(resp) => {
                    let success = resp
                        .payload
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    // P8-fix: 回归测试宽容判定
                    // 旧逻辑：success == expect_success 才算通过
                    // 问题：父代在 P5 前全部失败，test_suite 里 expect_success 全是 false。
                    //       子代变异后变好了（success=true），但与 expect_success=false 不匹配 → 误判回归失败。
                    // 新逻辑：
                    //   - 子代成功 → 永远算通过（变好了不算回归）
                    //   - 子代失败 + 期望成功 → 算回归失败（真退化）
                    //   - 子代失败 + 期望失败 → 算通过（和父代一样差，但没更差）
                    if success || !tc.expect_success {
                        passed += 1;
                    }
                }
                Err(_) => {
                    if !tc.expect_success {
                        passed += 1;
                    }
                }
            }
        }

        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            1.0
        };

        (pass_rate, total)
    }

    /// AB 对比（快照版本）— 不需要 EvolutionEngine 引用，可在锁外执行。
    pub async fn ab_compare_snapshot(
        &self,
        parent_genome: &crate::genome::CapabilityGenome,
        child_genome: &crate::genome::CapabilityGenome,
        test_input: &serde_json::Value,
    ) -> bool {
        if parent_genome.actions.is_empty() || child_genome.actions.is_empty() {
            return true;
        }

        let child_name = &child_genome.name;
        let action_name = child_genome.actions[0].name.clone();

        // 注册并测试子代
        let child_cap = self.build_capability(child_genome.clone());
        self.bus.register(Arc::new(child_cap)).await;

        let child_msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(child_name)
            .action(&action_name)
            .payload(test_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();
        let child_started = std::time::Instant::now();
        let child_result = self.bus.send(child_msg).await;
        let child_latency = child_started.elapsed().as_secs_f64() * 1000.0;
        let child_success = match &child_result {
            Ok(resp) => resp
                .payload
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        };

        // 注册并测试父代
        let parent_name = &parent_genome.name;
        let parent_action = parent_genome.actions[0].name.clone();
        let parent_cap = self.build_capability(parent_genome.clone());
        self.bus.register(Arc::new(parent_cap)).await;

        let parent_msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(parent_name)
            .action(&parent_action)
            .payload(test_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();
        let parent_started = std::time::Instant::now();
        let parent_result = self.bus.send(parent_msg).await;
        let parent_latency = parent_started.elapsed().as_secs_f64() * 1000.0;
        let parent_success = match &parent_result {
            Ok(resp) => resp
                .payload
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        };

        // AB 判定逻辑
        if child_success && !parent_success {
            return true;
        }
        if !child_success && parent_success {
            return false;
        }
        if !child_success && !parent_success {
            // P7-fix: 父子都失败 — 如果父代本身是 0% 成功率的弱能力，
            // 子代"同样失败"不代表更差。比较输出质量做宽容判定，
            // 给渐进修复留窗口（子代输出不比父代差 → 允许 promote）。
            let child_output_len = match &child_result {
                Ok(resp) => resp
                    .payload
                    .get("stdout")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len())
                    .unwrap_or(0),
                Err(_) => 0,
            };
            let parent_output_len = match &parent_result {
                Ok(resp) => resp
                    .payload
                    .get("stdout")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len())
                    .unwrap_or(0),
                Err(_) => 0,
            };
            // 子代输出不比父代差 → 允许 promote
            return child_output_len >= parent_output_len;
        }
        // 都成功 → 比延迟
        child_latency <= parent_latency * 2.0
    }

    /// P4: 回归测试 — 用父代的持久化测试套件验证变异体
    ///
    /// 返回 (通过率, 总测试数)
    pub async fn run_regression_tests(
        &self,
        evolution: &EvolutionEngine,
        parent_name: &str,
        child_name: &str,
    ) -> (f64, usize) {
        let parent = match evolution.genomes().get(parent_name) {
            Some(g) => g.clone(),
            None => return (0.0, 0),
        };

        if parent.test_suite.is_empty() {
            return (1.0, 0); // 无测试用例，视为通过
        }

        let child = match evolution.genomes().get(child_name) {
            Some(g) => g.clone(),
            None => return (0.0, 0),
        };

        if child.actions.is_empty() {
            return (0.0, parent.test_suite.len());
        }

        let action_name = child.actions[0].name.clone();
        let cap = self.build_capability(child);
        self.bus.register(Arc::new(cap)).await;

        let mut passed = 0usize;
        let total = parent.test_suite.len();

        for tc in &parent.test_suite {
            let msg = crate::message::Message::builder()
                .from("auto_evolver")
                .to(child_name)
                .action(&action_name)
                .payload(tc.input.clone())
                .metadata(
                    crate::message::FITNESS_CLASS_METADATA,
                    crate::message::FITNESS_CLASS_AUTO_TEST,
                )
                .build();

            match self.bus.send(msg).await {
                Ok(resp) => {
                    let success = resp
                        .payload
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    // P8-fix: 回归测试宽容判定（与 snapshot 版一致）
                    // 子代成功 → 永远算通过（变好了不算回归）
                    if success || !tc.expect_success {
                        passed += 1;
                    }
                }
                Err(_) => {
                    // 调用失败，如果期望成功则算不通过
                    if tc.expect_success {
                        // pass
                    } else {
                        passed += 1;
                    }
                }
            }
        }

        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            1.0
        };
        if total > 0 {
            println!(
                "  🧪 回归测试: {} → {} ({}/{} 通过, {:.0}%)",
                parent_name,
                child_name,
                passed,
                total,
                pass_rate * 100.0
            );
        }
        (pass_rate, total)
    }

    /// P3-AB: AB 对比测试 — 对比新旧版本在相同输入上的表现
    ///
    /// 返回 true 表示新版本可以推广，false 表示应回滚
    pub async fn ab_compare(
        &self,
        evolution: &EvolutionEngine,
        parent_name: &str,
        child_name: &str,
        test_input: &serde_json::Value,
    ) -> bool {
        let parent = match evolution.genomes().get(parent_name) {
            Some(g) => g.clone(),
            None => return true, // 父代不存在，无需对比
        };
        let child = match evolution.genomes().get(child_name) {
            Some(g) => g.clone(),
            None => return false,
        };

        if parent.actions.is_empty() || child.actions.is_empty() {
            return true;
        }

        let action_name = child.actions[0].name.clone();

        // 注册子代能力
        let child_cap = self.build_capability(child);
        self.bus.register(Arc::new(child_cap)).await;

        // 测试子代
        let child_msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(child_name)
            .action(&action_name)
            .payload(test_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();
        let child_started = std::time::Instant::now();
        let child_result = self.bus.send(child_msg).await;
        let child_latency = child_started.elapsed().as_secs_f64() * 1000.0;
        let child_success = match &child_result {
            Ok(resp) => resp
                .payload
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        };

        // 注册并测试父代
        let parent_action = parent.actions[0].name.clone();
        let parent_cap = self.build_capability(parent);
        self.bus.register(Arc::new(parent_cap)).await;

        let parent_msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(parent_name)
            .action(&parent_action)
            .payload(test_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();
        let parent_started = std::time::Instant::now();
        let parent_result = self.bus.send(parent_msg).await;
        let parent_latency = parent_started.elapsed().as_secs_f64() * 1000.0;
        let parent_success = match &parent_result {
            Ok(resp) => resp
                .payload
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        };

        // AB 判定逻辑：
        // - 子代成功且父代失败 → 推广
        // - 子代失败且父代成功 → 回滚
        // - 都成功 → 比延迟，子代不比父代慢 2 倍就推广
        // - 都失败 → 回滚；没有成功证据时不得晋升子代
        let promote = match (child_success, parent_success) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => child_latency <= parent_latency * 2.0,
            (false, false) => false,
        };

        if !promote {
            println!(
                "  ⚖️  AB 对比: {} vs {} — 子代未证明更优 (父:{}ms 子:{}ms), 回滚",
                parent_name, child_name, parent_latency as u64, child_latency as u64
            );
        }

        promote
    }

    /// 真实项目验证：对操作类能力在真实项目场景中验证
    ///
    /// 区别于自测试（用 LLM 生成合成输入），真实验证使用预设的真实场景输入，
    /// 在实际项目目录中执行，验证能力是否真的可用。
    pub async fn validate_in_real_project(
        &self,
        evolution: &EvolutionEngine,
        capability_name: &str,
    ) -> Option<RealValidationResult> {
        let genome = evolution.genomes().get(capability_name)?.clone();
        if genome.actions.is_empty() {
            return None;
        }

        // 只对操作类能力做真实验证（跳过分析类、监控类）
        let op_keywords = [
            "git", "cargo", "make", "shell", "fs", "file", "ssh", "curl", "http", "npm", "pip",
            "brew", "rg", "jq", "sqlite", "rustc", "wasm",
        ];
        let is_op_cap = op_keywords.iter().any(|k| capability_name.contains(k));
        if !is_op_cap {
            return None;
        }

        let action_name = genome.actions[0].name.clone();

        // 为操作类能力构造真实场景输入
        let real_input = match Self::build_real_test_input(capability_name, &action_name) {
            Some(input) => input,
            None => return None,
        };

        let cap = self.build_capability(genome);
        let msg = crate::message::Message::builder()
            .from("real_validator")
            .to(capability_name)
            .action(&action_name)
            .payload(real_input.clone())
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();

        self.bus.register(Arc::new(cap)).await;
        let start = std::time::Instant::now();
        let result = self.bus.send(msg).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(resp) => {
                // 环境验证器：把"能力自报成功"升级为"环境证明成功"
                //
                // 旧逻辑只看 resp.payload.success 字段，一个能力只要返回结构化 JSON
                // 就算成功，哪怕它声称做的事在真实世界里没发生。现在用匹配的验证器
                // 追加一次真实世界校验（cargo build / git status / 退出码），结果覆盖
                // 自报判定。fitness 的累计回写到调用方（run_evolution_loop）统一做，
                // 避免这里持 &mut evolution 与并行的其它验证 future 冲突。
                let mut validation_output = resp.payload.clone();
                if let Some(obj) = validation_output.as_object_mut() {
                    obj.insert("_validation_input".into(), real_input.clone());
                }
                let signal = self
                    .validators
                    .verify(capability_name, &action_name, &validation_output)
                    .await;

                let output = resp
                    .payload
                    .get("result")
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_default();

                Some(RealValidationResult {
                    capability: capability_name.to_string(),
                    action: action_name,
                    success: signal.success,
                    elapsed_ms,
                    output,
                    evidence: signal.evidence,
                    strength: signal.strength,
                })
            }
            Err(e) => {
                tracing::warn!("真实验证 '{}' 失败: {}", capability_name, e);
                Some(RealValidationResult {
                    capability: capability_name.to_string(),
                    action: action_name,
                    success: false,
                    elapsed_ms,
                    output: e.to_string(),
                    evidence: format!("总线调用失败: {}", e),
                    strength: crate::validator::SignalStrength::SelfReport,
                })
            }
        }
    }

    /// 为操作类能力构造真实场景测试输入
    fn build_real_test_input(cap_name: &str, action: &str) -> Option<serde_json::Value> {
        // 根据能力类型构造真实输入
        if cap_name.contains("git") {
            if action.contains("status") || action == "status" {
                return Some(serde_json::json!({"path": "."}));
            }
            if action.contains("log") || action == "log" {
                return Some(serde_json::json!({"path": ".", "limit": 5}));
            }
            if action.contains("diff") || action == "diff" {
                return Some(serde_json::json!({"path": ".", "cached": false}));
            }
        }
        if cap_name.contains("cargo") && (action.contains("run_cargo") || action == "run_cargo") {
            return Some(serde_json::json!({"command": "check", "cwd": "."}));
        }
        if cap_name.contains("rustc") && (action.contains("compile") || action == "compile") {
            return Some(serde_json::json!({"source_code": "fn main() { println!(\"hello\"); }"}));
        }
        if cap_name.contains("make") {
            return Some(serde_json::json!({"target": "help"}));
        }
        if cap_name.contains("rg") {
            return Some(serde_json::json!({"pattern": "fn main", "path": "."}));
        }
        if cap_name.contains("fs") || cap_name.contains("file") {
            return Some(serde_json::json!({"path": ".", "recursive": false}));
        }
        if cap_name.contains("curl") || cap_name.contains("http") {
            return Some(serde_json::json!({"url": "https://httpbin.org/get", "method": "GET"}));
        }
        if cap_name.contains("jq") {
            return Some(serde_json::json!({"input": "{\"a\":1,\"b\":2}", "filter": ".a"}));
        }
        if cap_name.contains("sqlite") {
            return Some(serde_json::json!({"query": "SELECT 1 as test", "database": ":memory:"}));
        }
        None
    }

    /// 检测能力缺口：根据环境分析还缺什么能力
    pub async fn detect_capability_gaps(&self, evolution: &EvolutionEngine) -> Vec<String> {
        let mut gaps = Vec::new();
        let existing: Vec<String> = evolution.genomes().keys().cloned().collect();

        // 检测环境中有哪些工具可用但还没有对应能力
        let env_tools: Vec<String> = self
            .platform
            .env
            .iter()
            .filter(|(k, v)| k.starts_with("has_") && v.as_str() == "true")
            .map(|(k, _)| k.strip_prefix("has_").unwrap_or(k).to_string())
            .collect();

        // 工具缺口检测：有工具但无对应能力，且最近未尝试过（滑动窗口）
        //
        // 滑动窗口机制：GAP_RETRY_ROUNDS 轮内的尝试会被跳过，
        // 之后自动过期，允许能力被淘汰后重新填补。
        for tool in &env_tools {
            let has_cap = existing.iter().any(|name| name.contains(tool));
            let recently_tried = self
                .tried_gaps
                .get(tool)
                .map(|&last_round| self.round_count.saturating_sub(last_round) < GAP_RETRY_ROUNDS)
                .unwrap_or(false);
            if !has_cap && tool != "python3" && tool != "node" && !recently_tried {
                gaps.push(format!("{} 操作能力 (有 {} 工具但无对应能力)", tool, tool));
            }
        }

        gaps
    }

    /// 自动填补能力缺口：根据缺口描述用 LLM 创造新能力
    pub async fn fill_gap(&self, evolution: &mut EvolutionEngine, gap: &str) -> Option<String> {
        let existing: Vec<String> = evolution.genomes().keys().cloned().collect();
        let prompt = format!(
            r#"创造一个新能力填补缺口。缺口: {gap}

已有能力: {}
平台: {} ({}) 工具: {}

返回精简 JSON 基因组（代码要短，description 一句话，不要多余字段）:
{{
  "name": "能力名",
  "version": "0.1.0",
  "description": "一句话描述",
  "actions": [{{"name": "动作名", "description": "描述", "input_schema": {{"properties": {{}}}}, "implementation": {{"type": "Script", "language": "python", "code": "简短Python代码", "timeout_secs": 30}}}}],
  "fitness": {{}},
  "lineage": {{}}
}}"#,
            existing.join(", "),
            self.platform.os,
            self.platform.arch,
            self.platform
                .env
                .iter()
                .filter(|(k, v)| k.starts_with("has_") && v.as_str() == "true")
                .map(|(k, _)| k.strip_prefix("has_").unwrap_or(k))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let result = match tokio::time::timeout(
            std::time::Duration::from_secs(120),
            self.llm.execute(&prompt, "coder:gapfill", None),
        )
        .await
        {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => {
                tracing::warn!("缺口填补 LLM 调用失败: {}", e);
                return None;
            }
            Err(_) => {
                tracing::warn!("缺口填补 LLM 调用超时 (120s)，跳过该缺口");
                return None;
            }
        };
        let json_str = extract_json(&result);
        let genome: CapabilityGenome = match serde_json::from_str(json_str) {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!(
                    "缺口填补 JSON 解析失败: {} | 原始前200字符: {}",
                    e,
                    safe_truncate(&result, 200)
                );
                return None;
            }
        };

        // P0-1: 语法预检
        if let Err(e) = Self::validate_genome_scripts(&genome).await {
            tracing::warn!("缺口填补语法预检失败: {}", e);
            return None;
        }

        let name = genome.name.clone();
        if let Err(e) = evolution.register_genome(genome) {
            tracing::warn!("register_genome 保存失败: {}", e);
        }

        // 注册到总线
        let genome = evolution.genomes().get(&name)?.clone();
        let cap = self.build_capability(genome);
        self.bus.register(Arc::new(cap)).await;

        Some(name)
    }

    /// 好奇心驱动探索：系统健康时主动探索新能力方向
    ///
    /// 让 LLM 自主分析当前能力库的认知边界，自己发现缺失的方向，
    /// 而不是从预设列表中选择。
    pub async fn explore_new_capability(
        &self,
        evolution: &mut EvolutionEngine,
        paradigm_shift: bool,
    ) -> Option<String> {
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        let cap_summary: Vec<String> = genomes
            .iter()
            .map(|g| {
                format!(
                    "{}: {} [{}]",
                    g.name,
                    g.description,
                    g.action_names().join(",")
                )
            })
            .collect();

        // 感知全部可用工具（包括 Python 包）
        let all_tools: Vec<String> = self
            .platform
            .env
            .iter()
            .filter(|(k, v)| {
                (k.starts_with("has_") || k.starts_with("has_py_")) && v.as_str() == "true"
            })
            .map(|(k, _)| {
                k.strip_prefix("has_")
                    .or_else(|| k.strip_prefix("has_py_"))
                    .unwrap_or(k)
                    .to_string()
            })
            .collect();

        // 第一步：让 LLM 自主分析当前能力库的认知边界
        let analysis_prompt = format!(
            r#"你是一个自主进化系统的认知分析器。

当前系统已拥有的能力:
{caps}

可用工具: {tools}

请分析：
1. 这些能力覆盖了哪些领域？
2. 这些能力有什么共同特征或局限？
3. 从系统的自主进化角度，你认为最值得探索的全新方向是什么？不要局限于软件开发领域——考虑任何可以通过计算实现的认知能力。
4. 给出一个具体的新能力提案。

返回 JSON:
{{
  "covered_domains": ["已覆盖领域1", "领域2"],
  "common_pattern": "当前能力的共同特征",
  "missing_direction": "你认为最值得探索的新方向及理由",
  "proposal": {{
    "name": "能力名",
    "version": "0.1.0",
    "description": "一句话描述",
    "actions": [{{"name": "动作名", "description": "描述", "input_schema": {{"properties": {{}}}}, "implementation": {{"type": "Script", "language": "python", "code": "简短Python代码", "timeout_secs": 30}}}}],
    "fitness": {{}},
    "lineage": {{}}
  }}
}}"#,
            caps = if cap_summary.is_empty() {
                "（空，系统刚启动）".to_string()
            } else {
                cap_summary.join("\n")
            },
            tools = all_tools.join(", "),
        );

        let shift_note = if paradigm_shift {
            "\n\n⚠️ 范式跃迁模式：系统已在当前领域停留太久，你需要提出一个与现有能力完全不同的方向。"
        } else {
            ""
        };

        let prompt = format!("{}{}", analysis_prompt, shift_note);

        let result = match self.llm.execute(&prompt, "smart:explore", None).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("探索 LLM 调用失败: {}", e);
                return None;
            }
        };
        let json_str = extract_json(&result);

        // 检查是否 LLM 认为不需要新能力
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
            if v.get("skip").and_then(|s| s.as_bool()).unwrap_or(false) {
                return None;
            }
            // 打印 LLM 的自主分析
            if let Some(domains) = v.get("covered_domains").and_then(|d| d.as_array()) {
                let names: Vec<String> = domains
                    .iter()
                    .filter_map(|d| d.as_str().map(String::from))
                    .collect();
                if !names.is_empty() {
                    println!("  🧐 自主分析: 已覆盖领域 [{}]", names.join(", "));
                }
            }
            if let Some(pattern) = v.get("common_pattern").and_then(|p| p.as_str()) {
                if !pattern.is_empty() {
                    println!("  💭 共同特征: {}", pattern);
                }
            }
            if let Some(direction) = v.get("missing_direction").and_then(|d| d.as_str()) {
                if !direction.is_empty() {
                    println!("  🧭 自主选择方向: {}", direction);
                }
            }
        }

        // 从 LLM 返回中提取能力基因组
        // LLM 可能返回 {"proposal": {...}} 或直接 {...}，用 serde_json::Value 提取
        let v: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| {
                tracing::warn!(
                    "探索 JSON 解析失败: {} | 原始前200字符: {}",
                    e,
                    safe_truncate(&result, 200)
                );
                e
            })
            .ok()?;
        let proposal = v.get("proposal").unwrap_or(&v);
        let genome: CapabilityGenome = match serde_json::from_value(proposal.clone()) {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!(
                    "探索基因组反序列化失败: {} | 原始前200字符: {}",
                    e,
                    safe_truncate(&result, 200)
                );
                return None;
            }
        };

        // P0-1: 语法预检
        if let Err(e) = Self::validate_genome_scripts(&genome).await {
            tracing::warn!("探索语法预检失败: {}", e);
            return None;
        }

        let name = genome.name.clone();
        if let Err(e) = evolution.register_genome(genome) {
            tracing::warn!("register_genome 保存失败: {}", e);
        }

        // 注册到总线
        let genome = evolution.genomes().get(&name)?.clone();
        let cap = self.build_capability(genome);
        self.bus.register(Arc::new(cap)).await;

        Some(name)
    }

    /// 交叉重组：取两个现有能力，组合产生新能力
    pub async fn crossover_capabilities(&self, evolution: &mut EvolutionEngine) -> Option<String> {
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        if genomes.len() < 2 {
            return None;
        }

        // 取适应度最高的两个能力作为父代
        let mut sorted = genomes.clone();
        sorted.sort_by(|a, b| {
            b.fitness
                .score
                .partial_cmp(&a.fitness.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let parent1 = &sorted[0];
        let parent2 = &sorted[1];

        let prompt = format!(
            r#"交叉重组两个能力，产生一个组合能力（Composite）：把父代1和父代2的真实动作串联成一条流水线。

父代1: {} — {} | 真实动作: {}
父代2: {} — {} | 真实动作: {}

要求:
1. 新能力的 implementation.type 必须为 "Composite"
2. steps 中 step1 调用父代1 的某个真实动作，step2 调用父代2 的某个真实动作
3. step2 的 input 可用 {{step1.result}} 引用 step1 的输出

返回精简 JSON 基因组:
{{
  "name": "新能力名",
  "version": "0.1.0",
  "description": "一句话描述",
  "actions": [{{"name": "动作名", "description": "描述", "input_schema": {{"properties": {{}}}}, "implementation": {{"type": "Composite", "steps": [{{"name": "step1", "capability": "{}", "action": "真实动作名", "input": {{}}}}, {{"name": "step2", "capability": "{}", "action": "真实动作名", "input": {{"data": "{{step1.result}}"}}}}]}}}}],
  "fitness": {{}},
  "lineage": {{"origin": "Crossbred"}}
}}"#,
            parent1.name,
            parent1.description,
            parent1.action_names().join(","),
            parent2.name,
            parent2.description,
            parent2.action_names().join(","),
            parent1.name,
            parent2.name,
        );

        let mut created = None;

        // 主路径：LLM 生成 Composite（含类型/引用校验）
        if let Some(result) = self
            .llm
            .execute(&prompt, "coder:crossover", None)
            .await
            .ok()
        {
            let json_str = extract_json(&result);
            if let Ok(genome) = serde_json::from_str::<CapabilityGenome>(json_str) {
                let is_composite = genome
                    .actions
                    .iter()
                    .any(|a| matches!(a.implementation, ActionImpl::Composite { .. }));
                let errors = Self::validate_composite_steps(evolution, &genome);
                if is_composite && errors.is_empty() {
                    let name = genome.name.clone();
                    if let Err(e) = evolution.register_genome(genome) {
                        tracing::warn!("register_genome 保存失败: {}", e);
                    }
                    let g = evolution.genomes().get(&name)?.clone();
                    let cap = self.build_capability(g);
                    self.bus.register(Arc::new(cap)).await;
                    created = Some(name);
                } else if !errors.is_empty() {
                    tracing::warn!("交叉重组: 步骤引用无效: {}", errors.join("; "));
                } else {
                    tracing::warn!("交叉重组: LLM 未返回 Composite 类型，丢弃");
                }
            } else {
                tracing::warn!("交叉重组: LLM 返回无法解析的 JSON");
            }
        } else {
            tracing::warn!("交叉重组: LLM 调用失败（可能为空响应），改用确定性兜底");
        }

        // 兜底：LLM 失败则把两个父代真实动作确定性串成 Composite
        if created.is_none() {
            let name =
                Self::unique_name(evolution, &format!("{}_x_{}", parent1.name, parent2.name));
            let genome = Self::build_composite_genome(parent1, parent2, &name);
            if let Err(e) = evolution.register_genome(genome.clone()) {
                tracing::warn!("register_genome 保存失败: {}", e);
            }
            let cap = self.build_capability(genome);
            self.bus.register(Arc::new(cap)).await;
            tracing::info!("交叉重组(确定性兜底): {}", name);
            created = Some(name);
        }

        created
    }

    /// 校验 Composite 能力的步骤是否都引用了真实存在的能力与动作
    fn validate_composite_steps(
        evolution: &EvolutionEngine,
        genome: &CapabilityGenome,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for a in &genome.actions {
            if let ActionImpl::Composite { steps } = &a.implementation {
                for s in steps {
                    match evolution.genomes().get(&s.capability) {
                        Some(g) => {
                            if !g.actions.iter().any(|x| x.name == s.action) {
                                errors.push(format!("动作 {}.{} 不存在", s.capability, s.action));
                            }
                        }
                        None => errors.push(format!("能力 {} 不存在", s.capability)),
                    }
                }
            }
        }
        errors
    }

    /// 确定性构造一个 Composite 能力：把 parent1 的首个动作与 parent2 的首个动作串联成流水线。
    /// 作为 LLM 调用失败时的兜底，保证"组合进化"真正发生（运行时编排现有能力）。
    fn build_composite_genome(
        parent1: &CapabilityGenome,
        parent2: &CapabilityGenome,
        name: &str,
    ) -> CapabilityGenome {
        let step1_action = parent1
            .actions
            .first()
            .map(|a| a.name.clone())
            .unwrap_or_else(|| "run".into());
        let step2_action = parent2
            .actions
            .first()
            .map(|a| a.name.clone())
            .unwrap_or_else(|| "run".into());
        let g = CapabilityGenome::new(
            name.to_string(),
            format!(
                "组合能力（确定性交叉）: {} × {}",
                parent1.name, parent2.name
            ),
        );
        let action = ActionGene {
            name: "run".into(),
            description: format!("串联 {} 与 {}", parent1.name, parent2.name),
            input_schema: serde_json::json!({"properties": {}}),
            implementation: ActionImpl::Composite {
                steps: vec![
                    CompositeStep {
                        name: "step1".into(),
                        capability: parent1.name.clone(),
                        action: step1_action,
                        input: serde_json::json!({}),
                    },
                    CompositeStep {
                        name: "step2".into(),
                        capability: parent2.name.clone(),
                        action: step2_action,
                        input: serde_json::json!({"data": "{{step1.result}}"}),
                    },
                ],
            },
        };
        g.with_action(action)
    }

    /// 生成不重名的能力名
    fn unique_name(evolution: &EvolutionEngine, base: &str) -> String {
        if !evolution.genomes().contains_key(base) {
            return base.to_string();
        }
        let mut i = 2;
        while evolution.genomes().contains_key(&format!("{}_{}", base, i)) {
            i += 1;
        }
        format!("{}_{}", base, i)
    }

    /// P2-1: 创建组合能力 — 让 LLM 分析现有能力并生成 Composite 类型的新能力
    ///
    /// 组合能力通过编排现有能力完成更复杂的任务，
    /// 例如 git_ops + code_quality_analyzer → 自动代码审查能力
    pub async fn create_composite_capability(
        &self,
        evolution: &mut EvolutionEngine,
    ) -> Option<String> {
        // 需要至少 3 个能力才能组合
        let genomes = evolution.genomes();
        if genomes.len() < 3 {
            return None;
        }

        // 选择适应度最高的能力作为候选组件
        let mut sorted: Vec<_> = genomes.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.fitness
                .score
                .partial_cmp(&a.1.fitness.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let candidates: Vec<String> = sorted
            .iter()
            .take(8)
            .map(|(name, g)| {
                format!(
                    "{}: {} (actions: {}, fitness: {:.2})",
                    name,
                    g.description,
                    g.action_names().join(","),
                    g.fitness.score
                )
            })
            .collect();

        let candidates_str = candidates.join("\n");
        let mut last_errors = String::new();
        let mut composite_genome: Option<CapabilityGenome> = None;

        // 最多尝试 2 次：首次失败（非 Composite / 步骤引用无效）时带错误反馈重试
        for attempt in 1..=2u32 {
            let mut prompt = format!(
                r#"你是一个能力进化引擎。请分析以下现有能力，创造一个组合能力（Composite 类型）。

现有能力（请严格使用下面列出的真实能力名与动作名）:
{}

组合能力通过编排现有能力的动作完成更复杂的任务。
例如：git_ops.status + code_quality_analyzer.analyze → 自动代码审查

重要约束:
1. implementation.type 必须为 "Composite"
2. 每个 step 的 capability 必须是上面列出的真实能力名，action 必须是该能力真实存在的动作名
3. step 之间可用 {{step名.result}} 传递上一步输出，用 {{input.field}} 引用输入

返回严格 JSON（基因组格式）:
{{
  "name": "组合能力名",
  "version": "0.1.0",
  "description": "组合能力描述",
  "actions": [{{
    "name": "动作名",
    "description": "动作描述",
    "input_schema": {{"properties": {{}}}},
    "implementation": {{
      "type": "Composite",
      "steps": [
        {{
          "name": "step1",
          "capability": "现有能力名",
          "action": "该能力的动作名",
          "input": {{"key": "value 或 {{input.field}} 引用"}}
        }},
        {{
          "name": "step2",
          "capability": "另一个能力名",
          "action": "动作名",
          "input": {{"data": "{{step1.result}}"}}
        }}
      ]
    }}
  }}],
  "fitness": {{}},
  "lineage": {{"origin": "Crossbred"}}
}}"#,
                candidates_str
            );

            if attempt > 1 {
                prompt.push_str(&format!(
                    "\n\n⚠️ 上一轮校验失败: {}。请务必让 implementation.type 为 Composite，且每个 step 的 capability/action 都来自上面的真实能力列表。",
                    last_errors
                ));
            }

            let result = match self.llm.execute(&prompt, "smart:composite", None).await {
                Ok(r) => r,
                Err(e) => {
                    last_errors = format!("LLM 调用失败: {}", e);
                    tracing::warn!("组合能力 LLM 调用失败: {}", e);
                    continue;
                }
            };
            let json_str = extract_json(&result);
            let g: CapabilityGenome = match serde_json::from_str(json_str) {
                Ok(g) => g,
                Err(e) => {
                    last_errors = format!("JSON 解析失败: {}", e);
                    tracing::warn!("组合能力 JSON 解析失败: {}", safe_truncate(&result, 200));
                    continue;
                }
            };

            let is_composite = g
                .actions
                .iter()
                .any(|a| matches!(a.implementation, ActionImpl::Composite { .. }));
            if !is_composite {
                last_errors = "implementation.type 不是 Composite".to_string();
                tracing::warn!("组合能力生成: 非 Composite，重试");
                continue;
            }

            let errors = Self::validate_composite_steps(evolution, &g);
            if !errors.is_empty() {
                last_errors = errors.join("; ");
                tracing::warn!("组合能力生成: 步骤引用无效 ({})，重试", last_errors);
                continue;
            }

            composite_genome = Some(g);
            break;
        }

        // 兜底：LLM 两次均未产出有效 Composite 时，用适应度最高的两个能力确定性组合
        if composite_genome.is_none() && sorted.len() >= 2 {
            let p1 = sorted[0].1;
            let p2 = sorted[1].1;
            let name = Self::unique_name(evolution, &format!("{}_combo_{}", p1.name, p2.name));
            let genome = Self::build_composite_genome(p1, p2, &name);
            if let Err(e) = evolution.register_genome(genome.clone()) {
                tracing::warn!("register_genome 保存失败: {}", e);
            }
            let cap = self.build_capability(genome);
            self.bus.register(Arc::new(cap)).await;
            tracing::info!("组合能力(确定性兜底): {}", name);
            return Some(name);
        }

        let genome = composite_genome?;

        let name = genome.name.clone();
        if let Err(e) = evolution.register_genome(genome) {
            tracing::warn!("register_genome 保存失败: {}", e);
        }

        // 注册到总线
        let genome = evolution.genomes().get(&name)?.clone();
        let cap = self.build_capability(genome);
        self.bus.register(Arc::new(cap)).await;

        Some(name)
    }

    /// 用 LLM 生成真实测试数据
    async fn generate_smart_test_input(
        &self,
        cap_name: &str,
        cap_desc: &str,
        action_name: &str,
        action_desc: &str,
        schema: &serde_json::Value,
    ) -> serde_json::Value {
        let schema_str = serde_json::to_string_pretty(schema).unwrap_or_default();

        let prompt = format!(
            r#"为能力测试生成真实合理的输入数据。

能力: {cap_name} — {cap_desc}
动作: {action_name} — {action_desc}
输入 Schema: {schema_str}

请生成一个真实场景下的测试输入，确保数据有意义、能触发核心逻辑。
返回严格 JSON（直接可用的输入对象，不要包裹在其他结构中）:"#
        );

        match self.llm.execute(&prompt, "fast:testinput", None).await {
            Ok(text) => {
                let json_str = extract_json(&text);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if v.is_object() {
                        return v;
                    }
                }
                // LLM 生成失败，回退到规则生成
                generate_test_input(schema)
            }
            Err(_) => generate_test_input(schema),
        }
    }

    /// 获取进化统计
    pub fn stats(&self) -> &AutoEvolveStats {
        &self.stats
    }

    /// 获取平台信息引用（供 MCP server 等外部调用者访问）
    pub fn platform(&self) -> &Platform {
        &self.platform
    }

    /// 将总线中的运行时统计合并回持久化 fitness。
    ///
    /// `ScriptedCapability` 只在运行时更新调用、延迟、输出质量和 token 成本；
    /// 人工反馈、环境验证、LLM 价值评估等选择证据则直接写入 genome。运行时
    /// fitness 是注册能力时的快照，因此不能整体覆盖 genome，否则注册后新增的
    /// 强证据会在下一次同步时丢失。
    fn merge_runtime_fitness(
        genome_fitness: &crate::genome::FitnessGene,
        runtime_fitness: &crate::genome::FitnessGene,
    ) -> crate::genome::FitnessGene {
        // 以持久化 fitness 为基底：新增的选择证据默认都由 genome 保留。
        let mut merged = genome_fitness.clone();

        // 仅合并 ScriptedCapability 实际会更新的运行时统计。
        // 同一探针可能同时由调用方和 runtime 记账；max 合并可避免重复计票，
        // 也不会丢掉在总线到达 handler 前就失败的 genome 侧评测记录。
        merged.call_count = merged.call_count.max(runtime_fitness.call_count);
        merged.success_count = merged.success_count.max(runtime_fitness.success_count);
        merged.failure_count = merged.failure_count.max(runtime_fitness.failure_count);
        merged.call_count = merged
            .call_count
            .max(merged.success_count.saturating_add(merged.failure_count));
        merged.success_rate = runtime_fitness.success_rate;
        merged.avg_latency_ms = runtime_fitness.avg_latency_ms;
        merged.output_quality = runtime_fitness.output_quality;
        merged.coverage_score = runtime_fitness.coverage_score;
        merged.non_empty_output_count = merged
            .non_empty_output_count
            .max(runtime_fitness.non_empty_output_count);
        merged.total_token_cost = runtime_fitness.total_token_cost;
        merged.last_token_cost = runtime_fitness.last_token_cost;
        merged.profit_ratio = runtime_fitness.profit_ratio;

        // 调用方与 runtime 都可能记录自动评测；取最大值合并同一次探针而不重复计票。
        merged.auto_test_count = merged.auto_test_count.max(runtime_fitness.auto_test_count);

        // 真实验证证据只能增强，不能被注册时的旧快照降级。
        merged.real_validation_passes = merged
            .real_validation_passes
            .max(runtime_fitness.real_validation_passes);
        merged.real_validation_failures = merged
            .real_validation_failures
            .max(runtime_fitness.real_validation_failures);
        merged.strongest_signal = merged
            .strongest_signal
            .max(runtime_fitness.strongest_signal);

        // 人工反馈正常只写 genome；若遇到状态不一致，则保留样本更多的一侧。
        if runtime_fitness.human_signals_count > merged.human_signals_count {
            merged.human_signals_count = runtime_fitness.human_signals_count;
            merged.human_score = runtime_fitness.human_score;
        }

        // 人工/验证反馈也会更新时间，保留两侧较新的时间戳。
        if runtime_fitness.last_evaluated > merged.last_evaluated {
            merged.last_evaluated = runtime_fitness.last_evaluated.clone();
        }

        merged
    }

    /// 同步运行时适应度到进化引擎
    ///
    /// 关键修复：
    /// 1. 只合并运行时统计，保留 genome 中的人工反馈、真实验证和价值评估
    /// 2. 合并 runtime 对自动评测/真实调用的分类计数
    /// 3. 只在真实调用计数本轮增加时清零 dormant，否则每轮递增
    /// 4. fail fast：用 ? 替代四层嵌套 if let，让不变量违反直接报错而非静默吞掉
    pub async fn sync_fitness(&self, evolution: &mut EvolutionEngine) -> Result<(), String> {
        self.sync_fitness_impl(evolution, true).await
    }

    /// 只刷新运行时统计，不推进休眠轮次。锁外归因写回前调用，既能观察归因期间
    /// 到达的真实请求，又不会把一次实际进化轮重复记为两轮 dormant。
    async fn refresh_runtime_fitness(&self, evolution: &mut EvolutionEngine) -> Result<(), String> {
        self.sync_fitness_impl(evolution, false).await
    }

    async fn sync_fitness_impl(
        &self,
        evolution: &mut EvolutionEngine,
        advance_dormancy: bool,
    ) -> Result<(), String> {
        // 通过总线获取能力列表
        let cap_names = self.bus.list_capabilities().await;

        // 记录哪些能力在总线上有注册
        let bus_caps: std::collections::HashSet<String> = cap_names.iter().cloned().collect();
        let mut runtime_snapshots = Vec::new();

        // 第一阶段只收集并验证全部快照，不改 evolution。任一能力响应损坏时整批失败，
        // 避免前半能力已推进 dormant、后半能力报错造成部分提交。
        for name in &cap_names {
            // 跳过原生能力（它们不是 ScriptedCapability，没有 runtime_fitness）
            // 用 is_native() 类型方法替代硬编码字符串列表
            let is_native = self
                .bus
                .get_capability(name)
                .await
                .map(|cap| cap.is_native())
                .unwrap_or(false);
            if is_native {
                continue;
            }
            if !evolution.genomes().contains_key(name) {
                tracing::warn!(
                    "sync_fitness: 能力 '{}' 在总线上但不在进化引擎中（幽灵能力），跳过同步",
                    name
                );
                continue;
            }

            // 尝试获取适应度：发送 __fitness__ 动作
            let msg = crate::message::Message::builder()
                .from("auto_evolver")
                .to(name)
                .action("__fitness__")
                .payload(serde_json::json!({}))
                .build();

            let resp = self.bus.send(msg).await.map_err(|e| {
                format!("sync_fitness: 能力 '{}' __fitness__ 调用失败: {}", name, e)
            })?;
            let fitness_json = resp
                .payload
                .get("fitness")
                .ok_or_else(|| format!("sync_fitness: 能力 '{}' 响应缺少 fitness 字段", name))?;
            let runtime_fitness: crate::genome::FitnessGene =
                serde_json::from_value(fitness_json.clone()).map_err(|e| {
                    format!("sync_fitness: 能力 '{}' fitness 反序列化失败: {}", name, e)
                })?;
            runtime_snapshots.push((name.clone(), runtime_fitness));
        }

        // 第二阶段统一提交已经完整验证的 runtime 快照。
        for (name, runtime_fitness) in runtime_snapshots {
            // bus 上有但 genomes 中没有的能力(幽灵能力): warn + continue, 不让整体 sync 失败
            // 这类能力可能是某处直接注册到 bus 但没调用 register_genome, 属于局部不一致,
            // 不应阻塞其他能力的 fitness 同步和后续进化阶段(变异/缺口检测/淘汰)
            let genome = match evolution.genomes_mut().get_mut(&name) {
                Some(g) => g,
                None => {
                    tracing::warn!(
                        "sync_fitness: 能力 '{}' 在总线上但不在进化引擎中（幽灵能力），跳过同步",
                        name
                    );
                    continue;
                }
            };

            let previous_real_calls = genome.fitness.real_call_count();
            let mut fitness = Self::merge_runtime_fitness(&genome.fitness, &runtime_fitness);

            // 休眠是“本轮是否新增真实调用”，不能用累计 real_call_count > 0；
            // 否则能力只要历史上被用过一次，就会永久保持非休眠。
            let prev_dormant = genome.fitness.rounds_dormant;
            let current_real_calls = fitness.real_call_count();
            if current_real_calls > previous_real_calls {
                fitness.rounds_dormant = 0;
            } else if advance_dormancy {
                fitness.rounds_dormant = prev_dormant.saturating_add(1);
            } else {
                fitness.rounds_dormant = prev_dormant;
            }
            // 用最新公式重算 score — 确保 genomes.json 中历史遗留的分数
            // (可能由旧公式计算) 在每次同步时升级为新公式结果
            fitness.recompute_score();
            genome.fitness = fitness;
        }

        // 对不在总线上但在进化引擎中的能力，也增加休眠计数
        for (_name, genome) in evolution.genomes_mut() {
            if advance_dormancy && !bus_caps.contains(_name) {
                // 能力未注册到总线，增加休眠计数
                genome.fitness.rounds_dormant = genome.fitness.rounds_dormant.saturating_add(1);
            }
            // 所有能力都用最新公式重算 score (覆盖 bus 上有/无两种情况)
            genome.fitness.recompute_score();
        }

        evolution.save_fitness()?;
        Ok(())
    }

    /// 生成自主进化报告
    pub fn report(&self) -> String {
        format!(
            "═══ 自主进化报告 ═══\n\
             自省次数: {}\n\
             归因次数: {}\n\
             自主变异: {} (成功 {})\n\
             缺口发现: {} (填补 {})\n\
             好奇探索: {} (创造 {})\n\
             交叉重组: {}\n\
             自测试: {} (通过 {})\n\
             自动淘汰: {}\n",
            self.stats.introspections,
            self.stats.attributions,
            self.stats.mutations,
            self.stats.mutation_successes,
            self.stats.gaps_found,
            self.stats.gaps_filled,
            self.stats.explorations,
            self.stats.explored_created,
            self.stats.crossovers,
            self.stats.auto_tests,
            self.stats.auto_test_passes,
            self.stats.eliminations,
        )
    }
}

/// 根据 input_schema 生成测试输入
fn generate_test_input(schema: &serde_json::Value) -> serde_json::Value {
    let mut test = serde_json::Map::new();

    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (key, schema) in props {
            let desc = schema
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let key_lower = key.to_lowercase();
            let desc_lower = desc.to_lowercase();
            let hint = format!("{} {}", key_lower, desc_lower);

            let value = match schema.get("type").and_then(|t| t.as_str()) {
                Some("string") => {
                    // 根据字段名和描述生成合理的真实测试值
                    if hint.contains("host") {
                        serde_json::Value::String("127.0.0.1".to_string())
                    } else if hint.contains("url") || hint.contains("endpoint") {
                        serde_json::Value::String("https://httpbin.org/get".to_string())
                    } else if hint.contains("path")
                        || hint.contains("file")
                        || hint.contains("目录")
                        || hint.contains("路径")
                    {
                        serde_json::Value::String("/tmp".to_string())
                    } else if hint.contains("db")
                        || hint.contains("database")
                        || hint.contains("数据库")
                    {
                        serde_json::Value::String(":memory:".to_string())
                    } else if hint.contains("sql") || hint.contains("query") {
                        serde_json::Value::String("SELECT 1".to_string())
                    } else if hint.contains("command")
                        || hint.contains("cmd")
                        || hint.contains("命令")
                    {
                        serde_json::Value::String("echo hello".to_string())
                    } else if hint.contains("json") {
                        serde_json::Value::String(r#"{"test": true}"#.to_string())
                    } else if hint.contains("pattern")
                        || hint.contains("filter")
                        || hint.contains("正则")
                    {
                        serde_json::Value::String("test".to_string())
                    } else if hint.contains("message")
                        || hint.contains("msg")
                        || hint.contains("消息")
                    {
                        serde_json::Value::String("test message".to_string())
                    } else if hint.contains("version") || hint.contains("版本") {
                        serde_json::Value::String("0.1.0".to_string())
                    } else if hint.contains("target") || hint.contains("make") {
                        serde_json::Value::String("all".to_string())
                    } else if hint.contains("script") {
                        serde_json::Value::String("test".to_string())
                    } else if hint.contains("package")
                        || hint.contains("pkg")
                        || hint.contains("包")
                    {
                        serde_json::Value::String("requests".to_string())
                    } else if hint.contains("data") || hint.contains("内容") {
                        serde_json::Value::String(r#"{"key": "value"}"#.to_string())
                    } else if hint.contains("text") || hint.contains("文本") {
                        serde_json::Value::String("Hello, world!".to_string())
                    } else if hint.contains("count") || hint.contains("数量") {
                        serde_json::Value::String("10".to_string())
                    } else if hint.contains("level") || hint.contains("级别") {
                        serde_json::Value::String("INFO".to_string())
                    } else {
                        serde_json::Value::String("test".to_string())
                    }
                }
                Some("integer") | Some("number") => {
                    if hint.contains("port") {
                        serde_json::json!(8080)
                    } else if hint.contains("count")
                        || hint.contains("limit")
                        || hint.contains("数量")
                    {
                        serde_json::json!(10)
                    } else if hint.contains("timeout") || hint.contains("超时") {
                        serde_json::json!(30)
                    } else if hint.contains("duration") || hint.contains("时长") {
                        serde_json::json!(3)
                    } else if hint.contains("depth") || hint.contains("深度") {
                        serde_json::json!(5)
                    } else {
                        serde_json::json!(42)
                    }
                }
                Some("boolean") => serde_json::json!(true),
                Some("array") => {
                    if hint.contains("ignore") || hint.contains("过滤") {
                        serde_json::json!(["*.tmp", "*.log"])
                    } else {
                        serde_json::json!([])
                    }
                }
                Some("object") => {
                    if hint.contains("header") || hint.contains("头") {
                        serde_json::json!({"Content-Type": "application/json"})
                    } else {
                        serde_json::json!({})
                    }
                }
                _ => serde_json::Value::String("test".to_string()),
            };
            test.insert(key.clone(), value);
        }
    }

    serde_json::Value::Object(test)
}

/// 安全截断字符串到指定字节长度（不在多字节字符中间截断）
///
/// 仅用于日志输出，不参与数据流，因此不是防御性代码。
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// 从 LLM 输出中提取 JSON
///
/// 只处理 markdown 代码块包裹（```json ... ```），不做容错修复。
/// 如果 LLM 返回的 JSON 格式错误，应该 fail fast 并改进 prompt，
/// 而不是用 repair_json 这样的"容错解析器"掩盖问题——
/// 后者会让 LLM 持续输出错误格式，防御代码越积越多。
fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // 处理 ```json ... ``` 包裹
    if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // 处理 ``` ... ``` 包裹
    if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // 无包裹，直接返回原文
    trimmed
}

/// 从 LLM 响应中提取 JSON 数组
fn extract_json_array(text: &str) -> &str {
    let trimmed = text.trim();

    // 处理 ```json ... ``` 包裹
    if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // 处理 ``` ... ``` 包裹
    if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // 尝试提取 [...] 数组
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if end > start {
                return &trimmed[start..=end];
            }
        }
    }

    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 MutationPlan::FixScript 能从 JSON 正确反序列化
    #[test]
    fn test_mutation_plan_fix_script_deserialize() {
        let json = r#"{
            "mutation_type": "fix_script",
            "capability": "git-tool",
            "action": "commit",
            "new_code": "print('hello')",
            "expected_improvement": "修复 bug"
        }"#;
        let plan: MutationPlan = serde_json::from_str(json).unwrap();
        match plan {
            MutationPlan::FixScript {
                capability,
                action,
                new_code,
                expected_improvement,
            } => {
                assert_eq!(capability, "git-tool");
                assert_eq!(action, "commit");
                assert_eq!(new_code, "print('hello')");
                assert_eq!(expected_improvement, "修复 bug");
            }
            _ => panic!("应该是 FixScript 变体"),
        }
    }

    /// 验证 MutationPlan::FixScriptPatch 能从 JSON 正确反序列化
    #[test]
    fn test_mutation_plan_fix_script_patch_deserialize() {
        let json = r#"{
            "mutation_type": "fix_script_patch",
            "capability": "py_tool",
            "action": "run",
            "find": "result = df.describe()",
            "replace": "try:\n    result = df.describe()\nexcept Exception:\n    result = None",
            "expected_improvement": "加异常兜底"
        }"#;
        let plan: MutationPlan = serde_json::from_str(json).unwrap();
        match plan {
            MutationPlan::FixScriptPatch {
                capability,
                action,
                find,
                replace,
                expected_improvement,
            } => {
                assert_eq!(capability, "py_tool");
                assert_eq!(action, "run");
                assert_eq!(find, "result = df.describe()");
                assert!(replace.contains("try:"));
                assert_eq!(expected_improvement, "加异常兜底");
            }
            _ => panic!("应该是 FixScriptPatch 变体"),
        }
    }

    /// 构造一个带 Python Script 动作的最小 genome + engine + evolver,供补丁测试
    fn make_patch_harness() -> (AutoEvolver, crate::evolution::EvolutionEngine) {
        use crate::genome::{ActionGene, ActionImpl, CapabilityGenome};
        let tmp = std::env::temp_dir().join(format!("patch_test_{}", uuid_str()));
        let mut evo = crate::evolution::EvolutionEngine::new(&tmp);
        let mut g = CapabilityGenome::new("py_cap", "test py capability");
        g.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Script {
                code: "import json\nimport sys\n\ndef run(p):\n    x = p['x']\n    return {'success': True, 'result': x * 2}\n".into(),
                language: "python".into(),
                timeout_secs: 30,
            },
        });
        evo.register_genome(g).unwrap();
        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());
        (evolver, evo)
    }

    fn uuid_str() -> String {
        use std::time::SystemTime;
        format!(
            "{:x}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        )
    }

    fn make_rule_genome(name: &str, success: bool) -> crate::genome::CapabilityGenome {
        use crate::genome::{ActionGene, ActionImpl, CapabilityGenome};

        let mut genome = CapabilityGenome::new(name, "rule capability for auto-evolve tests");
        genome.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Rule {
                template: serde_json::json!({"success": success}),
            },
        });
        genome
    }

    struct StaticSemanticEvaluationDriver;

    #[async_trait::async_trait]
    impl crate::driver::EvolutionDriver for StaticSemanticEvaluationDriver {
        fn has_llm_backend(&self) -> bool {
            true
        }

        async fn execute(
            &self,
            _prompt: &str,
            _model: &str,
            _system: Option<&str>,
        ) -> Result<String, String> {
            Ok(
                r#"[{"capability":"semantic-cap","innovation_score":1.5,"utility_score":0.9}]"#
                    .into(),
            )
        }

        async fn execute_conversation(
            &self,
            _initial_prompt: &str,
            _model: &str,
            _system: Option<&str>,
            _follow_ups: &[&str],
        ) -> Result<String, String> {
            self.execute("", "", None).await
        }
    }

    /// LLM 可以提出探索先验，但不能靠给自己高分直接改变选择适应度。
    #[tokio::test]
    async fn llm_semantic_evaluation_does_not_self_promote_fitness() {
        let tmp = std::env::temp_dir().join(format!("semantic_eval_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        let mut genome = make_rule_genome("semantic-cap", true);
        genome.fitness.record_real_call(true, 10.0);
        let observed_score = genome.fitness.score;
        evolution.register_genome(genome).unwrap();

        let llm = std::sync::Arc::new(StaticSemanticEvaluationDriver);
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());
        evolver.llm_evaluate_capabilities(&mut evolution).await;

        let fitness = &evolution.genomes()["semantic-cap"].fitness;
        assert_eq!(fitness.innovation_score, 1.0, "语义分数应限制在 0..=1");
        assert_eq!(fitness.utility_score, 0.9);
        assert_eq!(
            fitness.score, observed_score,
            "LLM 自评分不得直接改变观测适应度"
        );
        assert!(fitness.llm_evaluated_at > 0);
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn attribution_snapshot_is_invalidated_by_human_feedback() {
        let tmp = std::env::temp_dir().join(format!("attribution_stale_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        let mut genome = make_rule_genome("stale-cap", false);
        genome.fitness.record_real_call(false, 10.0);
        genome.fitness.record_real_call(false, 10.0);
        genome.fitness.record_real_call(true, 10.0);
        evolution.register_genome(genome).unwrap();

        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());
        let original_report = evolver.introspect(&evolution);
        let weak = original_report.weak_capabilities[0].clone();
        let snapshot =
            AutoEvolver::snapshot_for_attribution(&evolution, std::slice::from_ref(&weak));
        assert!(AutoEvolver::attribution_snapshot_is_current(
            &evolution,
            &original_report,
            &snapshot,
            &weak,
        ));

        evolution
            .genomes_mut()
            .get_mut("stale-cap")
            .unwrap()
            .fitness
            .record_human_signal(true);
        let current_report = evolver.introspect(&evolution);
        assert!(
            !AutoEvolver::attribution_snapshot_is_current(
                &evolution,
                &current_report,
                &snapshot,
                &weak,
            ),
            "锁外归因期间到达的人类反馈必须使旧归因失效"
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// A/B 两侧都失败时没有任何正向证据，子代必须回滚。
    #[tokio::test]
    async fn ab_compare_rejects_child_when_both_versions_fail() {
        let tmp = std::env::temp_dir().join(format!("ab_both_fail_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        evolution.register_genome(make_rule_genome("ab-parent", false));
        evolution.register_genome(make_rule_genome("ab-child", false));

        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());

        let promote = evolver
            .ab_compare(&evolution, "ab-parent", "ab-child", &serde_json::json!({}))
            .await;

        assert!(!promote, "父代和子代都失败时不得晋升子代");
        let _ = std::fs::remove_dir_all(tmp);
    }

    struct DelayedAbCapability {
        name: String,
        delay_ms: u64,
    }

    struct MalformedFitnessCapability;

    #[async_trait::async_trait]
    impl crate::capability::Capability for MalformedFitnessCapability {
        fn name(&self) -> &str {
            "bad-fitness-cap"
        }

        async fn handle(&self, msg: &crate::message::Message) -> crate::message::MessageResult {
            Ok(crate::message::Message::builder()
                .from(self.name())
                .to(msg.from.as_deref().unwrap_or("test"))
                .action(&msg.action)
                .payload(serde_json::json!({"not_fitness": true}))
                .build())
        }
    }

    #[async_trait::async_trait]
    impl crate::capability::Capability for DelayedAbCapability {
        fn name(&self) -> &str {
            &self.name
        }

        fn actions(&self) -> Vec<&str> {
            vec!["run"]
        }

        async fn handle(&self, msg: &crate::message::Message) -> crate::message::MessageResult {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            Ok(crate::message::Message::builder()
                .from(&self.name)
                .to(msg.from.as_deref().unwrap_or("test"))
                .action(&msg.action)
                .payload(serde_json::json!({"success": true}))
                .build())
        }
    }

    /// A/B 延迟必须由运行时实测，不能依赖能力自行返回一个不存在的 `_elapsed_ms`。
    #[tokio::test]
    async fn ab_compare_rejects_measurably_slower_child() {
        let tmp = std::env::temp_dir().join(format!("ab_latency_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        evolution.register_genome(make_rule_genome("fast-parent", true));
        evolution.register_genome(make_rule_genome("slow-child", true));

        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        bus.register(std::sync::Arc::new(DelayedAbCapability {
            name: "fast-parent".into(),
            delay_ms: 10,
        }))
        .await;
        bus.register(std::sync::Arc::new(DelayedAbCapability {
            name: "slow-child".into(),
            delay_ms: 80,
        }))
        .await;

        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());
        let promote = evolver
            .ab_compare(
                &evolution,
                "fast-parent",
                "slow-child",
                &serde_json::json!({}),
            )
            .await;

        assert!(!promote, "实测显著更慢的子代不得晋升");
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// sync_fitness 应同步调用统计，但不能用注册时的旧快照覆盖后来写入 genome 的强证据。
    #[tokio::test]
    async fn sync_fitness_preserves_genome_evidence_while_merging_runtime_stats() {
        use crate::validator::SignalStrength;

        let tmp = std::env::temp_dir().join(format!("sync_fitness_merge_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        let mut genome = make_rule_genome("fitness-merge", true);
        // 两次历史调用均为自测试；随后运行时再发生一次真实调用。
        genome.fitness.call_count = 2;
        genome.fitness.auto_test_count = 2;
        genome.fitness.success_count = 2;
        genome.fitness.non_empty_output_count = 2;
        genome.fitness.avg_latency_ms = 100.0;
        genome.fitness.recompute_score();
        evolution.register_genome(genome).unwrap();

        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());

        // runtime_fitness 在注册时取得快照；之后 genome 侧新增的证据不能被该快照覆盖。
        let runtime_genome = evolution.genomes().get("fitness-merge").unwrap().clone();
        evolver
            .bus
            .register(std::sync::Arc::new(
                evolver.build_capability(runtime_genome),
            ))
            .await;

        let call = crate::message::Message::builder()
            .from("test")
            .to("fitness-merge")
            .action("run")
            .payload(serde_json::json!({}))
            .build();
        evolver.bus.send(call).await.unwrap();

        {
            let persisted = &mut evolution
                .genomes_mut()
                .get_mut("fitness-merge")
                .unwrap()
                .fitness;
            persisted.real_validation_passes = 3;
            persisted.real_validation_failures = 2;
            persisted.record_human_signal(true);
            persisted.record_human_signal(false);
            persisted.innovation_score = 0.84;
            persisted.utility_score = 0.91;
            persisted.llm_evaluated_at = 42;
            persisted.dependency_complexity = 0.67;
            persisted.last_evaluated = Some("9999999999".into());
            assert_eq!(persisted.strongest_signal, SignalStrength::HumanValue);
        }

        evolver.sync_fitness(&mut evolution).await.unwrap();

        let fitness = &evolution.genomes().get("fitness-merge").unwrap().fitness;
        // 运行时统计已同步：2 次历史自测 + 1 次新真实调用。
        assert_eq!(fitness.call_count, 3);
        assert_eq!(fitness.success_count, 3);
        assert_eq!(fitness.failure_count, 0);
        assert_eq!(fitness.auto_test_count, 2);
        assert_eq!(fitness.real_call_count(), 1);
        assert_eq!(fitness.rounds_dormant, 0);

        // genome 侧的高可信选择证据和价值评估完整保留。
        assert_eq!(fitness.real_validation_passes, 3);
        assert_eq!(fitness.real_validation_failures, 2);
        assert_eq!(fitness.strongest_signal, SignalStrength::HumanValue);
        assert_eq!(fitness.human_signals_count, 2);
        assert!((fitness.human_score - 0.5).abs() < f64::EPSILON);
        assert!((fitness.innovation_score - 0.84).abs() < f64::EPSILON);
        assert!((fitness.utility_score - 0.91).abs() < f64::EPSILON);
        assert_eq!(fitness.llm_evaluated_at, 42);
        assert!((fitness.dependency_complexity - 0.67).abs() < f64::EPSILON);
        assert_eq!(fitness.last_evaluated.as_deref(), Some("9999999999"));

        let _ = std::fs::remove_dir_all(tmp);
    }

    #[tokio::test]
    async fn auto_evaluation_does_not_become_real_usage_or_double_dormancy() {
        let tmp = std::env::temp_dir().join(format!("fitness_class_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        evolution.register_genome(make_rule_genome("evaluation-cap", true));

        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());
        let runtime_genome = evolution.genomes()["evaluation-cap"].clone();
        evolver
            .bus
            .register(std::sync::Arc::new(
                evolver.build_capability(runtime_genome),
            ))
            .await;

        let probe = crate::message::Message::builder()
            .from("auto_evolver")
            .to("evaluation-cap")
            .action("run")
            .payload(serde_json::json!({}))
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();
        evolver.bus.send(probe).await.unwrap();

        evolver.sync_fitness(&mut evolution).await.unwrap();
        let after_round = &evolution.genomes()["evaluation-cap"].fitness;
        assert_eq!(after_round.call_count, 1);
        assert_eq!(after_round.auto_test_count, 1);
        assert_eq!(after_round.real_call_count(), 0);
        assert_eq!(after_round.rounds_dormant, 1);

        // 锁外归因写回前的刷新只合并新 runtime 状态，不代表新的一轮。
        evolver
            .refresh_runtime_fitness(&mut evolution)
            .await
            .unwrap();
        assert_eq!(
            evolution.genomes()["evaluation-cap"].fitness.rounds_dormant,
            1
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[tokio::test]
    async fn sync_fitness_is_all_or_nothing_when_a_snapshot_is_invalid() {
        let tmp = std::env::temp_dir().join(format!("fitness_atomic_{}", uuid_str()));
        let mut evolution = crate::evolution::EvolutionEngine::new(&tmp);
        evolution.register_genome(make_rule_genome("good-fitness-cap", true));
        evolution.register_genome(make_rule_genome("bad-fitness-cap", true));

        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());
        let good = evolution.genomes()["good-fitness-cap"].clone();
        evolver
            .bus
            .register(std::sync::Arc::new(evolver.build_capability(good)))
            .await;
        evolver
            .bus
            .register(std::sync::Arc::new(MalformedFitnessCapability))
            .await;

        let before =
            serde_json::to_value(&evolution.genomes()["good-fitness-cap"].fitness).unwrap();
        let error = evolver
            .sync_fitness(&mut evolution)
            .await
            .expect_err("损坏快照应使整批同步失败");
        assert!(error.contains("缺少 fitness 字段"));
        let after = serde_json::to_value(&evolution.genomes()["good-fitness-cap"].fitness).unwrap();
        assert_eq!(before, after, "同步失败不得留下前半批次修改");
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// 补丁:find 唯一匹配 → 替换成功,应用后整文件语法过
    #[tokio::test]
    async fn fix_script_patch_applies_single_match() {
        let (evolver, mut evo) = make_patch_harness();
        let plan = MutationPlan::FixScriptPatch {
            capability: "py_cap".into(),
            action: "run".into(),
            find: "x = p['x']".into(),
            replace: "x = p.get('x', 0)".into(),
            expected_improvement: "用 .get 防 KeyError".into(),
        };
        let result = evolver.apply_mutation(&mut evo, &plan).await;
        assert!(result.is_ok(), "补丁应成功: {:?}", result.err());
        // 验证替换真的生效
        let new_name = result.unwrap();
        let g = evo.genomes().get(&new_name).unwrap();
        if let crate::genome::ActionImpl::Script { code, .. } = &g.actions[0].implementation {
            assert!(code.contains("p.get('x', 0)"), "新代码应含替换内容");
            assert!(!code.contains("p['x']"), "旧 find 片段应被替换掉");
        } else {
            panic!("应为 Script 实现");
        }
    }

    /// 补丁:find 不存在 → Err(锚点未找到),不猜
    #[tokio::test]
    async fn fix_script_patch_rejects_no_match() {
        let (evolver, mut evo) = make_patch_harness();
        let plan = MutationPlan::FixScriptPatch {
            capability: "py_cap".into(),
            action: "run".into(),
            find: "这行代码不存在".into(),
            replace: "whatever".into(),
            expected_improvement: "x".into(),
        };
        let result = evolver.apply_mutation(&mut evo, &plan).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("锚点未找到"),
            "错误信息应说明锚点未找到: {}",
            err
        );
    }

    /// 补丁:find 匹配多处 → Err(需更具体上下文),不猜
    #[tokio::test]
    async fn fix_script_patch_rejects_ambiguous() {
        use crate::genome::{ActionGene, ActionImpl, CapabilityGenome};
        let tmp = std::env::temp_dir().join(format!("patch_amb_{}", uuid_str()));
        let mut evo = crate::evolution::EvolutionEngine::new(&tmp);
        let mut g = CapabilityGenome::new("py_cap2", "test");
        // 含两个 "import json" 的代码,find 匹配多处
        g.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Script {
                code: "import json\nimport json\n\ndef run():\n    pass\n".into(),
                language: "python".into(),
                timeout_secs: 30,
            },
        });
        evo.register_genome(g).unwrap();
        let llm = std::sync::Arc::new(crate::genome::LlmExecutor::new("fake", "fake"));
        let bus = std::sync::Arc::new(crate::message_bus::MessageBus::new());
        let evolver = AutoEvolver::new(llm, bus, crate::platform::Platform::detect());

        let plan = MutationPlan::FixScriptPatch {
            capability: "py_cap2".into(),
            action: "run".into(),
            find: "import json".into(),
            replace: "import json\nimport sys".into(),
            expected_improvement: "x".into(),
        };
        let result = evolver.apply_mutation(&mut evo, &plan).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("匹配") && err.contains("处"),
            "应提示匹配多处: {}",
            err
        );
    }

    /// 补丁:应用后的完整文件做语法预检 —— 应用后语法错则 Err
    #[tokio::test]
    async fn fix_script_patch_validates_full_file() {
        let (evolver, mut evo) = make_patch_harness();
        // replace 是语法错的片段,会让应用后的文件语法不合法
        let plan = MutationPlan::FixScriptPatch {
            capability: "py_cap".into(),
            action: "run".into(),
            find: "x = p['x']".into(),
            replace: "x = p['x'\n  bad syntax here (((".into(),
            expected_improvement: "x".into(),
        };
        let result = evolver.apply_mutation(&mut evo, &plan).await;
        assert!(result.is_err(), "应用后语法错应被预检拦截");
        let err = result.unwrap_err();
        assert!(err.contains("语法错误"), "应是语法错误: {}", err);
    }

    /// 验证 MutationPlan::FixComposite 反序列化
    #[test]
    fn test_mutation_plan_fix_composite_deserialize() {
        let json = r#"{
            "mutation_type": "fix_composite",
            "capability": "pipeline",
            "action": "run",
            "new_steps": [{"name": "step1", "capability": "fs", "action": "read"}],
            "expected_improvement": "优化流程"
        }"#;
        let plan: MutationPlan = serde_json::from_str(json).unwrap();
        match plan {
            MutationPlan::FixComposite {
                capability,
                action,
                new_steps,
                ..
            } => {
                assert_eq!(capability, "pipeline");
                assert_eq!(action, "run");
                assert!(new_steps.is_array());
            }
            _ => panic!("应该是 FixComposite 变体"),
        }
    }

    /// 验证 MutationPlan::FixPrompt 反序列化
    #[test]
    fn test_mutation_plan_fix_prompt_deserialize() {
        let json = r#"{
            "mutation_type": "fix_prompt",
            "capability": "analyzer",
            "action": "analyze",
            "new_prompt": "请分析以下内容",
            "expected_improvement": "改进提示"
        }"#;
        let plan: MutationPlan = serde_json::from_str(json).unwrap();
        match plan {
            MutationPlan::FixPrompt { new_prompt, .. } => {
                assert_eq!(new_prompt, "请分析以下内容");
            }
            _ => panic!("应该是 FixPrompt 变体"),
        }
    }

    /// 验证无效的 mutation_type 反序列化失败（fail fast）
    #[test]
    fn test_mutation_plan_invalid_type_fails() {
        let json = r#"{
            "mutation_type": "add_error_handling",
            "capability": "x",
            "action": "y",
            "new_code": "z",
            "expected_improvement": "w"
        }"#;
        let result: Result<MutationPlan, _> = serde_json::from_str(json);
        assert!(result.is_err(), "无效的 mutation_type 应该反序列化失败");
    }

    /// 验证 FixScript 缺少 new_code 字段时反序列化失败（fail fast）
    #[test]
    fn test_mutation_plan_missing_field_fails() {
        let json = r#"{
            "mutation_type": "fix_script",
            "capability": "x",
            "action": "y",
            "expected_improvement": "z"
        }"#;
        let result: Result<MutationPlan, _> = serde_json::from_str(json);
        assert!(result.is_err(), "缺少 new_code 字段应该反序列化失败");
    }

    /// 验证 MutationPlan 的访问器方法
    #[test]
    fn test_mutation_plan_accessors() {
        let plan = MutationPlan::FixScript {
            capability: "cap".into(),
            action: "act".into(),
            new_code: "code".into(),
            expected_improvement: "improvement".into(),
        };
        assert_eq!(plan.capability(), "cap");
        assert_eq!(plan.action(), "act");
        assert_eq!(plan.expected_improvement(), "improvement");
        assert_eq!(plan.mutation_type_str(), "fix_script");
    }

    /// 验证 extract_json 处理 ```json 包裹
    #[test]
    fn test_extract_json_markdown_block() {
        let input = "一些思考\n```json\n{\"key\": \"value\"}\n```\n更多内容";
        let extracted = extract_json(input);
        assert_eq!(extracted, r#"{"key": "value"}"#);
    }

    /// 验证 extract_json 处理普通 ``` 包裹
    #[test]
    fn test_extract_json_plain_code_block() {
        let input = "```\n{\"key\": \"value\"}\n```";
        let extracted = extract_json(input);
        assert_eq!(extracted, r#"{"key": "value"}"#);
    }

    /// 验证 extract_json 无包裹时直接返回
    #[test]
    fn test_extract_json_no_block() {
        let input = r#"{"key": "value"}"#;
        let extracted = extract_json(input);
        assert_eq!(extracted, r#"{"key": "value"}"#);
    }

    /// 验证 safe_truncate 在多字节字符边界正确截断
    #[test]
    fn test_safe_truncate_multibyte() {
        let s = "你好世界abc";
        let truncated = safe_truncate(s, 5);
        // "你好" 是 6 字节，5 字节会回退到 3 字节（"你"）
        assert_eq!(truncated, "你");
    }

    /// 验证 safe_truncate 短字符串不截断
    #[test]
    fn test_safe_truncate_short() {
        let s = "abc";
        assert_eq!(safe_truncate(s, 100), "abc");
    }
}
