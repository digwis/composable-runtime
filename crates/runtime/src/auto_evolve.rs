use crate::evolution::EvolutionEngine;
use crate::genome::{ActionImpl, CapabilityGenome, LlmExecutor, ScriptedCapability};
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
    llm: Arc<LlmExecutor>,
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
    /// 自上次创造新能力以来的轮数（用于范式跃迁软触发）
    rounds_since_last_creation: u32,
    /// 能力连续变异失败次数：能力名 -> 失败次数
    /// 连续失败 3 次后跳过，避免卡环
    mutation_failures: std::collections::HashMap<String, u32>,
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
            MutationPlan::FixComposite { capability, .. } => capability,
            MutationPlan::FixPrompt { capability, .. } => capability,
            MutationPlan::FixCustomParams { capability, .. } => capability,
        }
    }

    /// 目标动作名
    fn action(&self) -> &str {
        match self {
            MutationPlan::FixScript { action, .. } => action,
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
            MutationPlan::FixComposite { .. } => "fix_composite",
            MutationPlan::FixPrompt { .. } => "fix_prompt",
            MutationPlan::FixCustomParams { .. } => "fix_custom_params",
        }
    }
}

/// 工具缺口重试间隔：N 轮后允许重新尝试填补被淘汰的缺口
const GAP_RETRY_ROUNDS: u32 = 20;

/// 范式跃迁触发阈值：连续 N 轮没有创造新能力时触发
const PARADIGM_SHIFT_IDLE_ROUNDS: u32 = 15;

impl AutoEvolver {
    pub fn new(llm: Arc<LlmExecutor>, bus: Arc<MessageBus>, platform: Platform) -> Self {
        Self {
            llm,
            bus,
            platform,
            executor_registry: None,
            stats: AutoEvolveStats::default(),
            round_count: 0,
            tried_gaps: std::collections::HashMap::new(),
            rounds_since_last_creation: 0,
            mutation_failures: std::collections::HashMap::new(),
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
            Err(format!("语法错误: {}", &err[..200.min(err.len())]))
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
        let mut actions = Vec::new();

        // 递增轮次计数
        self.round_count += 1;

        // 0. 同步运行时适应度到进化引擎
        self.sync_fitness(evolution).await?;

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
                                    evolution.remove_genome(ver);
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
            let weak_list: Vec<_> = report
                .weak_capabilities
                .iter()
                .take(3)
                .filter(|weak| {
                    let fail_count = *self.mutation_failures.get(&weak.name).unwrap_or(&0);
                    if fail_count >= 3 {
                        println!("  ⏭️  跳过 {} (连续变异失败 {} 次)", weak.name, fail_count);
                        false
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();

            // 2a. 并行归因
            let mut attr_futures = Vec::new();
            for weak in &weak_list {
                attr_futures.push(self.attribute_failure(evolution, weak));
            }
            let attr_results = futures::future::join_all(attr_futures).await;

            // 2b. 串行变异（快，只替换代码 + 语法预检）
            let mut mutation_results: Vec<(String, Result<String, String>)> = Vec::new();
            for (weak, attr) in weak_list.iter().zip(attr_results.into_iter()) {
                if let Some(attr) = attr {
                    self.stats.attributions += 1;
                    println!("  🧠 归因: {} → {}", weak.name, attr.analysis);
                    let result = self.apply_mutation(evolution, &attr.mutation_plan).await;
                    self.stats.mutations += 1;
                    mutation_results.push((weak.name.clone(), result));
                }
            }

            // 2c. 并行测试
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

            let mut test_futures = Vec::new();
            for (_, new_name) in &test_targets {
                test_futures.push(self.test_capability(evolution, new_name));
            }
            let test_results = futures::future::join_all(test_futures).await;

            // 2d. 处理测试结果
            for ((parent, new_name), (pass, test_input)) in
                test_targets.iter().zip(test_results.into_iter())
            {
                if pass {
                    // P4: 回归测试 — 用父代的持久化测试套件验证
                    let (regression_rate, regression_total) =
                        self.run_regression_tests(evolution, parent, new_name).await;
                    if regression_total > 0 && regression_rate < 0.5 {
                        // 回归测试通过率太低，淘汰
                        *self.mutation_failures.entry(parent.clone()).or_insert(0) += 1;
                        println!(
                            "  ❌ 回归测试失败: {} → {} ({:.0}% 通过)",
                            parent,
                            new_name,
                            regression_rate * 100.0
                        );
                        actions.push(format!("变异 {} → {} (回归测试失败)", parent, new_name));
                        evolution.remove_genome(new_name);
                        println!("  🗑️  淘汰回归失败变体: {}", new_name);
                        continue;
                    }

                    // P3-AB: AB 对比 — 与父代对比性能
                    let ab_promote = if let Some(ref input) = test_input {
                        self.ab_compare(evolution, parent, new_name, input).await
                    } else {
                        true // 无测试输入，跳过 AB 对比
                    };

                    if !ab_promote {
                        *self.mutation_failures.entry(parent.clone()).or_insert(0) += 1;
                        println!("  ❌ AB 对比失败: {} → {} (父代更优)", parent, new_name);
                        actions.push(format!("变异 {} → {} (AB 回滚)", parent, new_name));
                        evolution.remove_genome(new_name);
                        println!("  🗑️  淘汰 AB 回滚变体: {}", new_name);
                        continue;
                    }

                    self.stats.mutation_successes += 1;
                    self.mutation_failures.remove(parent);
                    println!(
                        "  ✅ 变异成功: {} → {} (测试+回归+AB 通过)",
                        parent, new_name
                    );
                    actions.push(format!("变异 {} → {} (成功)", parent, new_name));

                    // P0-2: 淘汰旧版本（仅变异体）
                    let is_mutated = evolution
                        .genomes()
                        .get(parent)
                        .map(|g| g.lineage.origin == crate::genome::Origin::Mutated)
                        .unwrap_or(false);
                    if is_mutated {
                        evolution.remove_genome(parent);
                        println!("  🗑️  淘汰旧版本: {}", parent);
                    }
                } else {
                    *self.mutation_failures.entry(parent.clone()).or_insert(0) += 1;
                    println!("  ❌ 变异测试失败: {} → {}", parent, new_name);
                    actions.push(format!("变异 {} → {} (测试失败)", parent, new_name));
                    evolution.remove_genome(new_name);
                    println!("  🗑️  淘汰失败变体: {}", new_name);
                }
                // P4: 保存测试用例
                if let Some(input) = test_input {
                    if let Some(g) = evolution.genomes_mut().get_mut(new_name) {
                        g.add_test_case(input, pass, "mutation_test");
                    }
                }
            }

            // 处理变异应用失败
            for (parent, result) in &mutation_results {
                if let Err(e) = result {
                    *self.mutation_failures.entry(parent.clone()).or_insert(0) += 1;
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
            // 确保能力不仅"能跑通"而且"真的有用"。
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
                        if result.success {
                            println!("  ✅ 真实验证通过: {} ({}ms)", name, result.elapsed_ms);
                            actions.push(format!("真实验证: {} (通过)", name));
                            if let Some(g) = evolution.genomes_mut().get_mut(name) {
                                g.fitness.record_real_call(true, result.elapsed_ms as f64);
                            }
                        } else {
                            println!(
                                "  ❌ 真实验证失败: {} — {}",
                                name,
                                &result.output[..100.min(result.output.len())]
                            );
                            actions.push(format!("真实验证: {} (失败)", name));
                            if let Some(g) = evolution.genomes_mut().get_mut(name) {
                                g.fitness.record_real_call(false, result.elapsed_ms as f64);
                            }
                        }
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
            // - 从未被真实调用（real_calls == 0）+ rounds_dormant >= NEW_CAP_THRESHOLD → 淘汰
            // - 有真实调用但成功率极低（score < 0.01）+ rounds_dormant >= FAILED_CAP_THRESHOLD → 淘汰
            const NEW_CAP_THRESHOLD: u32 = 20; // 新能力 20 轮宽限期
            const FAILED_CAP_THRESHOLD: u32 = 5; // 失败能力 5 轮宽限期
            let mut to_eliminate = Vec::new();
            for (name, g) in evolution.genomes() {
                let real_calls = g.fitness.real_call_count();
                let threshold = if real_calls == 0 {
                    NEW_CAP_THRESHOLD
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
            // 实际从进化引擎中移除
            for name in &to_eliminate {
                evolution.genomes_mut().remove(name);
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
                self.rounds_since_last_creation = 0;
                println!("  🔧 自动填补缺口: 创造能力 {}", created);
                actions.push(format!("填补缺口: {} → {}", gap, created));
            } else {
                actions.push(format!("发现缺口: {} (未能自动填补)", gap));
            }
        }

        // 7. 好奇心探索：如果本轮没有弱能力和缺口，主动探索新能力方向
        //    范式跃迁：连续 PARADIGM_SHIFT_IDLE_ROUNDS 轮无新能力创造时触发
        //    （软触发，而非每 10 轮强制，避免与收敛机制矛盾）
        self.rounds_since_last_creation += 1;
        let paradigm_shift = self.rounds_since_last_creation >= PARADIGM_SHIFT_IDLE_ROUNDS
            && report.total_capabilities > 0;
        let need_explore = (report.weak_capabilities.is_empty()
            && gaps_to_fill.is_empty()
            && self.rounds_since_last_creation >= 3)  // 至少 3 轮无创造才探索
            || paradigm_shift;
        if need_explore {
            self.stats.explorations += 1;
            if paradigm_shift {
                println!(
                    "  ⚡ 范式跃迁: 连续 {} 轮无新能力，强制跳出当前领域探索...",
                    self.rounds_since_last_creation
                );
            } else if report.total_capabilities == 0 {
                println!("  🔬 好奇心驱动探索: 能力库为空，从零开始创造...");
            } else {
                println!("  🔬 好奇心驱动探索: 系统健康，主动寻找新能力方向...");
            }
            if let Some(created) = self.explore_new_capability(evolution, paradigm_shift).await {
                self.stats.explored_created += 1;
                self.rounds_since_last_creation = 0;
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
                self.rounds_since_last_creation = 0;
                println!("  🧪 交叉重组: {}", created);
                actions.push(format!("交叉重组: {}", created));
            }
        }

        // P2-1: 9.5 组合能力：偶尔生成 Composite 类型能力（编排现有能力）
        if report.total_capabilities >= 5 && (self.stats.introspections % 5 == 0) {
            if let Some(created) = self.create_composite_capability(evolution).await {
                self.rounds_since_last_creation = 0;
                println!("  🔗 组合能力: {}", created);
                actions.push(format!("组合能力: {}", created));
            }
        }

        // 9. 持久化适应度
        evolution.save_fitness();

        Ok(actions)
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
        evolution.save_fitness();

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
                evolution.save_fitness();
                println!("\n🧬 ✅ 目标达成！定向进化结束 (共 {} 轮)\n", round);
                println!("{}", self.report());
                return Ok(self.stats.clone());
            }

            // 如果目标未达成，让 LLM 生成朝目标方向的新能力
            println!("  🧠 思考朝目标方向的进化策略...");
            let created = self.evolve_towards_goal(evolution, goal, &assessment).await;
            if let Some(name) = created {
                self.rounds_since_last_creation = 0;
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
        evolution.save_fitness();

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
        evolution.register_genome(genome);

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
            } else {
                // 无真实业务调用（含仅自测试的），列为休眠
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
                            format!("{}...（截断，共 {} 字符）", &code[..2000], code.len())
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
                            format!("{}...", &prompt[..500])
                        } else {
                            prompt.clone()
                        };
                        format!("Llm: {}", truncated)
                    }
                    ActionImpl::Rule { template } => format!("Rule: {}", template),
                    ActionImpl::Native { capability, action } => {
                        format!("Native: {} -> {}", capability, action)
                    }
                    ActionImpl::Custom {
                        executor_type,
                        params,
                    } => format!("Custom({}): {:?}", executor_type, params),
                };
                format!(
                    "  - action: {} | {} | impl: {}",
                    a.name, a.description, impl_summary
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

请分析失败原因，并给出具体的变异方案。
返回严格 JSON（mutation_type 决定携带的字段，不要携带无关字段）:
{{
  "analysis": "失败原因分析",
  "mutation_plan": {{
    "capability": "能力名",
    "action": "动作名",
    "mutation_type": "fix_script",
    "new_code": "改进后的完整代码",
    "expected_improvement": "预期改进效果"
  }}
}}

mutation_type 必须是以下之一，且只携带对应字段：
- fix_script: {{ mutation_type, capability, action, new_code, expected_improvement }}
- fix_composite: {{ mutation_type, capability, action, new_steps, expected_improvement }}
- fix_prompt: {{ mutation_type, capability, action, new_prompt, expected_improvement }}"#,
            genome.name,
            genome.description,
            action_summaries.join("\n"),
            weak.success_rate * 100.0,
            weak.call_count,
            weak.failure_count,
            weak.avg_latency_ms,
        );

        let result = match self.llm.execute(&prompt, "smart:attribution", None).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("归因 LLM 调用失败: {}", e);
                return None;
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
                    let check = tokio::process::Command::new("python3")
                        .arg("-c")
                        .arg("import ast; ast.parse(open('/dev/stdin').read())")
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::piped())
                        .spawn();
                    if let Ok(mut child) = check {
                        if let Some(mut stdin) = child.stdin.take() {
                            use tokio::io::AsyncWriteExt;
                            let _ = stdin.write_all(new_code.as_bytes()).await;
                        }
                        let output = child.wait_with_output().await;
                        if let Ok(out) = output {
                            if !out.status.success() {
                                let err = String::from_utf8_lossy(&out.stderr);
                                return Err(format!(
                                    "语法预检失败: {}",
                                    &err[..200.min(err.len())]
                                ));
                            }
                        }
                    }
                }
                *code = new_code.clone();
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

        evolution.register_genome(new_genome);

        Ok(new_name)
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
                .build();

            match self.bus.send(msg).await {
                Ok(resp) => {
                    let success = resp
                        .payload
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if success == tc.expect_success {
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
            .build();
        let child_result = self.bus.send(child_msg).await;
        let child_success = match &child_result {
            Ok(resp) => resp
                .payload
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        };
        let child_latency = match &child_result {
            Ok(resp) => resp
                .payload
                .get("_elapsed_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(100.0),
            Err(_) => 9999.0,
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
            .build();
        let parent_result = self.bus.send(parent_msg).await;
        let parent_success = match &parent_result {
            Ok(resp) => resp
                .payload
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        };
        let parent_latency = match &parent_result {
            Ok(resp) => resp
                .payload
                .get("_elapsed_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(100.0),
            Err(_) => 9999.0,
        };

        // AB 判定逻辑：
        // - 子代成功且父代失败 → 推广
        // - 子代失败且父代成功 → 回滚
        // - 都成功 → 比延迟，子代不比父代慢 2 倍就推广
        // - 都失败 → 推广（反正都失败，给新版本机会）
        let promote = match (child_success, parent_success) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => child_latency <= parent_latency * 2.0,
            (false, false) => true,
        };

        if !promote {
            println!(
                "  ⚖️  AB 对比: {} vs {} — 父代更优 (父:{}ms 子:{}ms), 回滚",
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
            .payload(real_input)
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
                let output = resp
                    .payload
                    .get("result")
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_default();
                Some(RealValidationResult {
                    capability: capability_name.to_string(),
                    action: action_name,
                    success,
                    elapsed_ms,
                    output,
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
            return Some(serde_json::json!({"command": "version"}));
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

        let result = match self.llm.execute(&prompt, "coder:gapfill", None).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("缺口填补 LLM 调用失败: {}", e);
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
        evolution.register_genome(genome);

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
        evolution.register_genome(genome);

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
            r#"交叉重组两个能力产生新能力。
父代1: {} — {} [{}]
父代2: {} — {} [{}]

返回精简 JSON 基因组（代码要短，description 一句话）:
{{
  "name": "新能力名",
  "version": "0.1.0",
  "description": "一句话描述",
  "actions": [{{"name": "动作名", "description": "描述", "input_schema": {{"properties": {{}}}}, "implementation": {{"type": "Script", "language": "python", "code": "简短Python代码", "timeout_secs": 30}}}}],
  "fitness": {{}},
  "lineage": {{}}
}}"#,
            parent1.name,
            parent1.description,
            parent1.action_names().join(","),
            parent2.name,
            parent2.description,
            parent2.action_names().join(",")
        );

        let result = self
            .llm
            .execute(&prompt, "coder:crossover", None)
            .await
            .ok()?;
        let json_str = extract_json(&result);
        let genome: CapabilityGenome = serde_json::from_str(json_str).ok()?;

        // P0-1: 语法预检
        if let Err(e) = Self::validate_genome_scripts(&genome).await {
            tracing::warn!("交叉重组语法预检失败: {}", e);
            return None;
        }

        let name = genome.name.clone();
        evolution.register_genome(genome);

        // 注册到总线
        let genome = evolution.genomes().get(&name)?.clone();
        let cap = self.build_capability(genome);
        self.bus.register(Arc::new(cap)).await;

        Some(name)
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

        let prompt = format!(
            r#"你是一个能力进化引擎。请分析以下现有能力，创造一个组合能力（Composite 类型）。

现有能力:
{}

组合能力通过编排现有能力的动作完成更复杂的任务。
例如：git_ops.status + code_quality_analyzer.analyze → 自动代码审查

返回严格 JSON（基因组格式，implementation.type 必须为 "Composite"）:
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
            candidates.join("\n")
        );

        let result = self
            .llm
            .execute(&prompt, "smart:composite", None)
            .await
            .ok()?;
        let json_str = extract_json(&result);
        let genome: CapabilityGenome = serde_json::from_str(json_str).ok()?;

        // 验证确实是 Composite 类型
        let is_composite = genome
            .actions
            .iter()
            .any(|a| matches!(a.implementation, ActionImpl::Composite { .. }));
        if !is_composite {
            tracing::warn!("LLM 返回的不是 Composite 类型能力");
            return None;
        }

        let name = genome.name.clone();
        evolution.register_genome(genome);

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

    /// 同步运行时适应度到进化引擎
    ///
    /// 关键修复：
    /// 1. 保留 genome 中的 auto_test_count（runtime_fitness 不含此字段）
    /// 2. 用 real_call_count 判断 dormant，而非 call_count
    ///    这样自测试过的能力如果没有真实业务调用，rounds_dormant 仍会递增
    /// 3. fail fast：用 ? 替代四层嵌套 if let，让不变量违反直接报错而非静默吞掉
    pub async fn sync_fitness(&self, evolution: &mut EvolutionEngine) -> Result<(), String> {
        // 通过总线获取能力列表
        let cap_names = self.bus.list_capabilities().await;

        // 记录哪些能力在总线上有注册
        let bus_caps: std::collections::HashSet<String> = cap_names.iter().cloned().collect();

        // 对每个已注册能力，发送一个自省消息获取适应度
        // 使用特殊动作 __fitness__ 来获取运行时适应度
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
            let mut fitness: crate::genome::FitnessGene =
                serde_json::from_value(fitness_json.clone()).map_err(|e| {
                    format!("sync_fitness: 能力 '{}' fitness 反序列化失败: {}", name, e)
                })?;
            let genome = evolution.genomes_mut().get_mut(name).ok_or_else(|| {
                format!(
                    "sync_fitness: 能力 '{}' 在总线上但不在进化引擎中（数据不一致）",
                    name
                )
            })?;

            // 保留 genome 中的 auto_test_count（runtime_fitness 不跟踪此字段）
            fitness.auto_test_count = genome.fitness.auto_test_count;

            // 用 real_call_count 判断 dormant
            // 这样自测试过的能力如果没有真实业务调用，仍会被标记为休眠
            let prev_dormant = genome.fitness.rounds_dormant;
            let real_calls = fitness.real_call_count();
            if real_calls > 0 {
                fitness.rounds_dormant = 0;
            } else {
                fitness.rounds_dormant = prev_dormant + 1;
            }
            genome.fitness = fitness;
        }

        // 对不在总线上但在进化引擎中的能力，也增加休眠计数
        for (_name, genome) in evolution.genomes_mut() {
            if !bus_caps.contains(_name) {
                // 能力未注册到总线，增加休眠计数
                genome.fitness.rounds_dormant += 1;
            }
        }

        evolution.save_fitness();
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
