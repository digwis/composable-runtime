use crate::evolution::EvolutionEngine;
use crate::genome::{CapabilityGenome, ActionImpl, LlmExecutor, ScriptedCapability};
use crate::message_bus::MessageBus;
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// 自主进化引擎 — 不依赖用户任务的后台进化循环
///
/// 实现"有认知的自创生"的阶段 3：
/// 1. 自省：分析能力图谱，发现弱能力和缺口
/// 2. 归因：LLM 分析失败原因
/// 3. 变异：自动生成改进方案
/// 4. 测试：自动验证变异效果
/// 5. 选择：保留更优的变异体
pub struct AutoEvolver {
    /// LLM 执行器（用于归因分析和变异方案生成）
    llm: Arc<LlmExecutor>,
    /// 消息总线（用于测试能力）
    bus: Arc<MessageBus>,
    /// 平台信息
    platform: Platform,
    /// 进化统计
    stats: AutoEvolveStats,
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
    /// 自动淘汰次数
    pub eliminations: u32,
}

/// 自省报告 — 系统自我分析的产物
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IntrospectionReport {
    /// 弱能力（成功率 < 0.8 且调用次数 > 2）
    weak_capabilities: Vec<WeakCapability>,
    /// 未使用能力（调用次数为 0 且存在时间较长）
    dormant_capabilities: Vec<String>,
    /// 能力总数
    total_capabilities: usize,
    /// 平均适应度
    avg_fitness: f64,
    /// 能力图谱密度（能力间引用关系数 / 能力数²）
    graph_density: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeakCapability {
    name: String,
    success_rate: f64,
    call_count: u32,
    failure_count: u32,
    avg_latency_ms: f64,
    actions: Vec<String>,
}

/// 归因结果 — LLM 分析失败原因后的产物
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttributionResult {
    /// 失败原因分析
    analysis: String,
    /// 建议的变异方案
    mutation_plan: MutationPlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MutationPlan {
    /// 目标能力名
    capability: String,
    /// 目标动作名
    action: String,
    /// 变异类型: "fix_script" | "fix_composite" | "add_error_handling" | "optimize"
    mutation_type: String,
    /// 新的实现代码（如果是 Script 类型）
    new_code: Option<String>,
    /// 新的步骤定义（如果是 Composite 类型）
    new_steps: Option<serde_json::Value>,
    /// 预期改进
    expected_improvement: String,
}

impl AutoEvolver {
    pub fn new(
        llm: Arc<LlmExecutor>,
        bus: Arc<MessageBus>,
        platform: Platform,
    ) -> Self {
        Self {
            llm,
            bus,
            platform,
            stats: AutoEvolveStats::default(),
        }
    }

    /// 运行一轮自主进化循环
    ///
    /// 自省 → 归因 → 变异 → 测试 → 选择
    pub async fn evolve_once(
        &mut self,
        evolution: &mut EvolutionEngine,
    ) -> Result<Vec<String>, String> {
        let mut actions = Vec::new();

        // 0. 同步运行时适应度到进化引擎
        self.sync_fitness(evolution).await;

        // 1. 自省：分析能力图谱
        let report = self.introspect(evolution);
        self.stats.introspections += 1;

        if report.total_capabilities == 0 {
            return Ok(vec!["无能力可分析".into()]);
        }

        println!("  🔍 自省: {} 个能力, 平均适应度 {:.2}, {} 个弱能力, {} 个休眠能力",
            report.total_capabilities,
            report.avg_fitness,
            report.weak_capabilities.len(),
            report.dormant_capabilities.len(),
        );

        // 2. 归因 + 变异：对每个弱能力分析原因并尝试改进
        for weak in &report.weak_capabilities {
            let attribution = self.attribute_failure(evolution, weak).await;
            if let Some(attr) = attribution {
                self.stats.attributions += 1;
                println!("  🧠 归因: {} → {}", weak.name, attr.analysis);

                // 3. 变异：应用改进方案
                let mutation_result = self.apply_mutation(evolution, &attr.mutation_plan).await;
                self.stats.mutations += 1;

                match mutation_result {
                    Ok(new_name) => {
                        // 4. 测试：验证变异效果
                        let test_result = self.test_capability(evolution, &new_name).await;

                        if test_result {
                            self.stats.mutation_successes += 1;
                            println!("  ✅ 变异成功: {} → {} (测试通过)", weak.name, new_name);
                            actions.push(format!("变异 {} → {} (成功)", weak.name, new_name));
                        } else {
                            println!("  ❌ 变异测试失败: {} → {}", weak.name, new_name);
                            actions.push(format!("变异 {} → {} (测试失败)", weak.name, new_name));
                        }
                    }
                    Err(e) => {
                        println!("  ❌ 变异应用失败: {}", e);
                    }
                }
            }
        }

        // 5. 选择：淘汰长期休眠且适应度低的能力
        for name in &report.dormant_capabilities {
            if let Some(g) = evolution.genomes().get(name) {
                if g.fitness.score < 0.1 {
                    println!("  🗑️  自动淘汰休眠能力: {} (适应度 {:.2})", name, g.fitness.score);
                    self.stats.eliminations += 1;
                    actions.push(format!("淘汰 {}", name));
                }
            }
        }

        // 6. 环境感知：检测环境变化，发现能力缺口
        let gaps = self.detect_capability_gaps(evolution).await;
        for gap in &gaps {
            self.stats.gaps_found += 1;
            println!("  💡 发现能力缺口: {}", gap);
            actions.push(format!("发现缺口: {}", gap));
        }

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

        while round < max_rounds {
            round += 1;
            println!("🧬 ── 第 {} 轮 ──", round);

            let actions = self.evolve_once(evolution).await?;

            if actions.is_empty() || actions.iter().all(|a| a.starts_with("无")) {
                idle_count += 1;
                println!("  💤 无进化动作 (连续空闲 {} / {})", idle_count, idle_threshold);
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
            println!("  📊 能力数: {} | 平均适应度: {:.2}",
                genomes.len(),
                genomes.values().map(|g| g.fitness.score).sum::<f64>() / genomes.len().max(1) as f64,
            );

            // 等待下一轮
            if round < max_rounds && idle_count < idle_threshold {
                tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)).await;
            }

            println!();
        }

        println!("🧬 ═══ 持续进化结束 (共 {} 轮) ═══\n", round);
        println!("{}", self.report());

        Ok(self.stats.clone())
    }

    /// 有目标进化 — 定向进化模式
    ///
    /// 用户给出进化目标，系统朝目标方向进化，直到：
    /// - 目标达成（LLM 判断目标已满足）
    /// - 达到最大轮数
    /// - 收到终止信号
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

        while round < max_rounds {
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
                println!("\n🧬 ✅ 目标达成！定向进化结束 (共 {} 轮)\n", round);
                println!("{}", self.report());
                return Ok(self.stats.clone());
            }

            // 如果目标未达成，让 LLM 生成朝目标方向的新能力
            println!("  🧠 思考朝目标方向的进化策略...");
            let created = self.evolve_towards_goal(evolution, goal, &assessment).await;
            if let Some(name) = created {
                println!("  🧬 为目标创造新能力: {}", name);
            }

            // 等待下一轮
            if round < max_rounds {
                tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)).await;
            }

            println!();
        }

        println!("🧬 ═══ 达到最大轮数，定向进化结束 (共 {} 轮) ═══\n", round);
        println!("{}", self.report());

        Ok(self.stats.clone())
    }

    /// 评估目标达成度
    async fn evaluate_goal(
        &self,
        evolution: &EvolutionEngine,
        goal: &str,
    ) -> (bool, String) {
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        let genomes_summary: Vec<String> = genomes.iter()
            .map(|g| format!("{} (适应度:{:.2}, 动作:{})", g.name, g.fitness.score, g.action_names().join(",")))
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
}}"#, genomes_summary.join("\n"));

        let result = self.llm.execute(&prompt, "claude-sonnet-4-6", None).await;
        match result {
            Ok(text) => {
                let json_str = extract_json(&text);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    let achieved = v.get("achieved").and_then(|v| v.as_bool()).unwrap_or(false);
                    let assessment = v.get("assessment").and_then(|v| v.as_str()).unwrap_or("未知").to_string();
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
        let genomes_summary: Vec<String> = genomes.iter()
            .map(|g| format!("{}: {} ({})", g.name, g.description, g.action_names().join(",")))
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
}}"#, genomes_summary.join("\n"), self.platform.os, self.platform.arch);

        let result = self.llm.execute(&prompt, "claude-sonnet-4-6", None).await.ok()?;
        let json_str = extract_json(&result);
        let genome: CapabilityGenome = serde_json::from_str(&json_str).ok()?;

        let name = genome.name.clone();
        evolution.register_genome(genome);

        // 注册到总线并测试
        let genome = evolution.genomes().get(&name)?.clone();
        let cap = ScriptedCapability::from_genome(genome)
            .with_llm(self.llm.clone())
            .with_bus(self.bus.clone());
        self.bus.register(Arc::new(cap)).await;

        Some(name)
    }

    /// 自省：分析能力图谱
    fn introspect(&self, evolution: &EvolutionEngine) -> IntrospectionReport {
        let genomes = evolution.genomes();
        let total = genomes.len();

        let mut weak = Vec::new();
        let mut dormant = Vec::new();
        let mut total_score = 0.0;
        let mut scored_count = 0;

        for (name, genome) in genomes {
            if genome.fitness.call_count > 2 {
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
            } else if genome.fitness.call_count == 0 {
                dormant.push(name.clone());
            }
        }

        // 按成功率排序，最差的优先处理
        weak.sort_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap_or(std::cmp::Ordering::Equal));

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

        IntrospectionReport {
            weak_capabilities: weak,
            dormant_capabilities: dormant,
            total_capabilities: total,
            avg_fitness,
            graph_density,
        }
    }

    /// 归因：用 LLM 分析弱能力为什么失败，并生成变异方案
    async fn attribute_failure(
        &self,
        evolution: &EvolutionEngine,
        weak: &WeakCapability,
    ) -> Option<AttributionResult> {
        let genome = evolution.genomes().get(&weak.name)?;
        let genome_json = serde_json::to_string_pretty(genome).ok()?;

        let prompt = format!(
r#"你是一个能力进化分析器。以下能力表现不佳，请分析原因并给出变异方案。

能力基因组:
{genome_json}

表现数据:
- 成功率: {:.0}%
- 调用次数: {}
- 失败次数: {}
- 平均延迟: {:.0}ms

请分析失败原因，并给出具体的变异方案。
返回严格 JSON:
{{
  "analysis": "失败原因分析",
  "mutation_plan": {{
    "capability": "能力名",
    "action": "动作名",
    "mutation_type": "fix_script | fix_composite | add_error_handling | optimize",
    "new_code": "如果是 Script 类型，给出改进后的完整代码（可选）",
    "new_steps": null,
    "expected_improvement": "预期改进效果"
  }}
}}"#, 
            weak.success_rate * 100.0,
            weak.call_count,
            weak.failure_count,
            weak.avg_latency_ms,
        );

        let result = match self.llm.execute(&prompt, "claude-sonnet-4-6", None).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("归因 LLM 调用失败: {}", e);
                return None;
            }
        };
        let json_str = extract_json(&result);
        let parsed: AttributionResult = match serde_json::from_str(&json_str) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("归因 JSON 解析失败: {} | 原始: {}", e, &result[..result.len().min(200)]);
                return None;
            }
        };
        Some(parsed)
    }

    /// 应用变异：根据变异方案修改基因组
    async fn apply_mutation(
        &self,
        evolution: &mut EvolutionEngine,
        plan: &MutationPlan,
    ) -> Result<String, String> {
        let genome = evolution.genomes().get(&plan.capability)
            .ok_or_else(|| format!("能力 '{}' 不存在", plan.capability))?
            .clone();

        let mut new_genome = genome.clone();
        new_genome.lineage.parent = Some(plan.capability.clone());
        new_genome.lineage.origin = crate::genome::Origin::Mutated;
        new_genome.lineage.generation += 1;
        new_genome.record_mutation(&plan.mutation_type, &plan.expected_improvement);

        // 找到目标动作并修改
        let action = new_genome.actions.iter_mut()
            .find(|a| a.name == plan.action)
            .ok_or_else(|| format!("动作 '{}' 不存在", plan.action))?;

        match &mut action.implementation {
            ActionImpl::Script { code, .. } => {
                if let Some(new_code) = &plan.new_code {
                    *code = new_code.clone();
                }
            }
            ActionImpl::Composite { steps } => {
                if let Some(new_steps) = &plan.new_steps {
                    if let Ok(parsed_steps) = serde_json::from_value::<Vec<crate::genome::CompositeStep>>(new_steps.clone()) {
                        *steps = parsed_steps;
                    }
                }
            }
            ActionImpl::Llm { prompt, .. } => {
                if let Some(new_code) = &plan.new_code {
                    *prompt = new_code.clone();
                }
            }
            _ => {
                return Err(format!("变异类型 {:?} 不适用于该动作的实现类型", plan.mutation_type));
            }
        }

        let new_name = format!("{}-v{}", plan.capability, new_genome.lineage.generation);
        new_genome.name = new_name.clone();

        // 重置适应度（新变异体从零开始评估）
        new_genome.fitness = crate::genome::FitnessGene::default();

        evolution.register_genome(new_genome);

        Ok(new_name)
    }

    /// 测试变异后的能力是否正常工作
    async fn test_capability(
        &self,
        evolution: &EvolutionEngine,
        capability_name: &str,
    ) -> bool {
        let genome = match evolution.genomes().get(capability_name) {
            Some(g) => g.clone(),
            None => return false,
        };

        if genome.actions.is_empty() {
            return false;
        }

        // 先 clone action 信息，因为 genome 会被 move
        let action_name = genome.actions[0].name.clone();
        let action_schema = genome.actions[0].input_schema.clone();
        let test_input = generate_test_input(&action_schema);

        let cap = ScriptedCapability::from_genome(genome)
            .with_llm(self.llm.clone())
            .with_bus(self.bus.clone());

        let msg = crate::message::Message::builder()
            .from("auto_evolver")
            .to(capability_name)
            .action(&action_name)
            .payload(test_input)
            .build();

        // 先注册到总线
        self.bus.register(Arc::new(cap)).await;

        let result = self.bus.send(msg).await;

        match result {
            Ok(resp) => {
                // 检查响应中是否有错误标志
                if let Some(success) = resp.payload.get("success") {
                    success.as_bool().unwrap_or(false)
                } else {
                    // 如果没有 success 字段，只要不报错就认为通过
                    true
                }
            }
            Err(_) => false,
        }
    }

    /// 检测能力缺口：根据环境分析还缺什么能力
    async fn detect_capability_gaps(
        &self,
        evolution: &EvolutionEngine,
    ) -> Vec<String> {
        let mut gaps = Vec::new();
        let existing: Vec<String> = evolution.genomes().keys().cloned().collect();

        // 检测环境中有哪些工具可用但还没有对应能力
        let env_tools: Vec<String> = self.platform.env.iter()
            .filter(|(k, v)| k.starts_with("has_") && v.as_str() == "true")
            .map(|(k, _)| k.strip_prefix("has_").unwrap_or(k).to_string())
            .collect();

        // 如果有 git 但没有 git 相关能力
        if env_tools.contains(&"git".to_string()) {
            let has_git_cap = existing.iter().any(|name| name.contains("git"));
            if !has_git_cap {
                gaps.push("git 操作能力 (有 git 工具但无对应能力)".into());
            }
        }

        // 如果有 docker 但没有 docker 相关能力
        if env_tools.contains(&"docker".to_string()) {
            let has_docker_cap = existing.iter().any(|name| name.contains("docker") || name.contains("container"));
            if !has_docker_cap {
                gaps.push("docker 操作能力 (有 docker 工具但无对应能力)".into());
            }
        }

        gaps
    }

    /// 获取进化统计
    pub fn stats(&self) -> &AutoEvolveStats {
        &self.stats
    }

    /// 同步运行时适应度到进化引擎
    async fn sync_fitness(&self, evolution: &mut EvolutionEngine) {
        // 通过总线获取能力列表
        let cap_names = self.bus.list_capabilities().await;

        // 对每个已注册能力，发送一个自省消息获取适应度
        // 使用特殊动作 __fitness__ 来获取运行时适应度
        for name in &cap_names {
            // 跳过原生能力（它们不是 ScriptedCapability）
            let native_caps = vec!["greet", "compute", "store", "fs", "shell", "http", "code", "web"];
            if native_caps.contains(&name.as_str()) {
                continue;
            }

            // 尝试获取适应度：发送 __fitness__ 动作
            let msg = crate::message::Message::builder()
                .from("auto_evolver")
                .to(name)
                .action("__fitness__")
                .payload(serde_json::json!({}))
                .build();

            if let Ok(resp) = self.bus.send(msg).await {
                if let Some(fitness_json) = resp.payload.get("fitness") {
                    if let Ok(fitness) = serde_json::from_value::<crate::genome::FitnessGene>(fitness_json.clone()) {
                        if let Some(genome) = evolution.genomes_mut().get_mut(name) {
                            genome.fitness = fitness;
                        }
                    }
                }
            }
        }
        evolution.save_fitness();
    }

    /// 生成自主进化报告
    pub fn report(&self) -> String {
        format!(
            "═══ 自主进化报告 ═══\n\
             自省次数: {}\n\
             归因次数: {}\n\
             自主变异: {} (成功 {})\n\
             发现缺口: {}\n\
             自动淘汰: {}\n",
            self.stats.introspections,
            self.stats.attributions,
            self.stats.mutations,
            self.stats.mutation_successes,
            self.stats.gaps_found,
            self.stats.eliminations,
        )
    }
}

/// 根据 input_schema 生成测试输入
fn generate_test_input(schema: &serde_json::Value) -> serde_json::Value {
    let mut test = serde_json::Map::new();

    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (key, schema) in props {
            let value = match schema.get("type").and_then(|t| t.as_str()) {
                Some("string") => {
                    // 根据描述生成合理的测试值
                    let desc = schema.get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    if desc.contains("path") || key.contains("path") {
                        serde_json::Value::String("/tmp/test_auto_evolve.txt".to_string())
                    } else if desc.contains("json") || key.contains("json") {
                        serde_json::Value::String(r#"{"test": true}"#.to_string())
                    } else {
                        serde_json::Value::String("test_value".to_string())
                    }
                }
                Some("integer") | Some("number") => serde_json::json!(42),
                Some("boolean") => serde_json::json!(true),
                Some("array") => serde_json::json!([]),
                _ => serde_json::Value::String("test".to_string()),
            };
            test.insert(key.clone(), value);
        }
    }

    serde_json::Value::Object(test)
}

/// 从 LLM 输出中提取 JSON（处理 markdown 包裹）
fn extract_json(text: &str) -> String {
    let trimmed = text.trim();

    // 尝试直接解析
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return trimmed.to_string();
    }

    // 尝试提取 ```json ... ``` 中的内容
    if let Some(start) = trimmed.find("```json") {
        if let Some(end) = trimmed.rfind("```") {
            let inner = &trimmed[start + 7..end].trim();
            return inner.to_string();
        }
    }

    // 尝试提取 ``` ... ``` 中的内容
    if let Some(start) = trimmed.find("```") {
        if let Some(end) = trimmed.rfind("```") {
            let inner = &trimmed[start + 3..end].trim();
            return inner.to_string();
        }
    }

    // 尝试找到第一个 { 和最后一个 }
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_string();
        }
    }

    trimmed.to_string()
}
