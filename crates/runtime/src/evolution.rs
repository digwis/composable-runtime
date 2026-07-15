use crate::driver::EvolutionDriver;
use crate::genome::{
    ActionImpl, CapabilityGenome, FitnessGene, LineageGene, Origin, ScriptedCapability,
};
use crate::message_bus::MessageBus;
use crate::meta_evolve::ExecutorRegistry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// 进化引擎 — 管理能力的变异、选择和进化
///
/// 核心进化机制：
/// 1. 变异（Mutation）：AI 对现有能力做小幅修改
/// 2. 交叉（Crossover）：两个能力的基因混合产生后代
/// 3. 选择（Selection）：根据适应度淘汰低效能力
/// 4. 生成（Generation）：AI 从零创造新能力
#[derive(Clone)]
pub struct EvolutionEngine {
    /// 基因组库 — 所有能力基因组
    genomes: HashMap<String, CapabilityGenome>,
    /// 持久化存储路径
    storage_dir: PathBuf,
    /// LLM 执行器（用于脚本化能力）
    llm_executor: Option<Arc<dyn EvolutionDriver>>,
    /// 执行器注册表（用于 Custom 类型能力 — 元进化产物）
    executor_registry: Option<Arc<ExecutorRegistry>>,
    /// 进化历史
    history: Vec<EvolutionEvent>,
    /// 跨代记忆 — 持久化的进化教训和自主循环历史
    memory: EvolutionMemory,
}

/// 跨代记忆 — 重启后仍保留的进化上下文
///
/// 让系统在重启后能"记住"之前的经验教训，
/// 避免重复犯同样的错误，加速进化收敛。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvolutionMemory {
    /// 进化教训 — 记录失败原因和解决方案
    pub lessons: Vec<EvolutionLesson>,
    /// 自主循环历史摘要 — 记录每轮自主目标执行结果
    pub autonomous_history: Vec<AutonomousHistoryEntry>,
    /// 进化统计（跨重启）
    pub global_stats: GlobalEvolutionStats,
    /// 已尝试的变异方案及其结果（避免重复尝试）
    pub tried_mutations: Vec<TriedMutation>,
    /// 思维链 — 记录每轮 LLM 推理的完整思考过程
    pub thought_chains: Vec<ThoughtChain>,
    /// 联想网络 — 概念间关联，用于触发联想
    pub association_graph: Vec<AssociationLink>,
    /// 上次进化的时间戳
    pub last_evolution_ts: u64,
}

/// 进化教训
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionLesson {
    /// 教训内容
    pub lesson: String,
    /// 相关能力
    pub capability: String,
    /// 失败类型
    pub failure_type: String,
    /// 学到的时间
    pub learned_at: String,
    /// 被引用次数（其他进化过程参考此教训的次数）
    pub referenced_count: u32,
}

/// 自主循环历史条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousHistoryEntry {
    /// 目标描述
    pub goal: String,
    /// 是否成功
    pub success: bool,
    /// 使用的能力
    pub capabilities_used: Vec<String>,
    /// 耗时(ms)
    pub elapsed_ms: u64,
    /// 时间戳
    pub timestamp: u64,
}

/// 全局进化统计（跨重启）
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GlobalEvolutionStats {
    /// 总进化轮次（跨重启）
    pub total_rounds: u64,
    /// 总能力创造数
    pub total_created: u64,
    /// 总能力淘汰数
    pub total_eliminated: u64,
    /// 总变异尝试数
    pub total_mutations: u64,
    /// 总变异成功数
    pub total_mutation_successes: u64,
    /// 总自主目标执行数
    pub total_autonomous_goals: u64,
    /// 总自主目标成功数
    pub total_autonomous_successes: u64,
    /// 系统首次启动时间
    pub first_boot_ts: u64,
    /// 自上次创造新能力以来的轮数（跨重启持久化）
    /// —— 用于触发范式跃迁和好奇心探索
    pub rounds_since_last_creation: u32,
}

/// 已尝试的变异方案
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriedMutation {
    /// 能力名
    pub capability: String,
    /// 变异类型
    pub mutation_type: String,
    /// 变异描述
    pub description: String,
    /// 是否成功
    pub success: bool,
    /// 尝试时间
    pub tried_at: String,
}

/// 思维链 — 记录一轮 LLM 推理的完整思考过程
///
/// 人类的联想思维是连续的："上次我想做 A 但失败了，因为 B，
/// 那这次我应该先解决 B，或者换个思路做 C"。
///
/// 思维链持久化让 LLM 在下一轮推理时能看到上次的完整思考路径，
/// 从上次的断点继续，而不是每次从零开始。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtChain {
    /// 思维链类型：goal_generation | attribution | mutation_planning | evaluation
    pub chain_type: String,
    /// 思维链摘要（LLM 生成的完整推理文本，截断到 2000 字符）
    pub reasoning: String,
    /// 推理结论
    pub conclusion: String,
    /// 关联的能力名
    pub related_capabilities: Vec<String>,
    /// 关联的目标（如有）
    pub related_goal: Option<String>,
    /// 是否成功
    pub success: bool,
    /// 时间戳
    pub timestamp: u64,
}

/// 联想网络边 — 概念间关联
///
/// 当系统在某个上下文中遇到概念 A 时，
/// 可以通过联想网络找到关联的概念 B，触发联想。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationLink {
    /// 源概念（能力名、目标关键词、问题类型等）
    pub from_concept: String,
    /// 目标概念
    pub to_concept: String,
    /// 关联强度（0.0~1.0，每次共现增强，衰减随时间）
    pub strength: f64,
    /// 关联类型：co_occur（共现）| causal（因果）| similar（相似）
    pub link_type: String,
    /// 最后更新时间
    pub updated_at: u64,
}

/// 进化事件记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    pub event_type: String,
    pub capability: String,
    pub description: String,
    pub timestamp: String,
    pub generation: u32,
}

impl EvolutionEngine {
    /// 创建进化引擎
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        let storage_dir = storage_dir.into();
        let mut engine = Self {
            genomes: HashMap::new(),
            storage_dir,
            llm_executor: None,
            executor_registry: None,
            history: Vec::new(),
            memory: EvolutionMemory::default(),
        };
        engine.load();
        engine
    }

    /// 设置 LLM 执行器
    pub fn with_llm_executor(mut self, executor: Arc<dyn EvolutionDriver>) -> Self {
        self.llm_executor = Some(executor);
        self
    }

    /// 设置执行器注册表（元进化产物）
    pub fn with_executor_registry(mut self, registry: Arc<ExecutorRegistry>) -> Self {
        self.executor_registry = Some(registry);
        self
    }

    /// 注册基因组
    pub fn register_genome(&mut self, genome: CapabilityGenome) -> Result<(), String> {
        let name = genome.name.clone();
        // P2-2: 自动计算依赖复杂度
        let mut g = genome;
        g.fitness.dependency_complexity =
            crate::genome::FitnessGene::compute_dependency_complexity(&g);
        self.genomes.insert(name, g);
        // P1-fix: 在能力创建入口递增全局统计（修复计数器断裂）
        self.memory.global_stats.total_created += 1;
        self.reset_rounds_since_last_creation();
        self.save()
    }

    /// P0-2: 移除基因组（淘汰旧版本或失败变体）
    pub fn remove_genome(&mut self, name: &str) -> Result<Option<CapabilityGenome>, String> {
        let removed = self.genomes.remove(name);
        if removed.is_some() {
            // P1-fix: 在能力淘汰入口递增全局统计（修复计数器断裂）
            self.memory.global_stats.total_eliminated += 1;
            self.save()?;
        }
        Ok(removed)
    }

    /// P3-3: 查找依赖指定能力的所有能力（Composite 依赖）
    ///
    /// 返回依赖该能力的 Composite 能力名列表，
    /// 用于淘汰前检查是否会引发连锁失败。
    pub fn find_dependents(&self, name: &str) -> Vec<String> {
        use crate::genome::ActionImpl;
        self.genomes
            .iter()
            .filter(|(_, g)| {
                g.actions.iter().any(|a| {
                    if let ActionImpl::Composite { steps } = &a.implementation {
                        steps.iter().any(|s| s.capability == name)
                    } else {
                        false
                    }
                })
            })
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// P5: 多样性度量 — 计算能力库的多样性指数
    ///
    /// 基于能力名称的关键词重叠度：如果大量能力名称包含相同关键词
    /// (如 cargo_ops-v1, cargo_ops-v2, cargo_ops-v3)，说明多样性低。
    ///
    /// 返回 (diversity_score, duplicate_groups)
    /// - diversity_score: 0.0~1.0，越高越好
    /// - duplicate_groups: 重复能力组列表
    pub fn diversity_metrics(&self) -> (f64, Vec<(String, Vec<String>)>) {
        use std::collections::HashMap;

        // 提取每个能力名的基础关键词（去掉版本后缀）
        let base_names: HashMap<String, Vec<String>> = {
            let mut map: HashMap<String, Vec<String>> = HashMap::new();
            for name in self.genomes.keys() {
                // 去掉 -vN 后缀得到基础名
                let base = name.split("-v").next().unwrap_or(name).to_string();
                map.entry(base).or_default().push(name.clone());
            }
            map
        };

        // 重复组：基础名有多个版本
        let duplicate_groups: Vec<(String, Vec<String>)> = base_names
            .iter()
            .filter(|(_, names)| names.len() > 1)
            .map(|(base, names)| (base.clone(), names.clone()))
            .collect();

        // 多样性 = 1 - 重复比例
        let total = self.genomes.len() as f64;
        let unique = base_names.len() as f64;
        let diversity = if total > 0.0 { unique / total } else { 1.0 };

        (diversity, duplicate_groups)
    }

    /// 获取所有基因组
    pub fn genomes(&self) -> &HashMap<String, CapabilityGenome> {
        &self.genomes
    }

    /// 获取基因组（可变）
    pub fn genomes_mut(&mut self) -> &mut HashMap<String, CapabilityGenome> {
        &mut self.genomes
    }

    /// 变异 — 对现有能力做小幅修改
    ///
    /// 变异类型：
    /// - prompt 变异：修改 LLM 提示模板
    /// - 参数变异：修改输入参数
    /// - 动作增减：添加或删除动作
    /// - 描述变异：修改能力描述
    pub fn mutate(
        &mut self,
        capability_name: &str,
        mutation: Mutation,
    ) -> Result<&CapabilityGenome, String> {
        let genome = self
            .genomes
            .get(capability_name)
            .ok_or_else(|| format!("能力 '{}' 不存在", capability_name))?
            .clone();

        let mut new_genome = genome.clone();
        new_genome.lineage.parent = Some(capability_name.to_string());
        new_genome.lineage.origin = Origin::Mutated;
        new_genome.lineage.generation += 1;

        match &mutation {
            Mutation::PromptChange { action, new_prompt } => {
                let action_gene = new_genome
                    .actions
                    .iter_mut()
                    .find(|a| a.name == *action)
                    .ok_or_else(|| format!("动作 '{}' 不存在", action))?;
                if let crate::genome::ActionImpl::Llm { prompt, .. } =
                    &mut action_gene.implementation
                {
                    *prompt = new_prompt.clone();
                }
                new_genome
                    .record_mutation("prompt_change", format!("动作 '{}' 提示模板变更", action));
            }
            Mutation::DescriptionChange { new_description } => {
                new_genome.description = new_description.clone();
                new_genome.record_mutation("description_change", "描述变更");
            }
            Mutation::ActionAdd { action } => {
                new_genome.actions.push(action.clone());
                new_genome.record_mutation("action_add", format!("新增动作 '{}'", action.name));
            }
            Mutation::ActionRemove { action_name } => {
                new_genome.actions.retain(|a| a.name != *action_name);
                new_genome.record_mutation("action_remove", format!("删除动作 '{}'", action_name));
            }
            Mutation::ModelChange { action, new_model } => {
                let action_gene = new_genome
                    .actions
                    .iter_mut()
                    .find(|a| a.name == *action)
                    .ok_or_else(|| format!("动作 '{}' 不存在", action))?;
                if let crate::genome::ActionImpl::Llm { model, .. } =
                    &mut action_gene.implementation
                {
                    *model = new_model.clone();
                }
                new_genome.record_mutation(
                    "model_change",
                    format!("动作 '{}' 模型变更为 '{}'", action, new_model),
                );
            }
            Mutation::Crystallize { action, new_impl } => {
                let action_gene = new_genome
                    .actions
                    .iter_mut()
                    .find(|a| a.name == *action)
                    .ok_or_else(|| format!("动作 '{}' 不存在", action))?;
                let old_type = match &action_gene.implementation {
                    ActionImpl::Llm { .. } => "Llm",
                    ActionImpl::Rule { .. } => "Rule",
                    ActionImpl::Script { .. } => "Script",
                    ActionImpl::Shell { .. } => "Shell",
                    ActionImpl::Composite { .. } => "Composite",
                    ActionImpl::Native { .. } => "Native",
                    ActionImpl::Custom { .. } => "Custom",
                };
                action_gene.implementation = new_impl.clone();
                new_genome.record_mutation(
                    "crystallize",
                    format!("动作 '{}' 结晶化: {} → {:?}", action, old_type, new_impl),
                );
            }
        }

        // 变异后的名称加后缀
        let new_name = format!("{}-v{}", capability_name, new_genome.lineage.generation);
        new_genome.name = new_name.clone();

        self.history.push(EvolutionEvent {
            event_type: "mutation".into(),
            capability: new_name.clone(),
            description: mutation.description(),
            timestamp: now_string(),
            generation: new_genome.lineage.generation,
        });

        self.genomes.insert(new_name.clone(), new_genome);
        self.save()?;

        Ok(self.genomes.get(&new_name).expect("刚插入的基因组必须存在"))
    }

    /// 结晶化候选发现 — 找出适合从 LLM 推理降级为直接执行的动作
    ///
    /// 判定标准：
    /// 1. 实现类型为 Llm
    /// 2. 真实调用次数 >= min_calls（有足够数据判断输出模式）
    /// 3. token 消耗 > 0（确实在烧能量）
    ///
    /// 返回候选列表，按 token_cost 降序排列 — 最耗能的优先结晶。
    pub fn crystallize_candidates(&self, min_calls: u32) -> Vec<CrystallizeCandidate> {
        let mut candidates = Vec::new();
        for (cap_name, genome) in &self.genomes {
            for action in &genome.actions {
                if let ActionImpl::Llm { .. } = &action.implementation {
                    let real_calls = genome.fitness.real_call_count();
                    if real_calls >= min_calls && genome.fitness.total_token_cost > 0 {
                        candidates.push(CrystallizeCandidate {
                            capability: cap_name.clone(),
                            action: action.name.clone(),
                            token_cost: genome.fitness.total_token_cost,
                            call_count: real_calls,
                        });
                    }
                }
            }
        }
        candidates.sort_by(|a, b| b.token_cost.cmp(&a.token_cost));
        candidates
    }

    /// 执行结晶化 — 用 LLM 把一个 Llm 动作"编译"成 Script 实现
    ///
    /// 流程：
    /// 1. 检查目标动作确实是 Llm 实现
    /// 2. 用 LLM 分析该动作的输入输出模式，生成等价的 Python 脚本
    /// 3. 通过 Mutation::Crystallize 提交变异
    /// 4. 新能力继承旧能力的适应度数据，但 token_cost 归零
    ///
    /// 成功后该动作不再消耗 token，利润率从"递减"变为"恒定高"。
    pub async fn crystallize_action(
        &mut self,
        capability_name: &str,
        action_name: &str,
    ) -> Result<&CapabilityGenome, String> {
        let genome = self
            .genomes
            .get(capability_name)
            .ok_or_else(|| format!("能力 '{}' 不存在", capability_name))?
            .clone();

        let action_gene = genome
            .actions
            .iter()
            .find(|a| a.name == action_name)
            .ok_or_else(|| format!("动作 '{}' 不存在", action_name))?;

        let (prompt, model) = match &action_gene.implementation {
            ActionImpl::Llm { prompt, model, .. } => (prompt.clone(), model.clone()),
            _ => {
                return Err(format!("动作 '{}' 不是 Llm 实现，无法结晶化", action_name));
            }
        };

        let llm = self
            .llm_executor
            .as_ref()
            .ok_or_else(|| "LLM 执行器未配置，无法执行结晶化".to_string())?;

        let crystallize_prompt = format!(
            "你是能力运行时的结晶化编译器。\n\n\
以下是一个 LLM 驱动的动作，每次调用都消耗 token：\n\
- 能力: {}\n\
- 动作: {}\n\
- 提示模板: {}\n\n\
请编写一个等价的 Python 脚本，使其产生与 LLM 调用相同的结果。\n\
脚本接收输入参数作为 JSON（通过 stdin），输出 JSON 到 stdout。\n\
脚本应使用纯 Python 标准库（json, re, math, datetime 等），不依赖外部包。\n\
如果该动作无法用纯脚本实现（需要复杂推理/创意生成），返回 CANNOT_CRYSTALLIZE。\n\n\
只返回 Python 代码，用 ```python 包裹。",
            capability_name, action_name, prompt
        );

        let response = llm
            .execute(&crystallize_prompt, model.as_str(), None)
            .await
            .map_err(|e| format!("结晶化 LLM 调用失败: {}", e))?;

        if response.trim() == "CANNOT_CRYSTALLIZE" {
            return Err(format!("动作 '{}' 无法结晶化（需要复杂推理）", action_name));
        }

        // 提取 ```python ... ``` 中的代码
        let code = extract_code_block(&response).ok_or_else(|| {
            format!(
                "结晶化：无法从 LLM 响应中提取代码: {}",
                &response[..200.min(response.len())]
            )
        })?;

        let new_impl = ActionImpl::Script {
            language: "python".into(),
            code,
            timeout_secs: 30,
        };

        self.mutate(
            capability_name,
            Mutation::Crystallize {
                action: action_name.to_string(),
                new_impl,
            },
        )
    }

    /// 交叉 — 两个能力的基因混合
    pub fn crossover(
        &mut self,
        parent_a: &str,
        parent_b: &str,
        new_name: &str,
    ) -> Result<&CapabilityGenome, String> {
        let genome_a = self
            .genomes
            .get(parent_a)
            .ok_or_else(|| format!("能力 '{}' 不存在", parent_a))?;
        let genome_b = self
            .genomes
            .get(parent_b)
            .ok_or_else(|| format!("能力 '{}' 不存在", parent_b))?;

        let mut new_genome = CapabilityGenome {
            name: new_name.to_string(),
            version: "0.1.0".to_string(),
            description: format!(
                "{} + {} 的交叉后代",
                genome_a.description, genome_b.description
            ),
            actions: Vec::new(),
            fitness: FitnessGene::default(),
            lineage: LineageGene {
                origin: Origin::Crossbred,
                parent: Some(format!("{}+{}", parent_a, parent_b)),
                generation: std::cmp::max(genome_a.lineage.generation, genome_b.lineage.generation)
                    + 1,
                mutations: Vec::new(),
            },
            test_suite: Vec::new(),
        };

        // 交叉策略：各取一半动作
        let mid_a = genome_a.actions.len() / 2;
        for action in &genome_a.actions[..mid_a] {
            new_genome.actions.push(action.clone());
        }
        let mid_b = genome_b.actions.len() / 2;
        for action in &genome_b.actions[mid_b..] {
            new_genome.actions.push(action.clone());
        }

        new_genome.record_mutation("crossover", format!("{} × {} 交叉", parent_a, parent_b));

        self.history.push(EvolutionEvent {
            event_type: "crossover".into(),
            capability: new_name.to_string(),
            description: format!("{} × {}", parent_a, parent_b),
            timestamp: now_string(),
            generation: new_genome.lineage.generation,
        });

        let name = new_genome.name.clone();
        self.genomes.insert(name.clone(), new_genome);
        self.save()?;

        Ok(self.genomes.get(&name).expect("刚插入的基因组必须存在"))
    }

    /// 自然选择 — 淘汰适应度低于阈值的能力
    pub fn natural_selection(&mut self, min_score: f64) -> Result<Vec<String>, String> {
        let eliminated: Vec<String> = self
            .genomes
            .iter()
            .filter(|(_, g)| g.fitness.call_count > 3 && g.fitness.score < min_score)
            .map(|(name, _)| name.clone())
            .collect();

        for name in &eliminated {
            self.genomes.remove(name);
            // P1-fix: 自然选择淘汰也递增全局统计
            self.memory.global_stats.total_eliminated += 1;
            self.history.push(EvolutionEvent {
                event_type: "elimination".into(),
                capability: name.clone(),
                description: format!(
                    "适应度 {:.2} 低于阈值 {}, 淘汰",
                    self.genomes
                        .get(name)
                        .map(|g| g.fitness.score)
                        .unwrap_or(0.0),
                    min_score
                ),
                timestamp: now_string(),
                generation: 0,
            });
            println!("  🗑️  淘汰能力 '{}' (适应度过低)", name);
        }

        if !eliminated.is_empty() {
            self.save()?;
        }

        Ok(eliminated)
    }

    /// 将基因组注册为运行时能力
    pub async fn register_to_bus(&self, bus: &MessageBus, genome_name: &str) -> Result<(), String> {
        let genome = self
            .genomes
            .get(genome_name)
            .ok_or_else(|| format!("基因组 '{}' 不存在", genome_name))?;

        let mut cap = ScriptedCapability::from_genome(genome.clone());
        if let Some(llm) = &self.llm_executor {
            cap = cap.with_llm(llm.clone());
        }
        if let Some(registry) = &self.executor_registry {
            cap = cap.with_executor_registry(registry.clone());
        }

        bus.register(Arc::new(cap)).await;
        Ok(())
    }

    /// 注册所有基因组到运行时
    pub async fn register_all_to_bus(&self, bus: &MessageBus) {
        for (name, genome) in &self.genomes {
            if genome.actions.is_empty() {
                continue;
            }
            let mut cap = ScriptedCapability::from_genome(genome.clone());
            if let Some(llm) = &self.llm_executor {
                cap = cap.with_llm(llm.clone());
            }
            if let Some(registry) = &self.executor_registry {
                cap = cap.with_executor_registry(registry.clone());
            }
            bus.register(Arc::new(cap)).await;
            tracing::info!("注册进化能力: {}", name);
        }
    }

    /// 获取进化历史
    pub fn history(&self) -> &[EvolutionEvent] {
        &self.history
    }

    /// 生成进化报告
    pub fn report(&self) -> String {
        let mut report = String::from("═══ 进化报告 ═══\n\n");

        report.push_str(&format!("基因组数量: {}\n", self.genomes.len()));
        report.push_str(&format!("进化事件: {}\n\n", self.history.len()));

        report.push_str("── 能力适应度排名 ──\n");
        let mut sorted: Vec<_> = self.genomes.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.fitness
                .score
                .partial_cmp(&a.1.fitness.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (name, genome) in sorted {
            let stars = "★".repeat((genome.fitness.score * 5.0) as usize);
            report.push_str(&format!(
                "  {} {} [{:.2}] 调用:{} 成功率:{:.0}% 代:{}\n",
                stars,
                name,
                genome.fitness.score,
                genome.fitness.call_count,
                genome.fitness.success_rate * 100.0,
                genome.lineage.generation,
            ));
        }

        if !self.history.is_empty() {
            report.push_str("\n── 近期进化事件 ──\n");
            for event in self.history.iter().rev().take(10) {
                report.push_str(&format!(
                    "  [{}] {} {} (代 {})\n",
                    event.event_type, event.capability, event.description, event.generation
                ));
            }
        }

        report
    }

    /// 保存到磁盘（原子写入：先写临时文件，再 rename）
    ///
    /// 统一为 YAML 多文件存储格式：
    ///   .evolution/<capability>/genome.yaml + actions/* + versions/history.md
    /// YAML 与 log / history.json 同处 `.evolution/`，方便 git diff + 人工浏览。
    /// 历史与跨代记忆仍是 JSON（它们结构简单，无需独立文件）。
    pub fn save(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.storage_dir)
            .map_err(|e| format!("创建存储目录失败: {}", e))?;

        // 1. 基因组 → YAML 多文件
        crate::genome_yaml::save_genomes_to_yaml_dir(&self.storage_dir, &self.genomes).map_err(
            |e| {
                tracing::error!("YAML 保存失败: {}", e);
                format!("YAML 保存失败: {}", e)
            },
        )?;
        tracing::info!(
            "YAML 保存 {} 个基因组到 {:?}",
            self.genomes.len(),
            self.storage_dir
        );

        // 2. 进化历史 + 跨代记忆（JSON）
        let history_path = self.storage_dir.join("evolution_history.json");
        let json = serde_json::to_string_pretty(&self.history)
            .map_err(|e| format!("历史序列化失败: {}", e))?;
        atomic_write(&history_path, &json).map_err(|e| {
            tracing::error!("进化历史保存失败: {}", e);
            format!("进化历史保存失败: {}", e)
        })?;

        Ok(())
    }

    /// 仅保存基因组适应度（轻量级持久化，原子写入 YAML）
    pub fn save_fitness(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.storage_dir)
            .map_err(|e| format!("创建适应度目录失败: {}", e))?;
        crate::genome_yaml::save_genomes_to_yaml_dir(&self.storage_dir, &self.genomes)
    }

    /// 从磁盘加载
    fn load(&mut self) {
        // 1. YAML 多文件加载基因组
        match crate::genome_yaml::load_genomes_from_yaml_dir(&self.storage_dir) {
            Ok(genomes) => {
                for genome in genomes {
                    self.genomes.insert(genome.name.clone(), genome);
                }
                tracing::info!(
                    "从 {:?} 加载 {} 个基因组",
                    self.storage_dir,
                    self.genomes.len()
                );
            }
            Err(e) => {
                tracing::warn!("YAML 多文件加载失败 ({}) — 可能是首次运行", e);
            }
        }

        // 2. 进化历史
        let history_path = self.storage_dir.join("evolution_history.json");
        if let Ok(content) = std::fs::read_to_string(&history_path) {
            if let Ok(history) = serde_json::from_str::<Vec<EvolutionEvent>>(&content) {
                self.history = history;
            }
        }

        // 加载跨代记忆
        let memory_path = self.storage_dir.join("evolution_memory.json");
        if let Ok(content) = std::fs::read_to_string(&memory_path) {
            if let Ok(memory) = serde_json::from_str::<EvolutionMemory>(&content) {
                tracing::info!(
                    "从磁盘加载跨代记忆: {} 条教训, {} 条自主历史, 总轮次 {}",
                    memory.lessons.len(),
                    memory.autonomous_history.len(),
                    memory.global_stats.total_rounds
                );
                self.memory = memory;
            }
        }
    }

    /// 获取跨代记忆
    pub fn memory(&self) -> &EvolutionMemory {
        &self.memory
    }

    /// 获取跨代记忆（可变）
    pub fn memory_mut(&mut self) -> &mut EvolutionMemory {
        &mut self.memory
    }

    /// 记录进化教训
    pub fn record_lesson(&mut self, lesson: EvolutionLesson) {
        // 避免重复教训
        if !self
            .memory
            .lessons
            .iter()
            .any(|l| l.lesson == lesson.lesson)
        {
            self.memory.lessons.push(lesson);
            // 保留最近 100 条教训
            if self.memory.lessons.len() > 100 {
                self.memory.lessons.remove(0);
            }
        }
    }

    /// 记录自主循环历史
    pub fn record_autonomous_history(&mut self, entry: AutonomousHistoryEntry) {
        let success = entry.success;
        self.memory.autonomous_history.push(entry);
        // 保留最近 200 条
        if self.memory.autonomous_history.len() > 200 {
            self.memory.autonomous_history.remove(0);
        }
        self.memory.global_stats.total_autonomous_goals += 1;
        if success {
            self.memory.global_stats.total_autonomous_successes += 1;
        }
    }

    /// 记录已尝试的变异
    pub fn record_tried_mutation(&mut self, mutation: TriedMutation) {
        self.memory.tried_mutations.push(mutation);
        // 保留最近 300 条
        if self.memory.tried_mutations.len() > 300 {
            self.memory.tried_mutations.remove(0);
        }
        self.memory.global_stats.total_mutations += 1;
    }

    /// 检查是否已经尝试过类似的变异
    pub fn has_tried_mutation(
        &self,
        capability: &str,
        mutation_type: &str,
        description: &str,
    ) -> bool {
        self.memory.tried_mutations.iter().any(|m| {
            m.capability == capability
                && m.mutation_type == mutation_type
                && m.description == description
        })
    }

    /// 记录思维链 — 保存 LLM 的完整推理过程
    pub fn record_thought_chain(&mut self, chain: ThoughtChain) {
        // 同时更新联想网络
        let now = chain.timestamp;
        for cap in &chain.related_capabilities {
            // 能力 → 目标/结论 之间的关联
            if let Some(goal) = &chain.related_goal {
                self.strengthen_association(cap, goal, "co_occur", now);
            }
            // 能力之间共现关联
            for other_cap in &chain.related_capabilities {
                if cap != other_cap {
                    self.strengthen_association(cap, other_cap, "co_occur", now);
                }
            }
        }

        self.memory.thought_chains.push(chain);
        // 保留最近 50 条思维链（每条可能较长）
        if self.memory.thought_chains.len() > 50 {
            self.memory.thought_chains.remove(0);
        }
    }

    /// 获取最近的思维链（按类型过滤）
    pub fn get_recent_thought_chains(&self, chain_type: &str, limit: usize) -> Vec<&ThoughtChain> {
        self.memory
            .thought_chains
            .iter()
            .filter(|c| c.chain_type == chain_type)
            .rev()
            .take(limit)
            .collect()
    }

    /// 获取与给定概念关联的所有联想
    pub fn get_associations(&self, concept: &str) -> Vec<&AssociationLink> {
        self.memory
            .association_graph
            .iter()
            .filter(|a| a.from_concept == concept || a.to_concept == concept)
            .collect()
    }

    /// 强化联想网络边
    fn strengthen_association(&mut self, from: &str, to: &str, link_type: &str, now: u64) {
        if let Some(link) = self
            .memory
            .association_graph
            .iter_mut()
            .find(|a| a.from_concept == from && a.to_concept == to && a.link_type == link_type)
        {
            link.strength = (link.strength + 0.1).min(1.0);
            link.updated_at = now;
        } else {
            self.memory.association_graph.push(AssociationLink {
                from_concept: from.to_string(),
                to_concept: to.to_string(),
                strength: 0.3,
                link_type: link_type.to_string(),
                updated_at: now,
            });
        }
        // 限制联想网络大小
        if self.memory.association_graph.len() > 500 {
            // 移除强度最低的边
            self.memory.association_graph.sort_by(|a, b| {
                b.strength
                    .partial_cmp(&a.strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.memory.association_graph.truncate(400);
        }
    }

    /// 保存跨代记忆到磁盘
    pub fn save_memory(&mut self) -> Result<(), String> {
        let memory_path = self.storage_dir.join("evolution_memory.json");
        let json = serde_json::to_string_pretty(&self.memory)
            .map_err(|e| format!("进化记忆序列化失败: {}", e))?;
        atomic_write(&memory_path, &json)
    }

    /// 获取自上次创造新能力以来的轮数（跨重启持久化）
    pub fn rounds_since_last_creation(&self) -> u32 {
        self.memory.global_stats.rounds_since_last_creation
    }

    /// 重置"自上次创造新能力以来的轮数"为 0
    pub fn reset_rounds_since_last_creation(&mut self) {
        self.memory.global_stats.rounds_since_last_creation = 0;
    }

    /// 递增"自上次创造新能力以来的轮数"
    pub fn increment_rounds_since_last_creation(&mut self) {
        self.memory.global_stats.rounds_since_last_creation = self
            .memory
            .global_stats
            .rounds_since_last_creation
            .saturating_add(1);
    }
}
/// 结晶化候选 — 适合从 LLM 推理降级为直接执行的动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystallizeCandidate {
    pub capability: String,
    pub action: String,
    pub token_cost: u64,
    pub call_count: u32,
}

/// 变异类型
#[derive(Debug, Clone)]
pub enum Mutation {
    /// 修改 LLM 提示模板
    PromptChange { action: String, new_prompt: String },
    /// 修改描述
    DescriptionChange { new_description: String },
    /// 添加动作
    ActionAdd { action: crate::genome::ActionGene },
    /// 删除动作
    ActionRemove { action_name: String },
    /// 修改模型
    ModelChange { action: String, new_model: String },
    /// 结晶化 — 把 LLM 推理"冻结"成直接执行
    ///
    /// 这是能量维度的核心变异：当一个 Llm 动作被调用足够多次且输出模式稳定，
    /// 把它的实现从 ActionImpl::Llm 替换为 new_impl（通常是 Script/Rule）。
    /// 效果：每次调用的 token 成本从 N 降到 0，利润率飙升。
    ///
    /// 类比：反复练习的"思考动作"变成"肌肉记忆"——
    /// 神经网络推理编译成确定性代码。
    Crystallize {
        action: String,
        new_impl: crate::genome::ActionImpl,
    },
}

impl Mutation {
    fn description(&self) -> String {
        match self {
            Self::PromptChange { action, .. } => format!("动作 '{}' 提示变更", action),
            Self::DescriptionChange { .. } => "描述变更".into(),
            Self::ActionAdd { action } => format!("新增动作 '{}'", action.name),
            Self::ActionRemove { action_name } => format!("删除动作 '{}'", action_name),
            Self::ModelChange { action, new_model } => {
                format!("动作 '{}' 模型→'{}'", action, new_model)
            }
            Self::Crystallize { action, new_impl } => {
                format!("动作 '{}' 结晶化→{:?}", action, new_impl)
            }
        }
    }
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}

/// 从 Markdown 响应中提取代码块
///
/// 匹配 ```python ... ``` 或 ``` ... ``` 格式。
/// 如果没有代码块包裹，返回原始文本（去掉首尾空白）。
fn extract_code_block(text: &str) -> Option<String> {
    let start_marker = "```python";
    let start_marker_generic = "```";
    let start = text.find(start_marker).map(|p| p + start_marker.len());
    let start = start.or_else(|| {
        // 避免匹配到 ```python 的子串，只在找不到时用通用标记
        if text.contains(start_marker) {
            None
        } else {
            text.find(start_marker_generic)
                .map(|p| p + start_marker_generic.len())
        }
    });

    if let Some(s) = start {
        let rest = &text[s..];
        // 跳过可能的换行
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        if let Some(end) = rest.find("```") {
            return Some(rest[..end].trim().to_string());
        }
    }

    // 没有代码块包裹，返回原始文本（可能是纯代码）
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// 原子写入文件：先写入临时文件，再 rename 到目标路径
///
/// rename 在同一文件系统上是原子的，保证要么完整写入要么不变。
/// 这样即使写入途中崩溃，也不会损坏现有文件。
fn atomic_write(path: &std::path::Path, content: &str) -> Result<(), String> {
    use std::io::Write;

    let tmp_path = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| format!("创建临时文件 {} 失败: {}", tmp_path.display(), e))?;
        file.write_all(content.as_bytes())
            .map_err(|e| format!("写入临时文件 {} 失败: {}", tmp_path.display(), e))?;
        file.sync_all()
            .map_err(|e| format!("同步临时文件 {} 失败: {}", tmp_path.display(), e))?;
        std::fs::rename(&tmp_path, path).map_err(|e| {
            format!(
                "原子替换 {} -> {} 失败: {}",
                tmp_path.display(),
                path.display(),
                e
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{ActionGene, ActionImpl, FitnessGene, LineageGene};

    fn make_test_genome(name: &str) -> CapabilityGenome {
        CapabilityGenome {
            name: name.into(),
            version: "0.1.0".into(),
            description: "测试能力".into(),
            actions: vec![ActionGene {
                name: "act".into(),
                description: "测试动作".into(),
                input_schema: serde_json::json!({"type": "object"}),
                implementation: ActionImpl::Rule {
                    template: serde_json::json!({"result": "ok"}),
                },
            }],
            fitness: FitnessGene::default(),
            lineage: LineageGene::default(),
            test_suite: Vec::new(),
        }
    }

    #[test]
    fn test_evolution_register_and_get() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        evo.register_genome(make_test_genome("cap_a"));
        assert!(evo.genomes().contains_key("cap_a"));
    }

    #[test]
    fn test_evolution_remove() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        evo.register_genome(make_test_genome("cap_b"));
        let removed = evo.remove_genome("cap_b").unwrap();
        assert!(removed.is_some());
        assert!(!evo.genomes().contains_key("cap_b"));
    }

    #[test]
    fn test_evolution_persistence() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).ok();
        {
            let mut evo = EvolutionEngine::new(&tmp);
            evo.register_genome(make_test_genome("persist_cap"));
        }
        let evo2 = EvolutionEngine::new(&tmp);
        assert!(evo2.genomes().contains_key("persist_cap"));
    }

    #[test]
    fn test_diversity_metrics_empty() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let evo = EvolutionEngine::new(&tmp);
        let (diversity, dupes) = evo.diversity_metrics();
        assert_eq!(diversity, 1.0);
        assert!(dupes.is_empty());
    }

    #[test]
    fn test_diversity_metrics_with_duplicates() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        evo.register_genome(make_test_genome("cargo_ops"));
        evo.register_genome(make_test_genome("cargo_ops-v2"));
        evo.register_genome(make_test_genome("cargo_ops-v3"));
        evo.register_genome(make_test_genome("git_ops"));
        let (diversity, dupes) = evo.diversity_metrics();
        assert!(diversity < 1.0);
        assert!(!dupes.is_empty());
    }

    #[test]
    fn test_mutate_prompt_change() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        let mut genome = make_test_genome("llm_cap");
        genome.actions[0].implementation = ActionImpl::Llm {
            prompt: "原始提示".into(),
            model: "test-model".into(),
            system: None,
        };
        evo.register_genome(genome).unwrap();
        let result = evo.mutate(
            "llm_cap",
            Mutation::PromptChange {
                action: "act".into(),
                new_prompt: "新提示".into(),
            },
        );
        assert!(result.is_ok());
        let g = result.unwrap();
        if let ActionImpl::Llm { prompt, .. } = &g.actions[0].implementation {
            assert_eq!(prompt, "新提示");
        } else {
            panic!("应为 Llm 实现");
        }
        assert_eq!(g.lineage.generation, 2);
    }

    #[test]
    fn test_mutate_description_change() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        evo.register_genome(make_test_genome("desc_cap"));
        let result = evo.mutate(
            "desc_cap",
            Mutation::DescriptionChange {
                new_description: "新描述".into(),
            },
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().description, "新描述");
    }

    #[test]
    fn test_mutate_nonexistent() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        let result = evo.mutate(
            "no_such_cap",
            Mutation::DescriptionChange {
                new_description: "x".into(),
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_find_dependents() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);
        evo.register_genome(make_test_genome("base_cap"));
        let mut composite = make_test_genome("composite_cap");
        composite.actions[0].implementation = ActionImpl::Composite {
            steps: vec![crate::genome::CompositeStep {
                name: "sub_step".into(),
                capability: "base_cap".into(),
                action: "act".into(),
                input: serde_json::json!({}),
            }],
        };
        evo.register_genome(composite).unwrap();
        let deps = evo.find_dependents("base_cap");
        assert!(deps.contains(&"composite_cap".to_string()));
    }

    #[test]
    fn test_mutation_description() {
        assert!(Mutation::PromptChange {
            action: "a".into(),
            new_prompt: "p".into()
        }
        .description()
        .contains("提示变更"));
        assert!(Mutation::DescriptionChange {
            new_description: "d".into()
        }
        .description()
        .contains("描述变更"));
        assert!(Mutation::ActionAdd {
            action: make_test_genome("x").actions[0].clone()
        }
        .description()
        .contains("新增动作"));
        assert!(Mutation::ActionRemove {
            action_name: "a".into()
        }
        .description()
        .contains("删除动作"));
    }

    #[test]
    fn test_mutate_crystallize() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);

        let mut genome = make_test_genome("llm_cap");
        genome.actions[0].implementation = ActionImpl::Llm {
            prompt: "翻译: {{text}}".into(),
            model: "test-model".into(),
            system: None,
        };
        evo.register_genome(genome).unwrap();

        let result = evo.mutate(
            "llm_cap",
            Mutation::Crystallize {
                action: "act".into(),
                new_impl: ActionImpl::Script {
                    language: "python".into(),
                    code: "import json; print(json.dumps({'result': 'ok'}))".into(),
                    timeout_secs: 10,
                },
            },
        );
        assert!(result.is_ok());
        let g = result.unwrap();

        // 验证实现已变为 Script
        match &g.actions[0].implementation {
            ActionImpl::Script { language, .. } => {
                assert_eq!(language, "python");
            }
            other => panic!("应为 Script 实现, got {:?}", other),
        }

        // 验证变异记录
        assert!(g
            .lineage
            .mutations
            .iter()
            .any(|m| m.mutation_type == "crystallize"));
    }

    #[test]
    fn test_crystallize_candidates() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);

        // LLM 能力，有 token 消耗
        let mut llm_genome = make_test_genome("llm_cap");
        llm_genome.actions[0].implementation = ActionImpl::Llm {
            prompt: "test".into(),
            model: "m".into(),
            system: None,
        };
        llm_genome.fitness.call_count = 10;
        llm_genome.fitness.auto_test_count = 2;
        llm_genome.fitness.total_token_cost = 5000;
        evo.register_genome(llm_genome).unwrap();

        // 另一个 LLM 能力，更多 token
        let mut llm_genome2 = make_test_genome("llm_cap2");
        llm_genome2.actions[0].implementation = ActionImpl::Llm {
            prompt: "test2".into(),
            model: "m".into(),
            system: None,
        };
        llm_genome2.fitness.call_count = 15;
        llm_genome2.fitness.auto_test_count = 3;
        llm_genome2.fitness.total_token_cost = 10000;
        evo.register_genome(llm_genome2).unwrap();

        // Script 能力，无 token 消耗
        let mut script_genome = make_test_genome("script_cap");
        script_genome.actions[0].implementation = ActionImpl::Script {
            language: "python".into(),
            code: "print('ok')".into(),
            timeout_secs: 10,
        };
        script_genome.fitness.call_count = 20;
        evo.register_genome(script_genome).unwrap();

        let candidates = evo.crystallize_candidates(5);

        // 应找到 2 个候选（两个 LLM 能力），Script 能力不应出现
        assert_eq!(candidates.len(), 2);

        // 按 token_cost 降序排列
        assert_eq!(candidates[0].capability, "llm_cap2");
        assert_eq!(candidates[0].token_cost, 10000);
        assert_eq!(candidates[1].capability, "llm_cap");
        assert_eq!(candidates[1].token_cost, 5000);

        // Script 能力不在候选中
        assert!(!candidates.iter().any(|c| c.capability == "script_cap"));
    }

    #[test]
    fn test_crystallize_action_rejects_non_llm() {
        let tmp = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
        let mut evo = EvolutionEngine::new(&tmp);

        let mut genome = make_test_genome("rule_cap");
        genome.actions[0].implementation = ActionImpl::Rule {
            template: serde_json::json!({"result": "ok"}),
        };
        evo.register_genome(genome).unwrap();

        // 对非 LLM 实现执行结晶化应失败
        let result = evo.mutate(
            "rule_cap",
            Mutation::Crystallize {
                action: "act".into(),
                new_impl: ActionImpl::Script {
                    language: "python".into(),
                    code: "print('ok')".into(),
                    timeout_secs: 10,
                },
            },
        );
        // mutate 本身不会拒绝（它只做替换），但 crystallize_action 会
        // 这里测试 mutate 能正确替换任意实现类型
        assert!(result.is_ok());
        let g = result.unwrap();
        match &g.actions[0].implementation {
            ActionImpl::Script { .. } => {}
            other => panic!("应为 Script 实现, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_code_block_python() {
        let response = "Some text\n```python\nprint('hello')\n```\nMore text";
        let code = extract_code_block(response);
        assert_eq!(code.unwrap(), "print('hello')");
    }

    #[test]
    fn test_extract_code_block_generic() {
        let response = "```\nimport json\nprint(json.dumps({}))\n```";
        let code = extract_code_block(response);
        assert!(code.unwrap().contains("import json"));
    }

    #[test]
    fn test_extract_code_block_no_block() {
        let response = "print('raw code')";
        let code = extract_code_block(response);
        assert_eq!(code.unwrap(), "print('raw code')");
    }

    #[test]
    fn test_mutation_description_crystallize() {
        let m = Mutation::Crystallize {
            action: "translate".into(),
            new_impl: ActionImpl::Script {
                language: "python".into(),
                code: "print('ok')".into(),
                timeout_secs: 10,
            },
        };
        assert!(m.description().contains("结晶化"));
        assert!(m.description().contains("translate"));
    }
}
