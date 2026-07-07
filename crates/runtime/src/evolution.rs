use crate::genome::{CapabilityGenome, FitnessGene, LineageGene, Origin, ScriptedCapability, LlmExecutor};
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
pub struct EvolutionEngine {
    /// 基因组库 — 所有能力基因组
    genomes: HashMap<String, CapabilityGenome>,
    /// 持久化存储路径
    storage_dir: PathBuf,
    /// LLM 执行器（用于脚本化能力）
    llm_executor: Option<Arc<LlmExecutor>>,
    /// 执行器注册表（用于 Custom 类型能力 — 元进化产物）
    executor_registry: Option<Arc<ExecutorRegistry>>,
    /// 进化历史
    history: Vec<EvolutionEvent>,
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
        };
        engine.load();
        engine
    }

    /// 设置 LLM 执行器
    pub fn with_llm_executor(mut self, executor: Arc<LlmExecutor>) -> Self {
        self.llm_executor = Some(executor);
        self
    }

    /// 设置执行器注册表（元进化产物）
    pub fn with_executor_registry(mut self, registry: Arc<ExecutorRegistry>) -> Self {
        self.executor_registry = Some(registry);
        self
    }

    /// 注册基因组
    pub fn register_genome(&mut self, genome: CapabilityGenome) {
        let name = genome.name.clone();
        self.genomes.insert(name, genome);
        self.save();
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
        let genome = self.genomes.get(capability_name)
            .ok_or_else(|| format!("能力 '{}' 不存在", capability_name))?
            .clone();

        let mut new_genome = genome.clone();
        new_genome.lineage.parent = Some(capability_name.to_string());
        new_genome.lineage.origin = Origin::Mutated;
        new_genome.lineage.generation += 1;

        match &mutation {
            Mutation::PromptChange { action, new_prompt } => {
                let action_gene = new_genome.actions.iter_mut()
                    .find(|a| a.name == *action)
                    .ok_or_else(|| format!("动作 '{}' 不存在", action))?;
                if let crate::genome::ActionImpl::Llm { prompt, .. } = &mut action_gene.implementation {
                    *prompt = new_prompt.clone();
                }
                new_genome.record_mutation("prompt_change", 
                    format!("动作 '{}' 提示模板变更", action));
            }
            Mutation::DescriptionChange { new_description } => {
                new_genome.description = new_description.clone();
                new_genome.record_mutation("description_change", "描述变更");
            }
            Mutation::ActionAdd { action } => {
                new_genome.actions.push(action.clone());
                new_genome.record_mutation("action_add", 
                    format!("新增动作 '{}'", action.name));
            }
            Mutation::ActionRemove { action_name } => {
                new_genome.actions.retain(|a| a.name != *action_name);
                new_genome.record_mutation("action_remove", 
                    format!("删除动作 '{}'", action_name));
            }
            Mutation::ModelChange { action, new_model } => {
                let action_gene = new_genome.actions.iter_mut()
                    .find(|a| a.name == *action)
                    .ok_or_else(|| format!("动作 '{}' 不存在", action))?;
                if let crate::genome::ActionImpl::Llm { model, .. } = &mut action_gene.implementation {
                    *model = new_model.clone();
                }
                new_genome.record_mutation("model_change",
                    format!("动作 '{}' 模型变更为 '{}'", action, new_model));
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
        self.save();

        Ok(self.genomes.get(&new_name).unwrap())
    }

    /// 交叉 — 两个能力的基因混合
    pub fn crossover(
        &mut self,
        parent_a: &str,
        parent_b: &str,
        new_name: &str,
    ) -> Result<&CapabilityGenome, String> {
        let genome_a = self.genomes.get(parent_a)
            .ok_or_else(|| format!("能力 '{}' 不存在", parent_a))?;
        let genome_b = self.genomes.get(parent_b)
            .ok_or_else(|| format!("能力 '{}' 不存在", parent_b))?;

        let mut new_genome = CapabilityGenome {
            name: new_name.to_string(),
            version: "0.1.0".to_string(),
            description: format!("{} + {} 的交叉后代", genome_a.description, genome_b.description),
            actions: Vec::new(),
            fitness: FitnessGene::default(),
            lineage: LineageGene {
                origin: Origin::Crossbred,
                parent: Some(format!("{}+{}", parent_a, parent_b)),
                generation: std::cmp::max(genome_a.lineage.generation, genome_b.lineage.generation) + 1,
                mutations: Vec::new(),
            },
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

        new_genome.record_mutation("crossover",
            format!("{} × {} 交叉", parent_a, parent_b));

        self.history.push(EvolutionEvent {
            event_type: "crossover".into(),
            capability: new_name.to_string(),
            description: format!("{} × {}", parent_a, parent_b),
            timestamp: now_string(),
            generation: new_genome.lineage.generation,
        });

        let name = new_genome.name.clone();
        self.genomes.insert(name.clone(), new_genome);
        self.save();

        Ok(self.genomes.get(&name).unwrap())
    }

    /// 自然选择 — 淘汰适应度低于阈值的能力
    pub fn natural_selection(&mut self, min_score: f64) -> Vec<String> {
        let eliminated: Vec<String> = self.genomes.iter()
            .filter(|(_, g)| {
                g.fitness.call_count > 3 && g.fitness.score < min_score
            })
            .map(|(name, _)| name.clone())
            .collect();

        for name in &eliminated {
            self.genomes.remove(name);
            self.history.push(EvolutionEvent {
                event_type: "elimination".into(),
                capability: name.clone(),
                description: format!("适应度 {:.2} 低于阈值 {}, 淘汰", 
                    self.genomes.get(name).map(|g| g.fitness.score).unwrap_or(0.0), min_score),
                timestamp: now_string(),
                generation: 0,
            });
            println!("  🗑️  淘汰能力 '{}' (适应度过低)", name);
        }

        if !eliminated.is_empty() {
            self.save();
        }

        eliminated
    }

    /// 将基因组注册为运行时能力
    pub async fn register_to_bus(&self, bus: &MessageBus, genome_name: &str) -> Result<(), String> {
        let genome = self.genomes.get(genome_name)
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
        sorted.sort_by(|a, b| b.1.fitness.score.partial_cmp(&a.1.fitness.score).unwrap_or(std::cmp::Ordering::Equal));

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
    /// 原子化持久化是保护进化记忆的关键：
    /// 非原子写入在写入途中崩溃会导致 JSON 损坏，
    /// 进而丢失全部进化成果（基因组、适应度、谱系）。
    pub fn save(&self) {
        std::fs::create_dir_all(&self.storage_dir).ok();

        // 保存基因组
        let genomes_path = self.storage_dir.join("genomes.json");
        let genomes: Vec<_> = self.genomes.values().cloned().collect();
        if let Ok(json) = serde_json::to_string_pretty(&genomes) {
            atomic_write(&genomes_path, &json);
        }

        // 保存进化历史
        let history_path = self.storage_dir.join("evolution_history.json");
        if let Ok(json) = serde_json::to_string_pretty(&self.history) {
            atomic_write(&history_path, &json);
        }
    }

    /// 仅保存基因组适应度（轻量级持久化，原子写入）
    pub fn save_fitness(&self) {
        let genomes_path = self.storage_dir.join("genomes.json");
        let genomes: Vec<_> = self.genomes.values().cloned().collect();
        if let Ok(json) = serde_json::to_string_pretty(&genomes) {
            atomic_write(&genomes_path, &json);
        }
    }

    /// 从磁盘加载
    fn load(&mut self) {
        let genomes_path = self.storage_dir.join("genomes.json");
        if let Ok(content) = std::fs::read_to_string(&genomes_path) {
            if let Ok(genomes) = serde_json::from_str::<Vec<CapabilityGenome>>(&content) {
                for genome in genomes {
                    self.genomes.insert(genome.name.clone(), genome);
                }
                tracing::info!("从磁盘加载 {} 个基因组", self.genomes.len());
            }
        }

        let history_path = self.storage_dir.join("evolution_history.json");
        if let Ok(content) = std::fs::read_to_string(&history_path) {
            if let Ok(history) = serde_json::from_str::<Vec<EvolutionEvent>>(&content) {
                self.history = history;
            }
        }
    }
}

/// 变异类型
#[derive(Debug, Clone)]
pub enum Mutation {
    /// 修改 LLM 提示模板
    PromptChange {
        action: String,
        new_prompt: String,
    },
    /// 修改描述
    DescriptionChange {
        new_description: String,
    },
    /// 添加动作
    ActionAdd {
        action: crate::genome::ActionGene,
    },
    /// 删除动作
    ActionRemove {
        action_name: String,
    },
    /// 修改模型
    ModelChange {
        action: String,
        new_model: String,
    },
}

impl Mutation {
    fn description(&self) -> String {
        match self {
            Self::PromptChange { action, .. } => format!("动作 '{}' 提示变更", action),
            Self::DescriptionChange { .. } => "描述变更".into(),
            Self::ActionAdd { action } => format!("新增动作 '{}'", action.name),
            Self::ActionRemove { action_name } => format!("删除动作 '{}'", action_name),
            Self::ModelChange { action, new_model } => format!("动作 '{}' 模型→'{}'", action, new_model),
        }
    }
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}

/// 原子写入文件：先写入临时文件，再 rename 到目标路径
///
/// rename 在同一文件系统上是原子的，保证要么完整写入要么不变。
/// 这样即使写入途中崩溃，也不会损坏现有文件。
fn atomic_write(path: &std::path::Path, content: &str) {
    let tmp_path = path.with_extension("tmp");
    if std::fs::write(&tmp_path, content).is_err() {
        return;
    }
    let _ = std::fs::rename(&tmp_path, path);
}
