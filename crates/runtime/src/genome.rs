use crate::capability::Capability;
use crate::driver::EvolutionDriver;
use crate::message::{Message, MessageError, MessageResult};
use crate::message_bus::MessageBus;
use crate::sandbox::{Sandbox, SandboxConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

    /// P4: 持久化测试套件 — 积累的测试用例，变异后回归测试
    #[serde(default)]
    pub test_suite: Vec<TestCase>,
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
    /// Shell 命令 — 直接用 shell 执行命令（适合简单命令行工具包装）
    ///
    /// 与 Script 类似但更轻量：AI 只需给出 shell 命令字符串，
    /// 运行时用 `bash -c` 执行，支持 {{var}} 模板插值。
    /// 这是 LLM 常用的"Shell" 实现类型（如 brew/系统命令包装）。
    Shell {
        /// Shell 命令（支持 {{var}} 模板插值，变量来自输入参数）
        command: String,
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

impl ActionImpl {
    /// P3b-fix: 提取可变异的代码/提示词文本（用于失败 lesson 存档）
    ///
    /// 返回 Script.code / Shell.command / Llm.prompt 的内容，
    /// 其他变体返回 None。
    pub fn code_string(&self) -> Option<String> {
        match self {
            ActionImpl::Script { code, .. } => Some(code.clone()),
            ActionImpl::Shell { command, .. } => Some(command.clone()),
            ActionImpl::Llm { prompt, .. } => Some(prompt.clone()),
            _ => None,
        }
    }
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
    #[serde(default)]
    pub rounds_dormant: u32,
    /// P2-2: 输出质量评分（0.0 ~ 1.0）
    /// 基于输出内容非空率、结构化程度、信息量
    #[serde(default)]
    pub output_quality: f64,
    /// P2-2: 覆盖范围评分（0.0 ~ 1.0）
    /// 基于不同输入组合的测试覆盖度
    #[serde(default)]
    pub coverage_score: f64,
    /// P2-2: 依赖复杂度（0.0 ~ 1.0，越低越好）
    /// 基于能力依赖的外部工具数量和环境要求
    #[serde(default)]
    pub dependency_complexity: f64,
    /// P2-2: 累计输出非空次数（用于计算 output_quality）
    #[serde(default)]
    pub non_empty_output_count: u32,
    /// LLM 评估的创新性评分（0.0 ~ 1.0）
    /// 评估能力是否提供了独特的功能，而非重复已有能力
    #[serde(default)]
    pub innovation_score: f64,
    /// LLM 评估的实用性评分（0.0 ~ 1.0）
    /// 评估能力在实际场景中的有用程度
    #[serde(default)]
    pub utility_score: f64,
    /// LLM 评估的时间戳
    #[serde(default)]
    pub llm_evaluated_at: u64,
    /// 真实环境验证通过次数（环境验证器判定为真实成功的累计次数）
    ///
    /// 与 `call_count` / `auto_test_count` 正交：那些计数"能力说自己成功"，
    /// 这个字段记录"环境证明它真的成功了"（如 cargo 能力执行后 cargo build 通过、
    /// git 能力执行后 git status 有效）。参与 `recompute_score` 加权，让通过环境
    /// 验证的能力分数真实抬升——这是"真实世界压力"喂回 fitness 的硬信号。
    #[serde(default)]
    pub real_validation_passes: u32,
    /// 真实环境验证失败次数（负反馈）
    ///
    /// 能力自报成功但环境验证器判定真实失败(如 cargo build 没过、fork 仓库测试挂了)。
    /// 参与 `recompute_score` 真实轨的通过比计算,失败越多真实轨分越低,加速淘汰——
    /// 这修补了原来"只有正反馈、失败只走弱自报 failure_count"的短板。
    #[serde(default)]
    pub real_validation_failures: u32,
    /// 该能力获得过的最强真实信号层级
    ///
    /// 反映"真实世界背书的最高可信度"——SelfReport(0.1) 到 RealTask(0.9)。
    /// 决定 `recompute_score` 里真实轨对自报轨的压制程度:最强信号越强,
    /// 自报轨权重越被压缩,真实轨主导 score。这是"真实>>自报"的杠杆。
    #[serde(default)]
    pub strongest_signal: crate::validator::SignalStrength,
    /// 最强自动验证信号，与人类价值信号分开保存。
    ///
    /// 旧数据缺少该字段时回退到 SelfReport；recompute_score 会在
    /// strongest_signal 尚非 HumanValue 时自动兼容旧字段。
    #[serde(default)]
    pub strongest_automatic_signal: crate::validator::SignalStrength,
    /// 人类实测信号累计次数
    ///
    /// 与所有自动信号正交:人实测能力后判定"有用/无用"的次数。0 表示从未被人类评估。
    /// 决定 `recompute_score` 里人类轨是否启用——一旦 >0,人类评分主导 score
    /// (因 HumanValue 信号强度 0.95 高于所有自动信号)。
    #[serde(default)]
    pub human_signals_count: u32,
    /// 人类实测评分均值 (0.0 ~ 1.0)
    ///
    /// 每次"有用"记 +1.0,"无用"记 0.0,取均值。1.0=所有人都觉得有用,0.0=都觉得无用。
    /// 这是对主观价值型能力(数据科学类)唯一的"有用性"度量,填补自动 fitness
    /// 只测"能跑通"的盲区。随 YAML genome 持久化,跨重启保留。
    #[serde(default)]
    pub human_score: f64,
    /// 累计 token 消耗 — 该能力所有 LLM 调用的 token 总和
    ///
    /// 能量的直接度量：token 就是钱。LLM 实现每次调用都花 token，
    /// Script/Rule/Native 实现不花 token。这个字段追踪"这个能力烧了多少能量"。
    #[serde(default)]
    pub total_token_cost: u64,
    /// 上次调用的 token 消耗（用于增量追踪）
    #[serde(default)]
    pub last_token_cost: u64,
    /// 能量利用率（利润率）— 产出价值 / 能量消耗
    ///
    /// 利润率 = success_rate * call_count / (total_token_cost + 1)
    /// 高利润率 = 用极少 token 产出大量成功结果（直接执行 >> 推理）
    /// 低利润率 = 每次都烧 token 才能干活（纯 LLM 推理）
    /// 结晶化（Llm→Script）会让利润率飙升：分母趋近于 0。
    #[serde(default)]
    pub profit_ratio: f64,
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
            self.non_empty_output_count += 1;
        } else {
            self.failure_count += 1;
        }
        // 滚动平均延迟
        let n = self.call_count as f64;
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1.0) + latency_ms) / n;
        self.success_rate = self.success_count as f64 / self.call_count as f64;

        // P2-2: 计算输出质量
        self.output_quality = self.non_empty_output_count as f64 / self.call_count as f64;

        // P2-2: 计算覆盖范围（基于调用次数的 log 缩放）
        self.coverage_score = (self.call_count as f64).ln_1p() / 10.0;
        if self.coverage_score > 1.0 {
            self.coverage_score = 1.0;
        }

        // P2-2: 综合评分 = 成功率 * 速度因子 * (0.5 + 0.3*输出质量 + 0.2*覆盖范围) * (1 - 0.1*依赖复杂度)
        let speed_factor = 1.0 / (1.0 + self.avg_latency_ms / 1000.0);
        let quality_factor = 0.5 + 0.3 * self.output_quality + 0.2 * self.coverage_score;
        let dependency_penalty = 1.0 - 0.1 * self.dependency_complexity;
        self.score = self.success_rate * speed_factor * quality_factor * dependency_penalty;
        self.last_evaluated = Some(now_string());
        self.recompute_profit_ratio();
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
            self.non_empty_output_count += 1;
        } else {
            self.failure_count += 1;
        }
        let n = self.call_count as f64;
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1.0) + latency_ms) / n;
        self.success_rate = self.success_count as f64 / self.call_count as f64;
        // P2-2: 自测试评分也纳入输出质量
        self.output_quality = self.non_empty_output_count as f64 / self.call_count as f64;
        self.last_evaluated = Some(now_string());
        // 与其他信号共用同一评分入口，避免首次人类反馈前后采用不同基线。
        self.recompute_score();
        // 注意：不清零 rounds_dormant
    }

    /// 记录一次人类实测反馈
    ///
    /// 这是 fitness 系统里**唯一**的"有用性"信号来源。自动信号(success_rate/real_validation)
    /// 只能证明能力"能跑通",证明不了"对人有用"。人类实测后判 useful/useless 直接进 human_score,
    /// 因 HumanValue 信号强度(0.95)高于所有自动信号,recompute_score 里人类轨一旦启用即主导 score。
    ///
    /// useful=true → 人类评分记 1.0,false → 记 0.0,滚动均值。累加 human_signals_count,
    /// 并把 strongest_signal 升级为 HumanValue(若还不是更高层级——HumanValue 已是最高)。
    /// 清零 rounds_dormant:人类实测过 = 这能力被"真实使用"过,不应因休眠被淘汰。
    pub fn record_human_signal(&mut self, useful: bool) {
        let prev_total = self.human_score * self.human_signals_count as f64;
        self.human_signals_count += 1;
        let new_point = if useful { 1.0 } else { 0.0 };
        self.human_score = (prev_total + new_point) / self.human_signals_count as f64;
        // 人类信号是最强层级,升级 strongest_signal
        if self.strongest_signal < crate::validator::SignalStrength::HumanValue {
            self.strongest_signal = crate::validator::SignalStrength::HumanValue;
        }
        self.rounds_dormant = 0;
        self.last_evaluated = Some(now_string());
        self.recompute_score();
    }

    /// 记录一次 LLM 调用的 token 消耗 — 能量记账
    ///
    /// 这是能量维度的核心记账方法。每次 LLM 实现被执行时调用，
    /// 累加 token 消耗。Script/Rule/Native 实现不调用此方法，
    /// 因此它们的 total_token_cost 始终为 0 — 利润率无穷大。
    ///
    /// 结晶化前：每次调用都花 token，利润率低
    /// 结晶化后：不再花 token，利润率飙升
    pub fn record_token_cost(&mut self, tokens: u64) {
        self.last_token_cost = tokens;
        self.total_token_cost = self.total_token_cost.saturating_add(tokens);
        self.recompute_profit_ratio();
    }

    /// 计算能量利用率（利润率）
    ///
    /// 利润率 = 产出价值 / 能量消耗
    /// 产出价值 = success_count（成功是真实产出）
    /// 能量消耗 = total_token_cost + call_count（token 是直接成本，
    ///   call_count 是间接成本如 CPU/延迟，+1 防除零）
    ///
    /// Llm 实现：token 持续增长 → 利润率递减
    /// Script 实现：token = 0 → 利润率 = success_count / call_count ≈ success_rate
    /// 这就是"直接执行比推理利润率高"的量化表达。
    pub fn recompute_profit_ratio(&mut self) {
        let output_value = self.success_count as f64;
        let energy_cost = self.total_token_cost as f64 + self.call_count as f64;
        if energy_cost == 0.0 {
            self.profit_ratio = 0.0;
            return;
        }
        self.profit_ratio = output_value / energy_cost;
    }

    /// P2-2: 计算依赖复杂度（0.0 ~ 1.0，越低越好）
    ///
    /// 基于 action 实现类型和外部依赖数量：
    /// - Script: 根据代码中 import 的外部库数量
    /// - Composite: 根据编排步骤数量
    /// - Llm: 固定 0.3（需要 LLM 后端）
    /// - Native: 0.0（无额外依赖）
    pub fn compute_dependency_complexity(genome: &CapabilityGenome) -> f64 {
        let mut max_complexity = 0.0;
        for action in &genome.actions {
            let complexity = match &action.implementation {
                ActionImpl::Script { code, language, .. } => {
                    if language == "python" {
                        let import_count = code
                            .lines()
                            .filter(|l| {
                                l.trim_start().starts_with("import ")
                                    || l.trim_start().starts_with("from ")
                            })
                            .count();
                        (import_count as f64 / 10.0).min(1.0)
                    } else {
                        0.5
                    }
                }
                ActionImpl::Composite { steps } => (steps.len() as f64 / 5.0).min(1.0),
                ActionImpl::Llm { .. } => 0.3,
                ActionImpl::Native { .. } => 0.0,
                ActionImpl::Rule { .. } => 0.0,
                ActionImpl::Custom { .. } => 0.5,
                ActionImpl::Shell { .. } => 0.5,
            };
            if complexity > max_complexity {
                max_complexity = complexity;
            }
        }
        max_complexity
    }

    pub fn recompute_score(&mut self) {
        if self.call_count > 0 {
            self.success_rate = self.success_count as f64 / self.call_count as f64;
            self.output_quality = self.non_empty_output_count as f64 / self.call_count as f64;
            self.coverage_score = (self.call_count as f64).ln_1p() / 10.0;
            if self.coverage_score > 1.0 {
                self.coverage_score = 1.0;
            }
        }
        let speed_factor = 1.0 / (1.0 + self.avg_latency_ms / 1000.0);
        let quality_factor = 0.5 + 0.3 * self.output_quality + 0.2 * self.coverage_score;
        let dependency_penalty = 1.0 - 0.1 * self.dependency_complexity;

        // ── 双轨加权:真实信号 >> 自报信号 ──
        // 自报轨(弱):成功率×速度×质量×依赖惩罚,封顶 0.3。
        // 纯靠"能力自己说成功"的能力最高只能到 ~0.3,防止自报虚高。
        let self_reported =
            (self.success_rate * speed_factor * quality_factor * dependency_penalty).min(0.3);

        // 真实轨(强):由该能力获得过的最强信号层级 × 真实验证通过比决定。
        // 通过比 = passes / (passes + failures + 1),失败拉低通过比 → 强负反馈。
        let total_validations = self.real_validation_passes + self.real_validation_failures;
        let pass_ratio = if total_validations == 0 {
            0.0
        } else {
            self.real_validation_passes as f64 / total_validations as f64
        };
        let automatic_signal =
            if self.strongest_signal == crate::validator::SignalStrength::HumanValue {
                self.strongest_automatic_signal
            } else {
                std::cmp::max(self.strongest_signal, self.strongest_automatic_signal)
            };
        let signal_weight = automatic_signal.weight(); // 0.1 ~ 0.9
        let real_track = signal_weight * pass_ratio; // 最高 ~0.9

        // 真实轨压制自报轨:最强信号越强,自报轨权重 (1 - signal_weight) 越小。
        // - 纯自报(无环境验证): strongest=SelfReport(0.1),自报轨保留 0.9,real_track=0 → score ≈ 0.9×自报(≤0.27) + 0.01
        // - 通过 cargo test: strongest=TestPass(0.7),自报轨压到 0.3,真实轨主导 → score 可达 0.7+
        // - 真实验证全失败: pass_ratio=0,real_track=0,score 来自被压制的自报轨 → 低分加速淘汰
        let real_dominance = signal_weight;
        let auto_score = self_reported * (1.0 - real_dominance)
            + real_track * real_dominance
            + real_dominance * 0.1;

        // ── 能量效率因子:利润率调制 ──
        // profit_ratio = success_count / (total_token_cost + call_count)
        // 纯 LLM 能力: token 持续增长, profit_ratio 趋近 0, 因子压分
        // 结晶化能力(Script/Rule/Native): token=0, profit_ratio ≈ success_rate, 因子接近 1
        // 用 1/(1+1/profit_ratio) 做 S 形压缩,避免利润率极端值导致分数跳变
        self.recompute_profit_ratio();
        let energy_factor = if self.profit_ratio > 0.0 {
            self.profit_ratio / (self.profit_ratio + 1.0)
        } else {
            0.0
        };
        let auto_score = auto_score * (0.5 + 0.5 * energy_factor);
        // 纯自测只能证明能力能运行，不能证明它在真实任务中有价值。
        // 统一评分入口后仍保留原有上限，避免自测覆盖度和能效把能力抬高。
        let auto_score = if self.real_call_count() == 0 && total_validations == 0 {
            auto_score.min(0.1)
        } else {
            auto_score
        };

        // ── 人类轨:有用性是进化的最终标准 ──
        // 人类反馈按样本置信度逐步接管评分。Beta(2,2) 先验避免一次误判
        // 把新能力直接锁死；样本增多后，权重渐近 0.95，真实偏好最终仍主导。
        if self.human_signals_count > 0 {
            let samples = self.human_signals_count as f64;
            let useful = self.human_score * samples;
            let posterior_mean = (useful + 2.0) / (samples + 4.0);
            let confidence = samples / (samples + 2.0);
            let human_weight = 0.95 * confidence;
            let human_direction = (posterior_mean - 0.5) * 2.0;
            let directional_strength = human_direction.abs().sqrt().copysign(human_direction);
            self.score = if human_direction >= 0.0 {
                auto_score + (1.0 - auto_score) * directional_strength * human_weight
            } else {
                auto_score + auto_score * directional_strength * human_weight
            }
            .clamp(0.0, 1.0);
        } else {
            self.score = auto_score;
        }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
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

/// P4: 持久化测试用例 — 变异后回归测试使用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// 测试输入
    pub input: serde_json::Value,
    /// 预期成功
    pub expect_success: bool,
    /// 测试来源（auto_test / real_validation / autonomous）
    pub source: String,
    /// 记录时间
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
            test_suite: Vec::new(),
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

    /// P4: 添加测试用例到持久化测试套件
    pub fn add_test_case(&mut self, input: serde_json::Value, expect_success: bool, source: &str) {
        // 避免重复（简单去重：input 序列化相同）
        let input_str = serde_json::to_string(&input).unwrap_or_default();
        if self
            .test_suite
            .iter()
            .any(|t| serde_json::to_string(&t.input).unwrap_or_default() == input_str)
        {
            return;
        }
        // 最多保留 20 个测试用例
        if self.test_suite.len() >= 20 {
            self.test_suite.remove(0);
        }
        self.test_suite.push(TestCase {
            input,
            expect_success,
            source: source.to_string(),
            timestamp: now_string(),
        });
    }

    /// 获取动作描述（给 AI 看）
    pub fn describe(&self) -> String {
        let mut desc = format!(
            "  - {} (v{}): {}\n",
            self.name, self.version, self.description
        );
        for action in &self.actions {
            desc.push_str(&format!("    · {}: {}\n", action.name, action.description));
        }
        desc
    }

    /// 记录变异
    pub fn record_mutation(
        &mut self,
        mutation_type: impl Into<String>,
        description: impl Into<String>,
    ) {
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
    llm_client: Option<Arc<dyn EvolutionDriver>>,
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
    pub fn with_llm(mut self, client: Arc<dyn EvolutionDriver>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// 绑定消息总线（使 Composite 和 Native 实现可执行）
    pub fn with_bus(mut self, bus: Arc<MessageBus>) -> Self {
        self.bus = Some(bus);
        self
    }

    /// 绑定执行器注册表（使 Custom 实现可执行 — 元进化产物）
    pub fn with_executor_registry(
        mut self,
        registry: Arc<crate::meta_evolve::ExecutorRegistry>,
    ) -> Self {
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
    async fn execute_action(
        &self,
        action: &str,
        input: &serde_json::Value,
        fitness_class: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let action_gene = self
            .genome
            .actions
            .iter()
            .find(|a| a.name == action)
            .ok_or_else(|| format!("动作 '{}' 不存在于能力 '{}'", action, self.genome.name))?;

        match &action_gene.implementation {
            ActionImpl::Llm {
                prompt,
                model,
                system,
            } => {
                let client = self
                    .llm_client
                    .as_ref()
                    .ok_or_else(|| "LLM 客户端未配置".to_string())?;

                let rendered = render_template(prompt, input);
                let system_prompt = system.as_ref().map(|s| render_template(s, input));

                let result = client
                    .execute(&rendered, model, system_prompt.as_deref())
                    .await;
                result.map(|text| {
                    // 能量记账：估算 token 消耗（prompt + response）
                    // 粗略估计：英文 ~4 chars/token，中文 ~2 chars/token，取中间值 3
                    let input_tokens = (rendered.len() as u64
                        + system_prompt.as_ref().map(|s| s.len() as u64).unwrap_or(0))
                        / 3;
                    let output_tokens = text.len() as u64 / 3;
                    let total_tokens = input_tokens + output_tokens;
                    serde_json::json!({"result": text, "_token_cost": total_tokens})
                })
            }
            ActionImpl::Rule { template } => Ok(render_template_value(template, input)),
            ActionImpl::Composite { steps } => {
                // 组合调用：按步骤编排，每步调用其他能力
                let bus = self
                    .bus
                    .as_ref()
                    .ok_or_else(|| "组合能力需要消息总线绑定".to_string())?;

                let mut step_results = serde_json::Map::new();
                let mut context = input.clone();

                for step in steps {
                    // 渲染步骤输入（支持引用前序步骤的输出）
                    let step_input = render_template_value(&step.input, &context);

                    let mut msg = Message::builder()
                        .from(&self.genome.name)
                        .to(&step.capability)
                        .action(&step.action)
                        .payload(step_input);
                    if let Some(class) = fitness_class {
                        msg = msg.metadata(crate::message::FITNESS_CLASS_METADATA, class);
                    }
                    let msg = msg.build();

                    let resp = bus.send(msg).await.map_err(|e| {
                        format!(
                            "组合步骤 '{}' 调用 {}.{} 失败: {}",
                            step.name, step.capability, step.action, e
                        )
                    })?;

                    // 将步骤结果存入上下文，供后续步骤引用
                    if let Some(obj) = context.as_object_mut() {
                        obj.insert(step.name.clone(), resp.payload.clone());
                    }
                    step_results.insert(step.name.clone(), resp.payload);
                }

                Ok(serde_json::Value::Object(step_results))
            }
            ActionImpl::Native { capability, action } => {
                // 委托给原生能力：通过消息总线转发
                let bus = self
                    .bus
                    .as_ref()
                    .ok_or_else(|| "原生委托需要消息总线绑定".to_string())?;

                let mut msg = Message::builder()
                    .from(&self.genome.name)
                    .to(capability)
                    .action(action)
                    .payload(input.clone());
                if let Some(class) = fitness_class {
                    msg = msg.metadata(crate::message::FITNESS_CLASS_METADATA, class);
                }
                let msg = msg.build();

                let resp = bus
                    .send(msg)
                    .await
                    .map_err(|e| format!("原生委托 {}.{} 失败: {}", capability, action, e))?;

                Ok(resp.payload)
            }
            ActionImpl::Script {
                language,
                code,
                timeout_secs,
            } => {
                // 脚本能力：AI 编写的代码持久化在基因组中
                // 模板渲染后写入临时文件执行
                let rendered_code = render_template(code, input);

                // P5-fix: 将 JSON 输入注入为脚本可访问的变量
                // 运行时契约：
                //   - Python: 预定义变量 `input`（dict），用 input.get('field') 访问
                //   - Node:   预定义变量 `input`（object），用 input.field 访问
                //   - 两者都同时注入 ORCH_INPUT 环境变量作为备用通道
                let input_json = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());

                let full_code = if language == "python" || language == "py" {
                    let input_prelude = format!(
                        "import json as _json\ninput = _json.loads({})",
                        python_repr(&input_json)
                    );
                    format!(
                        "{}\n{}\n\n{}",
                        CAPABILITY_BRIDGE_PY, input_prelude, rendered_code
                    )
                } else if language == "node" || language == "js" || language == "javascript" {
                    format!(
                        "const input = JSON.parse(process.env.ORCH_INPUT || '{{}}');\n{}",
                        rendered_code
                    )
                } else {
                    rendered_code
                };

                let sandbox_language = match language.as_str() {
                    "python" | "py" => "python",
                    "node" | "js" | "javascript" => "node",
                    _ => return Err(format!("不支持的脚本语言: {}", language)),
                };
                let mut config = SandboxConfig::default();
                config.timeout = std::time::Duration::from_secs((*timeout_secs).max(1));
                config.allowed_paths = capability_allowed_paths(input);
                let mut env = std::collections::HashMap::new();
                env.insert("ORCH_CAPABILITY_NAME".into(), self.genome.name.clone());
                env.insert("ORCH_CAPABILITY_ACTION".into(), action.to_string());
                env.insert("ORCH_INPUT".into(), input_json);
                if let Some(class) = fitness_class {
                    env.insert("ORCH_FITNESS_CLASS".into(), class.to_string());
                }
                if self.bus.is_some() {
                    env.insert("ORCH_BUS_AVAILABLE".into(), "1".into());
                }
                let result = Sandbox::new(config)
                    .execute_script(sandbox_language, &full_code, input, env)
                    .await;
                Ok(serde_json::json!({
                    "language": language,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "success": result.success,
                    "timed_out": result.timed_out,
                    "validation_errors": result.validation_errors,
                    "isolation": result.isolation,
                    "worker_pid": result.worker_pid,
                    "sandbox_backend": result.sandbox_backend,
                }))
            }
            ActionImpl::Shell {
                command,
                timeout_secs,
            } => {
                // Shell 也必须经过统一 Worker 和 OS 沙箱。
                let rendered = render_template(command, input);
                let mut config = SandboxConfig::default();
                config.timeout = std::time::Duration::from_secs((*timeout_secs).max(1));
                config.allowed_paths = capability_allowed_paths(input);
                let mut env = std::collections::HashMap::new();
                env.insert("ORCH_CAPABILITY_NAME".into(), self.genome.name.clone());
                env.insert("ORCH_CAPABILITY_ACTION".into(), action.to_string());
                if let Some(class) = fitness_class {
                    env.insert("ORCH_FITNESS_CLASS".into(), class.to_string());
                }
                if self.bus.is_some() {
                    env.insert("ORCH_BUS_AVAILABLE".into(), "1".into());
                }
                let result = Sandbox::new(config)
                    .execute_script("shell", &rendered, input, env)
                    .await;
                Ok(serde_json::json!({
                    "language": "shell",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "success": result.success,
                    "timed_out": result.timed_out,
                    "validation_errors": result.validation_errors,
                    "isolation": result.isolation,
                    "worker_pid": result.worker_pid,
                    "sandbox_backend": result.sandbox_backend,
                }))
            }
            ActionImpl::Custom {
                executor_type,
                params,
            } => {
                let registry = self
                    .executor_registry
                    .as_ref()
                    .ok_or_else(|| "自定义执行器注册表未配置".to_string())?;

                let context = crate::meta_evolve::ExecutorContext {
                    capability_name: self.genome.name.clone(),
                    action_name: action.to_string(),
                };

                registry
                    .execute(executor_type, params, input, &context)
                    .await
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
        self.genome
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect()
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
        let fitness_class = msg
            .metadata
            .get(crate::message::FITNESS_CLASS_METADATA)
            .map(String::as_str);
        let is_auto_test = fitness_class == Some(crate::message::FITNESS_CLASS_AUTO_TEST);

        match self
            .execute_action(&msg.action, &msg.payload, fitness_class)
            .await
        {
            Ok(result) => {
                let latency = start.elapsed().as_millis() as f64;
                tracing::info!(
                    "脚本能力 '{}' 执行 '{}' 成功 ({:.1}ms)",
                    self.genome.name,
                    msg.action,
                    latency
                );

                // 更新运行时适应度（真实业务调用）
                {
                    let mut fitness = self.runtime_fitness.write().await;
                    // 真实调用允许 Llm/Rule 等无 success 字段的 Ok 输出；自动评测则要求
                    // 明确 success，避免 runtime 记成功而评测器记协议失败的冲突。
                    let actual_success = result
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(!is_auto_test);
                    if is_auto_test {
                        fitness.record_auto_test(actual_success, latency);
                    } else {
                        fitness.record_real_call(actual_success, latency);
                    }
                    // 能量记账：LLM 调用产生的 _token_cost 记入 fitness
                    if let Some(token_cost) = result.get("_token_cost").and_then(|v| v.as_u64()) {
                        fitness.record_token_cost(token_cost);
                    }
                }

                // 剥离内部记账字段，不暴露给调用方
                let mut clean_result = result;
                if let Some(obj) = clean_result.as_object_mut() {
                    obj.remove("_token_cost");
                    // auto_test 归一化：Ok 且无显式 success:false 时注入 success:true，
                    // 使 LLM({result}) / Composite(步骤结果) 也能通过自动评测。
                    // runtime 和 test_capability 读取同一个归一化结果，消除语义分歧。
                    if is_auto_test && !obj.contains_key("success") {
                        obj.insert("success".to_string(), serde_json::Value::Bool(true));
                    }
                }

                Ok(Message::builder()
                    .from(&self.genome.name)
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action(&msg.action)
                    .payload(clean_result)
                    .build())
            }
            Err(e) => {
                tracing::warn!(
                    "脚本能力 '{}' 执行 '{}' 失败: {}",
                    self.genome.name,
                    msg.action,
                    e
                );

                // 更新运行时适应度（真实业务调用失败）
                {
                    let mut fitness = self.runtime_fitness.write().await;
                    if is_auto_test {
                        fitness.record_auto_test(false, 0.0);
                    } else {
                        fitness.record_real_call(false, 0.0);
                    }
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
/// LLM 运行时配置 — 可被 HTTP API 热切换覆盖
///
/// LlmExecutor 构造时从环境变量/参数读入配置。HTTP API 的 POST /api/config
/// 可在运行时写入覆盖层（Arc<RwLock<Option<LlmConfig>>>），execute 时优先用覆盖配置。
/// 这让"界面切 API 不重启 daemon"成为可能——切完下一次 LLM 调用即生效。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LlmRoleConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub fast_model: String,
    pub smart_model: String,
    pub coder_model: String,
    /// 可选的按角色连接配置。缺失时回退到上面的旧版共享配置字段。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fast_config: Option<LlmRoleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smart_config: Option<LlmRoleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coder_config: Option<LlmRoleConfig>,
    /// Ordered provider/model fallbacks shared by all roles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_configs: Vec<LlmRoleConfig>,
    /// API 开放时段(本地时间)。None 表示全天可用(默认,旧配置兼容)。
    /// 跨日窗口用 start>end 表示,如 23:00-09:00 表示每晚 23:00 到次日 09:00。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_hours: Option<crate::llm_health::TimeWindow>,
}

impl LlmConfig {
    /// 返回指定角色的完整连接配置；兼容旧版单供应商配置。
    pub fn role_config(&self, role: &str) -> LlmRoleConfig {
        let fallback = || LlmRoleConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            model: match role {
                "smart" => self.smart_model.clone(),
                "coder" => self.coder_model.clone(),
                _ => self.fast_model.clone(),
            },
        };
        match role {
            "smart" => self.smart_config.clone().unwrap_or_else(fallback),
            "coder" => self.coder_config.clone().unwrap_or_else(fallback),
            _ => self.fast_config.clone().unwrap_or_else(fallback),
        }
    }
}

/// LLM 调用错误分类 — 供熔断器判定是否为"不可用"信号。
#[derive(Debug)]
pub enum LlmError {
    /// 5xx — 服务端问题(含 503 model_not_found)→ 熔断器计失败
    Server(u16, String),
    /// 4xx — 客户端问题(401/403/429 配置错)→ 不计(非不可用)
    Client(u16, String),
    /// 网络层失败(连接拒绝/重置)→ 计失败
    Network(String),
    /// 超时 → 计失败
    Timeout,
    /// JSON/空响应解析失败 → 不计(非不可用,偶发)
    Parse(String),
    /// Provider reports exhausted quota/rate plan, sometimes inside HTTP 200.
    Quota(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Server(s, m) => write!(f, "OpenAI API 错误 ({}): {}", s, m),
            LlmError::Client(s, m) => write!(f, "OpenAI API 错误 ({}): {}", s, m),
            LlmError::Network(m) => write!(f, "OpenAI API 请求失败: {}", m),
            LlmError::Timeout => write!(f, "OpenAI API 调用超时"),
            LlmError::Parse(m) => write!(f, "OpenAI API 响应解析失败: {}", m),
            LlmError::Quota(m) => write!(f, "LLM 供应商额度不可用: {}", m),
        }
    }
}

impl LlmError {
    /// 是否应触发熔断器计失败(即可用性问题)。
    pub fn is_availability_failure(&self) -> bool {
        matches!(
            self,
            LlmError::Server(_, _) | LlmError::Network(_) | LlmError::Timeout | LlmError::Quota(_)
        )
    }

    fn cooldown_duration(&self) -> Option<Duration> {
        match self {
            LlmError::Quota(_) => Some(Duration::from_secs(300)),
            LlmError::Client(status, _) if matches!(status, 401 | 403 | 404 | 408 | 409 | 429) => {
                Some(Duration::from_secs(300))
            }
            LlmError::Server(_, _) | LlmError::Network(_) | LlmError::Timeout => {
                Some(Duration::from_secs(60))
            }
            LlmError::Parse(_) => Some(Duration::from_secs(60)),
            LlmError::Client(_, _) => None,
        }
    }
}

fn provider_reported_error(body: &str) -> Option<LlmError> {
    let value = serde_json::from_str::<serde_json::Value>(body).ok()?;
    if let Some(base) = value.get("base_resp") {
        let code = base
            .get("status_code")
            .and_then(|value| value.as_i64())
            .unwrap_or_default();
        if code != 0 {
            let message = base
                .get("status_msg")
                .and_then(|value| value.as_str())
                .unwrap_or("供应商返回未知错误")
                .to_string();
            let lower = message.to_ascii_lowercase();
            if message.contains("额度")
                || message.contains("用量")
                || lower.contains("token plan")
                || lower.contains("quota")
                || lower.contains("rate limit")
            {
                return Some(LlmError::Quota(message));
            }
            return Some(LlmError::Parse(format!("供应商错误 {}: {}", code, message)));
        }
    }
    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(|value| value.as_str())
            .or_else(|| error.as_str())?;
        let lower = message.to_ascii_lowercase();
        if lower.contains("quota")
            || lower.contains("rate limit")
            || lower.contains("insufficient_quota")
        {
            return Some(LlmError::Quota(message.to_string()));
        }
    }
    None
}

#[derive(Debug, Clone)]
struct ProviderCooldown {
    until: Instant,
    reason: String,
}

/// 通过 OpenAI 兼容 API 或 Anthropic API 调用 LLM
pub struct LlmExecutor {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
    /// 快速模型（量大简单任务）— 默认 ORCH_MODEL
    fast_model: String,
    /// 深度推理模型（归因分析、变异方案）
    smart_model: String,
    /// 代码生成模型（新能力代码、变异代码）
    coder_model: String,
    /// 运行时覆盖配置（HTTP 热切换写入）— None 时用上面的原始字段
    ///
    /// 用 std::sync::RwLock 而非 tokio 的：读取极频繁（每次 LLM 调用）、写入极少（人切 API），
    /// 读不阻塞；且同步锁让 resolve_model 等非 async 方法也能读，避免 block_on 嵌套 panic。
    override_config: Arc<std::sync::RwLock<Option<LlmConfig>>>,
    /// LLM API 熔断器 — 共享给 daemon 循环与 HTTP API
    breaker: Arc<crate::llm_health::LlmCircuitBreaker>,
    provider_cooldowns: Mutex<HashMap<String, ProviderCooldown>>,
    provider_timeout: Duration,
    pi_fallback_enabled: bool,
    pi_binary: String,
    pi_fallback_model: String,
    pi_fallback_timeout: Duration,
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
        let provider_timeout = Duration::from_secs(
            std::env::var("ORCH_LLM_PROVIDER_TIMEOUT_SECS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(12)
                .clamp(2, 120),
        );
        let pi_fallback_enabled = std::env::var("ORCH_LLM_PI_FALLBACK")
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "off" | "false"
                )
            })
            .unwrap_or(true);
        Self {
            api_key: api_key.into(),
            base_url: base_url_str,
            http: reqwest::Client::new(),
            fast_model: std::env::var("ORCH_MODEL_FAST")
                .or_else(|_| std::env::var("ORCH_MODEL"))
                .unwrap_or_else(|_| "MiniMax-M3".to_string()),
            smart_model: std::env::var("ORCH_MODEL_SMART")
                .unwrap_or_else(|_| "MiniMax-M3".to_string()),
            coder_model: std::env::var("ORCH_MODEL_CODER")
                .unwrap_or_else(|_| "MiniMax-M3".to_string()),
            override_config: Arc::new(std::sync::RwLock::new(None)),
            breaker: Arc::new(crate::llm_health::LlmCircuitBreaker::new()),
            provider_cooldowns: Mutex::new(HashMap::new()),
            provider_timeout,
            pi_fallback_enabled,
            pi_binary: std::env::var("ORCH_PI_BIN").unwrap_or_else(|_| "pi".into()),
            pi_fallback_model: std::env::var("ORCH_LLM_PI_FALLBACK_MODEL")
                .or_else(|_| std::env::var("ORCH_PI_FALLBACK_MODEL"))
                .unwrap_or_else(|_| "opengateway/tencent/hy3".into()),
            pi_fallback_timeout: Duration::from_secs(
                std::env::var("ORCH_LLM_PI_TIMEOUT_SECS")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(90)
                    .clamp(10, 300),
            ),
        }
    }

    /// Explicitly control the final local Pi inference fallback.
    pub fn with_pi_fallback_enabled(mut self, enabled: bool) -> Self {
        self.pi_fallback_enabled = enabled;
        self
    }

    /// 热切换覆盖配置的共享句柄 — 供 HTTP API 的 DaemonHandle 持有
    pub fn override_handle(&self) -> Arc<std::sync::RwLock<Option<LlmConfig>>> {
        self.override_config.clone()
    }

    /// 熔断器句柄 — 供 daemon 循环与 HTTP API 共享
    pub fn breaker(&self) -> Arc<crate::llm_health::LlmCircuitBreaker> {
        self.breaker.clone()
    }

    /// 写入覆盖配置（热切换）— 下一次 execute 即生效
    pub fn set_config(&self, cfg: LlmConfig) -> Result<(), String> {
        match self.override_config.write() {
            Ok(mut guard) => {
                *guard = Some(cfg);
                Ok(())
            }
            Err(e) => Err(format!("覆盖配置写锁失败: {}", e)),
        }
    }

    /// 清除覆盖配置（回退到启动时配置）
    pub fn clear_config(&self) {
        if let Ok(mut guard) = self.override_config.write() {
            *guard = None;
        }
    }

    /// 读取当前生效配置的快照（覆盖层优先，回退原始字段）— 供 GET /api/config 用
    pub fn effective_config(&self) -> LlmConfig {
        if let Ok(guard) = self.override_config.read() {
            if let Some(cfg) = guard.clone() {
                return cfg;
            }
        }
        LlmConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            fast_model: self.fast_model.clone(),
            smart_model: self.smart_model.clone(),
            coder_model: self.coder_model.clone(),
            fast_config: None,
            smart_config: None,
            coder_config: None,
            fallback_configs: Vec::new(),
            active_hours: None,
        }
    }

    /// 是否有可用的 LLM 后端（已配置 api_key）
    pub fn has_llm_backend(&self) -> bool {
        let cfg = self.effective_config();
        !cfg.api_key.is_empty()
            || !cfg.role_config("fast").api_key.is_empty()
            || !cfg.role_config("smart").api_key.is_empty()
            || !cfg.role_config("coder").api_key.is_empty()
    }

    /// 将角色选择器解析为一次请求所需的完整连接配置。
    fn request_config(cfg: &LlmConfig, selector: &str) -> LlmRoleConfig {
        if selector.starts_with("fast:") || selector == "auto" {
            return cfg.role_config("fast");
        }
        if selector.starts_with("smart:") {
            return cfg.role_config("smart");
        }
        if selector.starts_with("coder:") {
            return cfg.role_config("coder");
        }
        LlmRoleConfig {
            api_key: cfg.api_key.clone(),
            base_url: cfg.base_url.clone(),
            model: selector.to_string(),
        }
    }

    fn request_configs(cfg: &LlmConfig, selector: &str) -> Vec<LlmRoleConfig> {
        let mut candidates = vec![Self::request_config(cfg, selector)];
        let fallback_roles: &[&str] = if selector.starts_with("smart:") {
            &["coder", "fast"]
        } else if selector.starts_with("coder:") {
            &["smart", "fast"]
        } else if selector.starts_with("fast:") || selector == "auto" {
            &["smart", "coder"]
        } else {
            &[]
        };
        for role in fallback_roles {
            let candidate = cfg.role_config(role);
            if !candidates.iter().any(|existing| {
                existing.base_url == candidate.base_url
                    && existing.model == candidate.model
                    && existing.api_key == candidate.api_key
            }) {
                candidates.push(candidate);
            }
        }
        for candidate in cfg.fallback_configs.iter().take(24) {
            if !candidates.iter().any(|existing| {
                existing.base_url == candidate.base_url
                    && existing.model == candidate.model
                    && existing.api_key == candidate.api_key
            }) {
                candidates.push(candidate.clone());
            }
        }
        candidates
            .into_iter()
            .filter(|candidate| {
                !candidate.api_key.trim().is_empty()
                    && !candidate.base_url.trim().is_empty()
                    && !candidate.model.trim().is_empty()
            })
            .collect()
    }

    fn provider_key(config: &LlmRoleConfig) -> String {
        format!("{}|{}", config.base_url.trim_end_matches('/'), config.model)
    }

    fn provider_cooldown(&self, config: &LlmRoleConfig) -> Option<String> {
        let key = Self::provider_key(config);
        let now = Instant::now();
        let mut cooldowns = self.provider_cooldowns.lock().ok()?;
        cooldowns.retain(|_, cooldown| cooldown.until > now);
        cooldowns.get(&key).map(|cooldown| cooldown.reason.clone())
    }

    fn mark_provider_cooldown(&self, config: &LlmRoleConfig, error: &LlmError) {
        let Some(duration) = error.cooldown_duration() else {
            return;
        };
        if let Ok(mut cooldowns) = self.provider_cooldowns.lock() {
            cooldowns.insert(
                Self::provider_key(config),
                ProviderCooldown {
                    until: Instant::now() + duration,
                    reason: error.to_string(),
                },
            );
        }
    }

    async fn execute_pi_fallback(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        if !self.pi_fallback_enabled {
            return Err("本地 Pi 推理备用已禁用".into());
        }
        let mut command = tokio::process::Command::new(&self.pi_binary);
        command.args([
            "--no-session",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--mode",
            "text",
            "--model",
            &self.pi_fallback_model,
        ]);
        if let Some(system) = system {
            command.args(["--system-prompt", system]);
        }
        command.args(["-p", prompt]).kill_on_drop(true);
        let output = tokio::time::timeout(self.pi_fallback_timeout, command.output())
            .await
            .map_err(|_| format!("本地 Pi 推理超时 ({:?})", self.pi_fallback_timeout))?
            .map_err(|error| format!("启动本地 Pi 推理失败: {}", error))?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if output.status.success() && !stdout.is_empty() {
            return Ok(stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(format!(
            "本地 Pi 推理失败: {}",
            if stderr.is_empty() {
                "无输出"
            } else {
                &stderr
            }
        ))
    }

    pub async fn execute(
        &self,
        prompt: &str,
        model: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        // 熔断器 Open 时短路(不发请求,各调用点走现有兜底)
        if self.breaker.is_open() {
            return self
                .execute_pi_fallback(prompt, system)
                .await
                .map_err(|error| format!("LLM API 熔断中；{}", error));
        }

        // 读生效配置（覆盖层优先）— 支持 HTTP 热切换
        let cfg = self.effective_config();
        let candidates = Self::request_configs(&cfg, model);
        let mut errors = Vec::new();
        if candidates.is_empty() {
            errors.push("没有为当前角色配置可用的 LLM 连接".into());
        }
        let mut attempted = false;
        for (index, request_cfg) in candidates.iter().enumerate() {
            if let Some(reason) = self.provider_cooldown(request_cfg) {
                errors.push(format!("{} 冷却中: {}", request_cfg.model, reason));
                continue;
            }
            attempted = true;
            let result = if request_cfg.base_url.contains("anthropic") {
                tokio::time::timeout(
                    self.provider_timeout,
                    self.execute_anthropic(prompt, request_cfg, system),
                )
                .await
                .unwrap_or(Err(LlmError::Timeout))
            } else {
                tokio::time::timeout(
                    self.provider_timeout,
                    self.execute_openai(prompt, request_cfg, system),
                )
                .await
                .unwrap_or(Err(LlmError::Timeout))
            };
            match result {
                Ok(output) => {
                    self.breaker.record_success();
                    if index > 0 {
                        tracing::warn!(model = %request_cfg.model, base_url = %request_cfg.base_url, "主 LLM 不可用，已自动切换角色备用连接");
                    }
                    return Ok(output);
                }
                Err(error) => {
                    tracing::warn!(model = %request_cfg.model, base_url = %request_cfg.base_url, error = %error, "LLM 连接失败，尝试备用连接");
                    self.mark_provider_cooldown(request_cfg, &error);
                    if error.is_availability_failure() {
                        self.breaker.record_failure();
                    }
                    errors.push(format!("{}: {}", request_cfg.model, error));
                }
            }
        }
        let api_error = if !attempted {
            format!("所有 LLM 连接均在冷却中: {}", errors.join(" | "))
        } else {
            format!("LLM 连接全部失败: {}", errors.join(" | "))
        };
        match self.execute_pi_fallback(prompt, system).await {
            Ok(output) => {
                self.breaker.record_success();
                tracing::warn!(model = %self.pi_fallback_model, "所有 API 连接不可用，已切换到本地 Pi 推理备用");
                Ok(output)
            }
            Err(pi_error) => Err(format!("{}；{}", api_error, pi_error)),
        }
    }

    /// Multi-turn 对话 — 让 LLM 进行深度推理
    ///
    /// 与单次 execute 不同，这个方法支持多轮对话：
    /// 1. 发送初始 prompt
    /// 2. LLM 回复后，自动追问"请进一步分析"
    /// 3. 最后要求 LLM 给出结构化结论
    ///
    /// 适用于需要深度推理的场景：归因分析、变异方案设计、能力评估
    pub async fn execute_conversation(
        &self,
        initial_prompt: &str,
        model: &str,
        system: Option<&str>,
        follow_ups: &[&str],
    ) -> Result<String, String> {
        // 熔断器 Open 时短路(不发请求,各调用点走现有兜底)
        if self.breaker.is_open() {
            return Err("LLM API 熔断中（连续失败）".to_string());
        }

        use serde::Serialize;

        #[derive(Serialize, Clone)]
        struct Msg {
            role: String,
            content: String,
        }

        // 读生效配置（覆盖层优先）— 支持 HTTP 热切换
        let cfg = self.effective_config();
        let request_cfg = Self::request_config(&cfg, model);
        let resolved_model = &request_cfg.model;
        let api_key = &request_cfg.api_key;
        let base = request_cfg.base_url.trim_end_matches('/');

        let url: String = if let Some(path) = std::env::var("ORCH_API_PATH").ok() {
            format!("{}{}", base, path)
        } else if base.ends_with("/v1") {
            format!("{}/chat/completions", base)
        } else {
            format!("{}/v1/chat/completions", base)
        };

        let mut messages: Vec<Msg> = Vec::new();

        if let Some(sys) = system {
            messages.push(Msg {
                role: "system".into(),
                content: sys.to_string(),
            });
        }

        messages.push(Msg {
            role: "user".into(),
            content: initial_prompt.to_string(),
        });

        // 对话轮次：初始 prompt + follow_ups
        let total_rounds = 1 + follow_ups.len();

        for round in 0..total_rounds {
            let req = serde_json::json!({
                "model": &resolved_model,
                "max_tokens": 8192,
                "messages": &messages,
            });

            let mut last_err = String::new();
            let mut success = false;

            for attempt in 1..=2u32 {
                if attempt > 1 {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }

                let resp = match self
                    .http
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key.as_str()))
                    .header("content-type", "application/json")
                    .json(&req)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = format!("对话 API 请求失败: {}", e);
                        continue;
                    }
                };

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    if status.is_server_error() {
                        last_err = format!(
                            "对话 API 错误 ({}): {}",
                            status,
                            &body[..200.min(body.len())]
                        );
                        continue;
                    }
                    return Err(format!("对话 API 错误 ({}): {}", status, body));
                }

                let body_text = resp.text().await.unwrap_or_default();
                let r: serde_json::Value = match serde_json::from_str(&body_text) {
                    Ok(v) => v,
                    Err(e) => {
                        last_err = format!("对话 API 响应解析失败: {}", e);
                        continue;
                    }
                };

                let text = r
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                if text.trim().is_empty() {
                    last_err = "对话 API 返回空内容".to_string();
                    continue;
                }

                messages.push(Msg {
                    role: "assistant".into(),
                    content: text.to_string(),
                });
                success = true;
                break;
            }

            if !success {
                return Err(format!(
                    "对话 API 第 {} 轮调用失败: {}",
                    round + 1,
                    last_err
                ));
            }

            // 如果还有后续追问，添加下一轮 user 消息
            if round < follow_ups.len() {
                messages.push(Msg {
                    role: "user".into(),
                    content: follow_ups[round].to_string(),
                });
            }
        }

        // 返回最后一轮 assistant 的回复
        let result = Ok(messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| m.content.clone())
            .unwrap_or_default());

        // 更新熔断器(粗粒度:成功 record_success,失败不细分)
        // execute_conversation 内部是独立 HTTP 循环,错误已是 String 无法分类,
        // 这里保守策略:成功清零,失败不计数(避免误把 4xx 计成可用性问题)。
        // 归因失败的真实可用性问题会通过 daemon 循环的 ping 探测兜住(Task 6)。
        if result.is_ok() {
            self.breaker.record_success();
        }

        result
    }

    /// P1-2: 将 prompt 分割为可缓存的 system 前缀和可变的 user 部分
    ///
    /// 进化系统的大部分 LLM 调用有公共前缀（角色设定、输出格式说明），
    /// 将这部分提取为 system message 可以被 API 自动缓存。
    fn split_prompt_for_cache(prompt: &str) -> (Option<String>, String) {
        // 寻找 "返回严格 JSON" 或 "只返回 JSON" 作为分割点
        let split_markers = [
            "返回严格 JSON",
            "只返回 JSON",
            "请判断目标是否已达成",
            "请创造一个新能力",
        ];
        for marker in &split_markers {
            if let Some(pos) = prompt.find(marker) {
                let system_part = &prompt[..pos];
                let user_part = &prompt[pos..];
                if system_part.len() > 100 {
                    return (Some(system_part.to_string()), user_part.to_string());
                }
            }
        }
        (None, prompt.to_string())
    }

    /// OpenAI 兼容格式调用
    async fn execute_openai(
        &self,
        prompt: &str,
        request_cfg: &LlmRoleConfig,
        system: Option<&str>,
    ) -> Result<String, LlmError> {
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

        // P1-2: prompt caching — 将公共前缀提取为 system prompt
        // DeepSeek API 自动缓存相同前缀，system message 会被优先缓存
        let (system_msg, user_prompt) = if let Some(sys) = system {
            (Some(sys.to_string()), prompt.to_string())
        } else {
            // 自动提取公共前缀作为 system prompt（以 "返回严格 JSON" 为分割点）
            Self::split_prompt_for_cache(prompt)
        };

        let mut messages = vec![];
        if let Some(sys) = &system_msg {
            messages.push(OpenAiMsg {
                role: "system".into(),
                content: sys.clone(),
            });
        }
        messages.push(OpenAiMsg {
            role: "user".into(),
            content: user_prompt,
        });

        let req = OpenAiReq {
            model: request_cfg.model.clone(),
            max_tokens: 8192,
            messages,
        };

        let api_key = &request_cfg.api_key;
        let base = request_cfg.base_url.trim_end_matches('/');

        // 允许通过环境变量指定精确 API 路径
        let custom_path = std::env::var("ORCH_API_PATH").ok();
        let urls: Vec<String> = if let Some(path) = custom_path {
            vec![format!("{}{}", base, path)]
        } else if base.ends_with("/v1") {
            vec![format!("{}/chat/completions", base)]
        } else {
            vec![format!("{}/v1/chat/completions", base)]
        };

        let mut last_err: LlmError = LlmError::Network("未知错误".to_string());
        for url in &urls {
            for attempt in 1..=2u32 {
                if attempt > 1 {
                    tracing::warn!("OpenAI API 调用重试 {}/2...", attempt);
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }

                let resp = match self
                    .http
                    .post(url)
                    .header("Authorization", format!("Bearer {}", api_key.as_str()))
                    .header("content-type", "application/json")
                    .json(&req)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = LlmError::Network(format!("{}", e));
                        continue;
                    }
                };

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    if status.is_server_error() {
                        last_err = LlmError::Server(
                            status.as_u16(),
                            body[..200.min(body.len())].to_string(),
                        );
                        continue;
                    }
                    return Err(LlmError::Client(status.as_u16(), body));
                }

                let body_text = resp.text().await.unwrap_or_default();

                if let Some(error) = provider_reported_error(&body_text) {
                    return Err(error);
                }

                // P1-2: 解析 usage 统计（含缓存命中信息）
                if let Ok(usage) = serde_json::from_str::<serde_json::Value>(&body_text)
                    .map(|v| v.get("usage").cloned().unwrap_or(serde_json::json!({})))
                {
                    let prompt_tokens = usage
                        .get("prompt_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let cached = usage
                        .get("prompt_cache_hit_tokens")
                        .or_else(|| usage.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if cached > 0 {
                        tracing::info!(
                            "💾 prompt cache: {}/{} tokens cached ({:.0}%)",
                            cached,
                            prompt_tokens,
                            cached as f64 / prompt_tokens as f64 * 100.0
                        );
                    }
                }

                let r: OpenAiResp = match serde_json::from_str(&body_text) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(
                            "OpenAI API 原始响应: {}",
                            &body_text[..500.min(body_text.len())]
                        );
                        last_err = LlmError::Parse(format!("{}", e));
                        continue;
                    }
                };

                if r.choices.is_empty() {
                    return Err(LlmError::Parse("OpenAI API 返回空 choices".to_string()));
                }

                let text = r.choices[0].message.content.clone().unwrap_or_default();
                if text.trim().is_empty() {
                    last_err = LlmError::Parse("OpenAI API 返回空内容".to_string());
                    continue;
                }
                return Ok(text);
            }
        }

        Err(last_err)
    }

    /// Anthropic 格式调用
    async fn execute_anthropic(
        &self,
        prompt: &str,
        request_cfg: &LlmRoleConfig,
        system: Option<&str>,
    ) -> Result<String, LlmError> {
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
            model: request_cfg.model.clone(),
            max_tokens: 8192,
            system: system.map(|s| s.to_string()),
            messages: vec![Msg {
                role: "user".into(),
                content: prompt.to_string(),
            }],
        };

        let api_key = &request_cfg.api_key;
        let url = format!("{}/v1/messages", request_cfg.base_url.trim_end_matches('/'));

        let mut last_err: LlmError = LlmError::Network("未知".into());
        for attempt in 1..=3u32 {
            if attempt > 1 {
                tracing::warn!("Anthropic API 调用重试 {}/3...", attempt);
                tokio::time::sleep(tokio::time::Duration::from_secs(2 * attempt as u64)).await;
            }

            let resp = match self
                .http
                .post(&url)
                .header("x-api-key", api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = LlmError::Network(format!("{}", e));
                    continue;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                if status.is_server_error() {
                    last_err =
                        LlmError::Server(status.as_u16(), body[..200.min(body.len())].to_string());
                    continue;
                }
                return Err(LlmError::Client(status.as_u16(), body));
            }

            let r: LlmResp = match resp.json().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = LlmError::Parse(format!("{}", e));
                    continue;
                }
            };

            return Self::parse_response(r).map_err(LlmError::Parse);
        }

        Err(last_err)
    }

    /// 解析 LLM 响应
    fn parse_response(r: LlmResp) -> Result<String, String> {
        if r.content.is_empty() {
            return Err("LLM 返回空 content".to_string());
        }

        // 记录所有 content block 类型（调试）
        let block_types: Vec<&str> = r.content.iter().map(|c| c.ct.as_str()).collect();
        tracing::debug!("LLM content blocks: {:?}", block_types);

        let text = r
            .content
            .iter()
            .filter(|c| c.ct == "text")
            .filter_map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            tracing::warn!("LLM 返回空文本 (block types: {:?})", block_types);
            // 尝试取所有 block 的 text 字段，不管 type
            let all_text: String = r.content.iter().filter_map(|c| c.text.clone()).collect();
            if !all_text.is_empty() {
                return Ok(all_text);
            }
            // 尝试从 thinking block 提取内容作为 fallback
            let thinking_text: String = r
                .content
                .iter()
                .filter(|c| c.ct == "thinking")
                .filter_map(|c| c.thinking.clone())
                .collect::<Vec<_>>()
                .join("\n");
            if !thinking_text.is_empty() {
                tracing::warn!(
                    "使用 thinking block 内容作为 fallback ({} 字符)",
                    thinking_text.len()
                );
                return Ok(thinking_text);
            }
            return Err(format!("LLM 返回空内容 (block types: {:?})", block_types));
        }

        Ok(text)
    }
}

/// Python 能力调用桥 — 注入到每个 Python 脚本前部
///
/// 提供 `call_capability(capability, action, input)` 函数，
/// 让 Python 脚本可以在运行时调用其他已注册的能力。
/// 通过 `orch exec` CLI 命令实现跨能力通信。
const CAPABILITY_BRIDGE_PY: &str = r#"
import json, os, subprocess, sys

def call_capability(capability, action, input_data=None):
    """调用另一个能力并返回结果"""
    if input_data is None:
        input_data = {}
    # 当前 CLI 桥没有类型化调用上下文协议。自动评测中宁可显式拒绝嵌套调用，
    # 也不能让下游把探针误记为真实用户需求。
    if os.environ.get("ORCH_FITNESS_CLASS") == "auto_test":
        return {
            "error": "automated evaluation cannot invoke cross-process capability bridge",
            "success": False,
        }
    cmd = ["orch", "exec", capability, action, json.dumps(input_data)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout) if result.stdout.strip() else {}
        else:
            return {"error": result.stderr, "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

def list_capabilities():
    """列出所有可用能力"""
    cmd = ["orch", "list"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return []
    except:
        return []
"#;

/// P5: 将 JSON 字符串转为 Python 安全的字符串字面量
///
/// 用单引号包裹，转义内部反斜杠和单引号。
/// 这样 `json.loads(...)` 能安全地解析含任意字符的输入 JSON。
fn python_repr(json_str: &str) -> String {
    let escaped = json_str.replace('\\', "\\\\").replace('\'', "\\'");
    format!("'{}'", escaped)
}

fn capability_allowed_paths(input: &serde_json::Value) -> Vec<std::path::PathBuf> {
    let Some(object) = input.as_object() else {
        return Vec::new();
    };
    let mut paths = ["path", "cwd", "repo_path", "source_dir", "build_dir"]
        .iter()
        .filter_map(|key| object.get(*key).and_then(|value| value.as_str()))
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_absolute())
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    paths
}

/// 模板渲染 — 将 {{var}} 或 {{nested.path}} 替换为输入中的值
fn render_template(template: &str, input: &serde_json::Value) -> String {
    let mut result = template.to_string();

    // 支持 {{a.b.c}} 形式的嵌套路径引用
    // 用正则找到所有 {{...}} 占位符
    let re = regex::Regex::new(r"\{\{([\w.]+)\}\}").expect("静态正则表达式编译必须成功");
    for cap in re.captures_iter(template) {
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
fn render_template_value(
    template: &serde_json::Value,
    input: &serde_json::Value,
) -> serde_json::Value {
    match template {
        serde_json::Value::String(s) => serde_json::Value::String(render_template(s, input)),
        serde_json::Value::Object(map) => {
            let mut result = serde_json::Map::new();
            for (k, v) in map {
                result.insert(k.clone(), render_template_value(v, input));
            }
            serde_json::Value::Object(result)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.iter()
                .map(|v| render_template_value(v, input))
                .collect(),
        ),
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

    fn role_config(base_url: &str, model: &str) -> LlmRoleConfig {
        LlmRoleConfig {
            api_key: "test-key".into(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    fn multi_role_config(
        fast: LlmRoleConfig,
        smart: LlmRoleConfig,
        coder: LlmRoleConfig,
    ) -> LlmConfig {
        LlmConfig {
            api_key: fast.api_key.clone(),
            base_url: fast.base_url.clone(),
            fast_model: fast.model.clone(),
            smart_model: smart.model.clone(),
            coder_model: coder.model.clone(),
            fast_config: Some(fast),
            smart_config: Some(smart),
            coder_config: Some(coder),
            fallback_configs: Vec::new(),
            active_hours: None,
        }
    }

    #[test]
    fn llm_role_fallback_chain_prefers_requested_role_and_deduplicates() {
        let fast = role_config("http://fast.test/v1", "fast-model");
        let smart = role_config("http://smart.test/v1", "smart-model");
        let coder = role_config("http://coder.test/v1", "coder-model");
        let cfg = multi_role_config(fast, smart, coder);
        let candidates = LlmExecutor::request_configs(&cfg, "smart:project");
        assert_eq!(
            candidates
                .iter()
                .map(|item| item.model.as_str())
                .collect::<Vec<_>>(),
            vec!["smart-model", "coder-model", "fast-model"]
        );

        let shared = role_config("http://shared.test/v1", "shared-model");
        let duplicate_cfg = multi_role_config(shared.clone(), shared.clone(), shared);
        assert_eq!(
            LlmExecutor::request_configs(&duplicate_cfg, "smart:test").len(),
            1
        );
    }

    #[test]
    fn provider_quota_payload_is_classified_and_cooled_down() {
        let body = r#"{"base_resp":{"status_code":2056,"status_msg":"已达到 Token Plan 用量上限"},"choices":null}"#;
        let error = provider_reported_error(body).expect("应识别供应商额度错误");
        assert!(matches!(error, LlmError::Quota(_)));
        assert!(error.is_availability_failure());

        let executor = LlmExecutor::new("test-key", "http://provider.test/v1");
        let config = role_config("http://provider.test/v1", "quota-model");
        executor.mark_provider_cooldown(&config, &error);
        assert!(executor.provider_cooldown(&config).is_some());
    }

    #[tokio::test]
    async fn llm_execute_falls_back_to_another_role_after_quota_error() {
        async fn serve(body: serde_json::Value) -> String {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("绑定测试端口");
            let address = listener.local_addr().expect("读取测试端口");
            let app = axum::Router::new().fallback(move || {
                let body = body.clone();
                async move { axum::Json(body) }
            });
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            format!("http://{}/v1", address)
        }

        let quota_url = serve(serde_json::json!({
            "base_resp": {"status_code": 2062, "status_msg": "Token Plan rate limit"},
            "choices": null
        }))
        .await;
        let fallback_url = serve(serde_json::json!({
            "choices": [{"message": {"content": "fallback-ok"}}],
            "usage": {"prompt_tokens": 1}
        }))
        .await;
        let smart = role_config(&quota_url, "quota-model");
        let coder = role_config(&fallback_url, "fallback-model");
        let executor = LlmExecutor::new("test-key", &quota_url);
        executor
            .set_config(multi_role_config(coder.clone(), smart.clone(), coder))
            .expect("设置测试配置");

        let output = executor
            .execute("ping", "smart:test", None)
            .await
            .expect("备用角色应接管");
        assert_eq!(output, "fallback-ok");
        assert!(executor.provider_cooldown(&smart).is_some());
        assert!(!executor.breaker().is_open());
    }

    #[tokio::test]
    async fn llm_execute_skips_slow_provider_after_individual_timeout() {
        async fn serve_slow() -> String {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("绑定慢速测试端口");
            let address = listener.local_addr().expect("读取慢速测试端口");
            let app = axum::Router::new().fallback(|| async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                axum::Json(serde_json::json!({
                    "choices": [{"message": {"content": "too-late"}}]
                }))
            });
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            format!("http://{}/v1", address)
        }

        async fn serve_fast() -> String {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("绑定快速测试端口");
            let address = listener.local_addr().expect("读取快速测试端口");
            let app = axum::Router::new().fallback(|| async {
                axum::Json(serde_json::json!({
                    "choices": [{"message": {"content": "fast-fallback"}}]
                }))
            });
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            format!("http://{}/v1", address)
        }

        let slow_url = serve_slow().await;
        let fast_url = serve_fast().await;
        let slow = role_config(&slow_url, "slow-model");
        let fast = role_config(&fast_url, "fast-model");
        let mut executor = LlmExecutor::new("test-key", &slow_url);
        executor.provider_timeout = Duration::from_millis(50);
        executor
            .set_config(multi_role_config(fast.clone(), slow.clone(), fast))
            .expect("设置超时测试配置");

        let started = Instant::now();
        let output = executor
            .execute("ping", "smart:test", None)
            .await
            .expect("慢连接超时后备用连接应接管");
        assert_eq!(output, "fast-fallback");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(executor.provider_cooldown(&slow).is_some());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn llm_execute_uses_local_pi_after_all_api_connections_fail() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("绑定额度错误测试端口");
        let address = listener.local_addr().expect("读取额度错误测试端口");
        let app = axum::Router::new().fallback(|| async {
            axum::Json(serde_json::json!({
                "base_resp": {"status_code": 2056, "status_msg": "quota exhausted"},
                "choices": null
            }))
        });
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let url = format!("http://{}/v1", address);
        let failed = role_config(&url, "quota-model");
        let mut executor = LlmExecutor::new("test-key", &url);
        executor.pi_binary = "/bin/echo".into();
        executor.pi_fallback_model = "fake/model".into();
        executor.pi_fallback_timeout = Duration::from_secs(3);
        executor
            .set_config(multi_role_config(failed.clone(), failed.clone(), failed))
            .expect("设置 Pi 备用测试配置");

        let output = executor
            .execute("fallback-prompt", "smart:test", None)
            .await
            .expect("本地 Pi 应接管全部失败的 API 连接");
        assert!(output.contains("--no-session"));
        assert!(output.contains("fallback-prompt"));
        assert!(!executor.breaker().is_open());
    }

    #[tokio::test]
    async fn scripted_python_uses_unified_sandbox_runtime() {
        let mut genome = CapabilityGenome::new("sandbox-python", "sandbox python");
        genome.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Script {
                language: "python".into(),
                code: "import json; print(json.dumps(dict(success=True)))".into(),
                timeout_secs: 5,
            },
        });
        let output = ScriptedCapability::from_genome(genome)
            .execute_action("run", &serde_json::json!({}), None)
            .await
            .unwrap();
        assert_eq!(output["success"], true);
        if cfg!(target_os = "macos") {
            assert_eq!(output["sandbox_backend"], "macos_sandbox_exec");
        }
    }

    #[tokio::test]
    async fn scripted_shell_uses_unified_sandbox_runtime() {
        let mut genome = CapabilityGenome::new("sandbox-shell", "sandbox shell");
        genome.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Shell {
                command: "printf shell-ok".into(),
                timeout_secs: 5,
            },
        });
        let output = ScriptedCapability::from_genome(genome)
            .execute_action("run", &serde_json::json!({}), None)
            .await
            .unwrap();
        assert_eq!(output["success"], true);
        assert_eq!(output["stdout"], "shell-ok");
        if cfg!(target_os = "macos") {
            assert_eq!(output["sandbox_backend"], "macos_sandbox_exec");
        }
    }

    #[tokio::test]
    async fn composite_auto_test_context_propagates_to_dependencies() {
        let bus = Arc::new(MessageBus::new());

        let mut dependency = CapabilityGenome::new("dependency-cap", "dependency");
        dependency.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Rule {
                template: serde_json::json!({"success": true}),
            },
        });
        let dependency = Arc::new(ScriptedCapability::from_genome(dependency));
        bus.register(dependency.clone()).await;

        let mut composite = CapabilityGenome::new("composite-cap", "composite");
        composite.actions.push(ActionGene {
            name: "run".into(),
            description: "run".into(),
            input_schema: serde_json::json!({}),
            implementation: ActionImpl::Composite {
                steps: vec![CompositeStep {
                    name: "dependency".into(),
                    capability: "dependency-cap".into(),
                    action: "run".into(),
                    input: serde_json::json!({}),
                }],
            },
        });
        let composite = Arc::new(ScriptedCapability::from_genome(composite).with_bus(bus.clone()));
        bus.register(composite.clone()).await;

        let probe = Message::builder()
            .from("test")
            .to("composite-cap")
            .action("run")
            .payload(serde_json::json!({}))
            .metadata(
                crate::message::FITNESS_CLASS_METADATA,
                crate::message::FITNESS_CLASS_AUTO_TEST,
            )
            .build();
        bus.send(probe).await.unwrap();

        let parent_fitness = composite.runtime_fitness().await;
        let dependency_fitness = dependency.runtime_fitness().await;
        assert_eq!(parent_fitness.auto_test_count, 1);
        assert_eq!(parent_fitness.real_call_count(), 0);
        assert_eq!(dependency_fitness.auto_test_count, 1);
        assert_eq!(dependency_fitness.real_call_count(), 0);
    }

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
        f1.record_auto_test(true, 10.0); // 快速通过

        let mut f2 = FitnessGene::default();
        f2.record_real_call(true, 10.0); // 同样快速通过

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

    /// 验证 token 成本追踪
    #[test]
    fn test_token_cost_tracking() {
        let mut f = FitnessGene::default();
        assert_eq!(f.total_token_cost, 0);
        assert_eq!(f.last_token_cost, 0);

        f.record_token_cost(500);
        assert_eq!(f.total_token_cost, 500);
        assert_eq!(f.last_token_cost, 500);

        f.record_token_cost(300);
        assert_eq!(f.total_token_cost, 800);
        assert_eq!(f.last_token_cost, 300);
    }

    /// 验证利润率计算 — LLM 推理 vs 直接执行
    #[test]
    fn test_profit_ratio_llm_vs_script() {
        // LLM 能力：每次调用都花 token
        let mut llm_fitness = FitnessGene::default();
        for _ in 0..10 {
            llm_fitness.record_real_call(true, 200.0);
            llm_fitness.record_token_cost(800); // 每次花 800 token
        }
        // 利润率 = success_count / (token_cost + call_count)
        // = 10 / (8000 + 10) = 10/8010 ≈ 0.00125
        assert!(llm_fitness.profit_ratio < 0.01, "LLM 能力利润率应该很低");

        // Script 能力：不花 token
        let mut script_fitness = FitnessGene::default();
        for _ in 0..10 {
            script_fitness.record_real_call(true, 5.0); // 快速执行
                                                        // 不调用 record_token_cost — token_cost 始终为 0
        }
        // 利润率 = 10 / (0 + 10) = 1.0
        assert!(
            script_fitness.profit_ratio > 0.9,
            "Script 能力利润率应该接近 1.0, got {}",
            script_fitness.profit_ratio
        );

        // Script 利润率远高于 LLM
        assert!(
            script_fitness.profit_ratio > llm_fitness.profit_ratio * 100.0,
            "直接执行利润率应比 LLM 推理高 100 倍以上"
        );
    }

    /// 验证结晶化后利润率飙升
    #[test]
    fn test_crystallize_profit_boost() {
        let mut f = FitnessGene::default();

        // 阶段 1：作为 LLM 能力运行，烧 token
        for _ in 0..5 {
            f.record_real_call(true, 200.0);
            f.record_token_cost(1000);
        }
        let llm_profit = f.profit_ratio;
        assert!(llm_profit < 0.01);

        // 阶段 2：结晶化后，不再花 token
        // total_token_cost 保持 5000 不变，但后续调用不增加 token
        for _ in 0..20 {
            f.record_real_call(true, 5.0);
            // 不调用 record_token_cost
        }
        let crystallized_profit = f.profit_ratio;

        // 利润率应该显著提升
        // 结晶化后 token_cost 不再增长，利润率随调用次数提升而持续改善
        // 初始: 5/(5000+5) ≈ 0.001, 结晶化后 25/(5000+25) ≈ 0.005
        // 随着更多无 token 调用累积，利润率趋近 success_rate
        assert!(
            crystallized_profit > llm_profit * 3.0,
            "结晶化后利润率应显著提升: {} -> {}",
            llm_profit,
            crystallized_profit
        );

        // 继续累积调用，利润率持续逼近 success_rate
        for _ in 0..100 {
            f.record_real_call(true, 5.0);
        }
        let long_term_profit = f.profit_ratio;
        assert!(
            long_term_profit > crystallized_profit,
            "结晶化后利润率应随调用持续改善: {} -> {}",
            crystallized_profit,
            long_term_profit
        );
    }

    /// 验证能量因子影响适应度分数
    #[test]
    fn test_energy_factor_in_score() {
        // 两个能力：相同成功率和延迟，但一个花 token 一个不花
        let mut llm_fit = FitnessGene::default();
        let mut script_fit = FitnessGene::default();

        for _ in 0..5 {
            llm_fit.record_real_call(true, 100.0);
            llm_fit.record_token_cost(500);
            script_fit.record_real_call(true, 100.0);
        }

        llm_fit.recompute_score();
        script_fit.recompute_score();

        assert!(
            script_fit.score > llm_fit.score,
            "不花 token 的能力应该有更高适应度: script={} llm={}",
            script_fit.score,
            llm_fit.score
        );
    }
}
