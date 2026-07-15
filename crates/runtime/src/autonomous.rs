use crate::driver::EvolutionDriver;
use crate::evolution::EvolutionEngine;
use crate::genome::{CapabilityGenome, ScriptedCapability};
use crate::message_bus::MessageBus;
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 环境状态报告
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentReport {
    pub disk_usage_pct: f64,
    pub cpu_usage_pct: f64,
    pub memory_usage_pct: f64,
    pub uptime_secs: u64,
    pub recent_files: Vec<String>,
    pub git_status: Option<String>,
    pub running_processes: Vec<String>,
    pub network_listening: Vec<u16>,
    pub log_anomalies: Vec<String>,
    pub timestamp: u64,
    /// 网络连通性检测结果
    #[serde(default)]
    pub network_connectivity: Vec<NetworkProbe>,
    /// API 端点健康状态
    #[serde(default)]
    pub api_health: Vec<ApiHealthStatus>,
    /// 数据库连接状态
    #[serde(default)]
    pub database_status: Vec<DatabaseStatus>,
    /// 环境变量中的关键配置
    #[serde(default)]
    pub env_config: EnvConfigSnapshot,
    /// Docker 容器状态（如有）
    #[serde(default)]
    pub docker_containers: Vec<String>,
}

/// 网络探测结果
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkProbe {
    pub target: String,
    pub reachable: bool,
    pub latency_ms: u64,
}

/// API 健康状态
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApiHealthStatus {
    pub url: String,
    pub status: String,
    pub response_time_ms: u64,
}

/// 数据库连接状态
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatabaseStatus {
    pub db_type: String,
    pub connected: bool,
    pub details: String,
}

/// 环境配置快照
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvConfigSnapshot {
    pub has_api_key: bool,
    pub has_database_url: bool,
    pub has_redis_url: bool,
    pub rust_target: Option<String>,
    pub python_version: Option<String>,
    pub node_version: Option<String>,
}

/// 自主目标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousGoal {
    pub description: String,
    pub priority: GoalPriority,
    pub suggested_capabilities: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalPriority {
    Critical,
    High,
    Medium,
    Low,
    Exploratory,
}

impl std::fmt::Display for GoalPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GoalPriority::Critical => write!(f, "Critical"),
            GoalPriority::High => write!(f, "High"),
            GoalPriority::Medium => write!(f, "Medium"),
            GoalPriority::Low => write!(f, "Low"),
            GoalPriority::Exploratory => write!(f, "Exploratory"),
        }
    }
}

/// 自主执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousResult {
    pub goal: AutonomousGoal,
    pub success: bool,
    pub output: serde_json::Value,
    pub error: Option<String>,
    pub elapsed_ms: u64,
    /// LLM-derived usefulness feedback from the closed loop. Kept separate
    /// from execution success so the daemon can merge it from a snapshot.
    #[serde(default)]
    pub value_feedback: Option<bool>,
}

/// 外部反馈追踪 — 行为后果记录
///
/// 记录能力执行后环境的变化，形成闭环学习：
/// 执行前环境状态 → 执行能力 → 执行后环境状态 → 差异分析 → 反馈评分
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeedbackRecord {
    /// 关联的自主目标
    pub goal: String,
    /// 使用的能力
    pub capability: String,
    /// 执行前环境快照（关键字段）
    pub env_before: serde_json::Value,
    /// 执行后环境快照（关键字段）
    pub env_after: serde_json::Value,
    /// 检测到的环境变化
    pub env_changes: Vec<String>,
    /// 执行是否成功
    pub success: bool,
    /// 反馈评分（-1.0 ~ 1.0，正值=正面影响，负值=负面影响）
    pub impact_score: f64,
    /// 时间戳
    pub timestamp: u64,
}

/// 自主运行时 — 从"被动器官"到"主动生物体"
///
/// 三层自主性：
/// 1. 环境感知 — 主动观测环境状态变化
/// 2. 目标生成 — 基于环境+能力+记忆，自主决定"该做什么"
/// 3. 主动实验 — 用新能力在真实环境中尝试
pub struct AutonomousRuntime {
    llm: Arc<dyn EvolutionDriver>,
    bus: Arc<MessageBus>,
    #[allow(dead_code)]
    platform: Platform,
    history: Vec<AutonomousResult>,
    max_history: usize,
    /// 外部反馈追踪记录
    feedback_records: Vec<FeedbackRecord>,
}

impl AutonomousRuntime {
    pub fn new(llm: Arc<dyn EvolutionDriver>, bus: Arc<MessageBus>, platform: Platform) -> Self {
        Self {
            llm,
            bus,
            platform,
            history: Vec::new(),
            max_history: 100,
            feedback_records: Vec::new(),
        }
    }

    /// 1. 环境感知 — 通过已有能力观测环境
    pub async fn perceive(&self) -> EnvironmentReport {
        let mut report = EnvironmentReport {
            timestamp: now_secs(),
            ..Default::default()
        };

        // 磁盘使用率
        if let Ok(out) = tokio::process::Command::new("df")
            .args(["-h", "/"])
            .output()
            .await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            for line in s.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    if let Ok(pct) = parts[4].trim_end_matches('%').parse::<f64>() {
                        report.disk_usage_pct = pct;
                    }
                }
            }
        }

        // CPU/内存 — 用 vmstat 或 top
        if let Ok(out) = tokio::process::Command::new("vm_stat").output().await {
            let s = String::from_utf8_lossy(&out.stdout);
            // 简单解析
            for line in s.lines() {
                if line.contains("Pages free") || line.contains("Pages active") {
                    // macOS vm_stat 格式
                }
            }
        }

        // 运行进程
        if let Ok(out) = tokio::process::Command::new("ps")
            .args(["aux", "--sort=-%cpu"])
            .output()
            .await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            for line in s.lines().skip(1).take(10) {
                let name = line.split_whitespace().nth(10).unwrap_or("").to_string();
                if !name.is_empty() {
                    report.running_processes.push(name);
                }
            }
        }

        // 监听端口
        if let Ok(out) = tokio::process::Command::new("lsof")
            .args(["-iTCP", "-sTCP:LISTEN", "-P", "-n"])
            .output()
            .await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            for line in s.lines().skip(1) {
                if let Some(port) = line.split_whitespace().nth(8) {
                    if let Some(p) = port.rsplit(':').next() {
                        if let Ok(n) = p.parse::<u16>() {
                            if !report.network_listening.contains(&n) {
                                report.network_listening.push(n);
                            }
                        }
                    }
                }
            }
        }

        // Git 状态
        if let Ok(out) = tokio::process::Command::new("git")
            .args(["status", "--short"])
            .output()
            .await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            if !s.trim().is_empty() {
                report.git_status = Some(s.lines().take(20).collect::<Vec<_>>().join("\n"));
            }
        }

        // 最近修改的文件
        if let Ok(out) = tokio::process::Command::new("find")
            .args([
                ".", "-name", "*.rs", "-o", "-name", "*.py", "-o", "-name", "*.ts",
            ])
            .arg("-mtime")
            .arg("-1")
            .output()
            .await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            report.recent_files = s.lines().take(20).map(String::from).collect();
        }

        // 网络连通性探测 — 检测关键端点
        let probe_targets = ["8.8.8.8", "github.com", "api.iamhc.cn"];
        for target in &probe_targets {
            let start = std::time::Instant::now();
            let reachable = tokio::process::Command::new("ping")
                .args(["-c", "1", "-W", "2", target])
                .output()
                .await
                .map(|o| o.status.success())
                .unwrap_or(false);
            report.network_connectivity.push(NetworkProbe {
                target: target.to_string(),
                reachable,
                latency_ms: start.elapsed().as_millis() as u64,
            });
        }

        // API 健康检测 — 检测 LLM API 端点
        if let Ok(api_base) = std::env::var("ORCH_API_BASE") {
            let base = api_base.trim_end_matches('/');
            let health_url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            let start = std::time::Instant::now();
            if let Ok(api_key) = std::env::var("ORCH_API_KEY") {
                let client = reqwest::Client::new();
                let resp = client
                    .get(&health_url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .timeout(std::time::Duration::from_secs(5))
                    .send()
                    .await;
                let (status, rt) = match resp {
                    Ok(r) => (
                        format!("{}", r.status()),
                        start.elapsed().as_millis() as u64,
                    ),
                    Err(e) => (format!("error: {}", e), start.elapsed().as_millis() as u64),
                };
                report.api_health.push(ApiHealthStatus {
                    url: health_url,
                    status,
                    response_time_ms: rt,
                });
            }
        }

        // 数据库连接检测
        if let Ok(db_url) = std::env::var("DATABASE_URL").or_else(|_| std::env::var("POSTGRES_URL"))
        {
            let db_type = if db_url.starts_with("postgres") {
                "postgresql"
            } else if db_url.starts_with("mysql") {
                "mysql"
            } else if db_url.starts_with("sqlite") {
                "sqlite"
            } else {
                "unknown"
            };
            report.database_status.push(DatabaseStatus {
                db_type: db_type.to_string(),
                connected: false, // 仅检测是否存在配置
                details: format!("DATABASE_URL 已配置 (类型: {})", db_type),
            });
        }
        if let Ok(_) = std::env::var("REDIS_URL") {
            report.database_status.push(DatabaseStatus {
                db_type: "redis".to_string(),
                connected: false,
                details: "REDIS_URL 已配置".to_string(),
            });
        }

        // 环境配置快照
        let env_config = EnvConfigSnapshot {
            has_api_key: std::env::var("ORCH_API_KEY").is_ok(),
            has_database_url: std::env::var("DATABASE_URL").is_ok(),
            has_redis_url: std::env::var("REDIS_URL").is_ok(),
            rust_target: std::env::var("RUST_TARGET").ok().or_else(|| {
                std::process::Command::new("rustc")
                    .arg("--version")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
            }),
            python_version: std::process::Command::new("python3")
                .arg("--version")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string()),
            node_version: std::process::Command::new("node")
                .arg("--version")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string()),
        };
        report.env_config = env_config;

        // Docker 容器状态
        if let Ok(out) = tokio::process::Command::new("docker")
            .args(["ps", "--format", "{{.Names}}: {{.Status}}"])
            .output()
            .await
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                report.docker_containers = s.lines().take(20).map(String::from).collect();
            }
        }

        report
    }

    /// 2. 自主目标生成 — LLM 根据环境+能力+记忆+思维链生成"该做的事"
    ///
    /// 返回 (目标列表, LLM原始响应) — 原始响应用于记录思维链
    pub async fn generate_goals(
        &self,
        report: &EnvironmentReport,
        evolution: &crate::evolution::EvolutionEngine,
        memory: Option<&crate::evolution::EvolutionMemory>,
    ) -> (Vec<AutonomousGoal>, String) {
        let report_json = serde_json::to_string_pretty(report).unwrap_or_default();

        // P4a-fix: 注入能力得分+描述，标记零分能力引导 LLM 施压
        let genomes = evolution.genomes();
        let caps_str = genomes
            .iter()
            .map(|(name, g)| {
                let score = g.fitness.score;
                let sr = g.fitness.success_rate;
                let calls = g.fitness.call_count;
                let real_calls = g.fitness.real_call_count();
                let desc = &g.description;
                if score < 0.05 && calls < 5 {
                    format!(
                        "⚠️ {} (score={:.2} calls={} real={} 需要真实调用验证): {}",
                        name, score, calls, real_calls, desc
                    )
                } else {
                    format!(
                        "✓ {} (score={:.2} sr={:.0}%): {}",
                        name,
                        score,
                        sr * 100.0,
                        desc
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n  ");

        // 构建记忆上下文
        let memory_context = if let Some(mem) = memory {
            let mut parts = Vec::new();

            // === 思维链注入：让 LLM 看到上次的思考过程 ===
            let recent_chains: Vec<&crate::evolution::ThoughtChain> = mem
                .thought_chains
                .iter()
                .filter(|c| c.chain_type == "goal_generation")
                .rev()
                .take(3)
                .collect();
            if !recent_chains.is_empty() {
                let chain_text: Vec<String> = recent_chains
                    .iter()
                    .map(|c| {
                        format!(
                            "  [{}] 思考: {} → 结论: {}{}",
                            if c.success { "✅" } else { "❌" },
                            safe_truncate(&c.reasoning, 300),
                            safe_truncate(&c.conclusion, 200),
                            if let Some(g) = &c.related_goal {
                                format!(" (目标: {})", g)
                            } else {
                                String::new()
                            }
                        )
                    })
                    .collect();
                parts.push(format!(
                    "上次思考过程（请从这里继续思考，不要从零开始）:\n{}",
                    chain_text.join("\n")
                ));
            }

            // === 联想网络注入：触发相关概念联想 ===
            // 从最近思维链的关联能力出发，查找联想
            let recent_caps: Vec<String> = recent_chains
                .iter()
                .flat_map(|c| c.related_capabilities.iter().cloned())
                .collect();
            if !recent_caps.is_empty() {
                let mut associations = Vec::new();
                for cap in &recent_caps {
                    for link in mem.association_graph.iter().filter(|a| {
                        (a.from_concept == *cap || a.to_concept == *cap) && a.strength > 0.4
                    }) {
                        let other = if link.from_concept == *cap {
                            &link.to_concept
                        } else {
                            &link.from_concept
                        };
                        if !recent_caps.contains(other) && !associations.contains(other) {
                            associations.push(other.clone());
                        }
                    }
                }
                if !associations.is_empty() {
                    parts.push(format!(
                        "联想提示: 上次涉及的能力可能与这些概念有关: {}",
                        associations.join(", ")
                    ));
                }
            }

            // 近期自主目标历史（避免重复）
            let recent_goals: Vec<String> = mem
                .autonomous_history
                .iter()
                .rev()
                .take(10)
                .map(|h| {
                    format!(
                        "  - [{}] {} ({})",
                        if h.success { "✅" } else { "❌" },
                        h.goal,
                        if h.success { "成功" } else { "失败" }
                    )
                })
                .collect();
            if !recent_goals.is_empty() {
                parts.push(format!(
                    "近期自主目标历史（避免重复）:\n{}",
                    recent_goals.join("\n")
                ));
            }

            // 进化教训
            let lessons: Vec<String> = mem
                .lessons
                .iter()
                .rev()
                .take(5)
                .map(|l| format!("  - {}", l.lesson))
                .collect();
            if !lessons.is_empty() {
                parts.push(format!("进化教训:\n{}", lessons.join("\n")));
            }

            // 全局统计
            let stats = &mem.global_stats;
            if stats.total_rounds > 0 {
                parts.push(format!(
                    "系统统计: 已运行 {} 轮, 创造 {} 个能力, 淘汰 {} 个, 变异成功率 {:.0}%",
                    stats.total_rounds,
                    stats.total_created,
                    stats.total_eliminated,
                    if stats.total_mutations > 0 {
                        stats.total_mutation_successes as f64 / stats.total_mutations as f64 * 100.0
                    } else {
                        0.0
                    }
                ));
            }

            if parts.is_empty() {
                String::new()
            } else {
                format!("\n{}\n", parts.join("\n\n"))
            }
        } else {
            String::new()
        };

        let prompt = format!(
            r#"你是自主运行时的目标生成器。你具有连续性思维——你会记住上次的思考过程，并从中断处继续。

当前环境状态：
{}

可用能力：{}
{}

请根据环境状态和可用能力，生成 1-5 个"值得做的事"。

目标生成导向 — 你的核心使命是让能力库更强、更全、更有用，而非维护系统:

优先级规则:
- 组合现有能力解决新问题 → High
  例: 用 jq_ops + sqlite3_ops 组合出数据清洗流水线
  例: 用 cargo_ops + git_ops 组合出发布自动化
- 创造系统缺失的能力填补缺口 → High
  例: 发现没有网络抓包能力 → 提议创建 tcpdump_ops
  例: 发现没有定时任务能力 → 提议创建 cron_ops
- 验证零分能力(⚠️)的真实价值 → Medium
  对 score<0.05 的能力安排真实调用验证
- 环境维护(磁盘/提交/日志) → 仅在真正 Critical 时
  磁盘>90% 才生成清理目标；git 提交不是自主目标

硬性要求:
- 每个目标必须指定 suggested_capabilities (至少1个现有能力名)
- 不要生成 "提交变更"、"清理临时文件"、"检查日志" 这类不调用任何能力的维护目标
- 除非磁盘>90%，否则不生成环境维护类目标

思维连续性要求：
- 如果有"上次思考过程"，请从中断处继续推理，不要从零开始
- 如果上次某个目标失败了，思考失败的原因，并生成后续目标来解决根本问题
- 如果联想提示中提到了相关概念，请考虑是否值得探索
- 避免与近期自主目标历史重复，参考进化教训避免重蹈覆辙

返回 JSON 数组：
[
  {{
    "description": "具体要做的事",
    "priority": "High",
    "suggested_capabilities": ["能力名"],
    "reason": "为什么需要做这件事（包含与上次思考的关联）"
  }}
]
只返回 JSON。"#,
            report_json, caps_str, memory_context
        );

        let response = match tokio::time::timeout(
            std::time::Duration::from_secs(60),
            self.llm.execute(&prompt, "fast:goals", None),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                tracing::warn!("自主目标生成: LLM 调用失败: {}，回退到规则目标", e);
                return (Self::fallback_goals(report, evolution), String::new());
            }
            Err(_) => {
                tracing::warn!("自主目标生成: LLM 调用超时 (60s)，回退到规则目标");
                return (Self::fallback_goals(report, evolution), String::new());
            }
        };

        if let Ok(goals) = serde_json::from_str::<Vec<AutonomousGoal>>(&response) {
            let filtered = Self::filter_goals(goals, memory);
            return (filtered, response);
        }
        // 尝试提取 JSON 数组
        if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                if let Ok(goals) =
                    serde_json::from_str::<Vec<AutonomousGoal>>(&response[start..=end])
                {
                    let filtered = Self::filter_goals(goals, memory);
                    return (filtered, response);
                }
            }
        }
        (vec![], response)
    }

    /// 目标过滤 — 硬性去重 + 维护类黑名单 + caps=[] 过滤
    ///
    /// 弥补 prompt 软提示的不足：即使 LLM 忽略了"避免重复"和"不要生成维护目标"的指令，
    /// 这里也会硬性拦截，确保自主循环不会陷入低价值重复。
    fn filter_goals(
        goals: Vec<AutonomousGoal>,
        memory: Option<&crate::evolution::EvolutionMemory>,
    ) -> Vec<AutonomousGoal> {
        // 收集最近 20 条历史目标描述用于去重
        let recent_descs: Vec<String> = memory
            .map(|m| {
                m.autonomous_history
                    .iter()
                    .rev()
                    .take(20)
                    .map(|h| h.goal.clone())
                    .collect()
            })
            .unwrap_or_default();

        let original = goals.len();
        let result: Vec<AutonomousGoal> = goals
            .into_iter()
            .filter(|g| {
                // Critical 环境紧急目标可以无能力（如磁盘>90%清理）
                if matches!(g.priority, GoalPriority::Critical) {
                    return true;
                }
                // 非 Critical 目标必须指定能力（过滤掉"提交变更"等空 caps 目标）
                if g.suggested_capabilities.is_empty() {
                    tracing::warn!(
                        "目标过滤: 丢弃无能力目标 [{}] — caps 为空",
                        safe_truncate(&g.description, 50)
                    );
                    return false;
                }
                // 硬性去重：与近期历史高度相似的目标直接丢弃
                if is_duplicate_goal(&g.description, &recent_descs) {
                    tracing::warn!(
                        "目标过滤: 丢弃重复目标 [{}]",
                        safe_truncate(&g.description, 50)
                    );
                    return false;
                }
                true
            })
            .collect();

        if result.len() < original {
            tracing::info!(
                "目标过滤: {} → {} (丢弃 {} 个低价值/重复目标)",
                original,
                result.len(),
                original - result.len()
            );
        }
        result
    }

    /// LLM 不可用（失败/超时）时的规则兜底目标生成
    /// 基于环境报告 + 现有能力推导能力导向目标，保证自主循环在 LLM 缺失时仍能运转。
    /// 不再生成"提交变更"/"清理临时文件"等不调用能力的维护目标。
    fn fallback_goals(
        report: &EnvironmentReport,
        evolution: &crate::evolution::EvolutionEngine,
    ) -> Vec<AutonomousGoal> {
        let mut goals = Vec::new();

        // 仅磁盘真正告急才生成清理目标（Critical，允许 caps 为空）
        if report.disk_usage_pct > 90.0 {
            goals.push(AutonomousGoal {
                description: format!("清理磁盘空间（当前使用 {:.0}%）", report.disk_usage_pct),
                priority: GoalPriority::Critical,
                suggested_capabilities: vec![],
                reason: "磁盘使用率超过 90%，需要释放空间避免系统异常".into(),
            });
        }

        // 兜底：从现有能力中选低分/未验证的，生成验证目标
        let genomes = evolution.genomes();
        let candidates: Vec<(&String, &crate::genome::CapabilityGenome)> = genomes
            .iter()
            .filter(|(_, g)| g.fitness.score < 0.3 && g.fitness.call_count < 10)
            .take(3)
            .collect();
        for (name, g) in candidates {
            goals.push(AutonomousGoal {
                description: format!("验证能力 {} 的真实价值", name),
                priority: GoalPriority::Medium,
                suggested_capabilities: vec![name.clone()],
                reason: format!(
                    "该能力 score={:.2} 调用次数少，需要真实验证其价值",
                    g.fitness.score
                ),
            });
        }

        // 如果没有候选，生成能力组合探索目标
        if goals.is_empty() {
            let explore_caps: Vec<String> = genomes.keys().take(2).cloned().collect();
            goals.push(AutonomousGoal {
                description: "探索能力组合: 选两个互补能力尝试串联调用".into(),
                priority: GoalPriority::Exploratory,
                suggested_capabilities: explore_caps,
                reason: "LLM 不可用，基于规则生成能力探索目标以保持自主循环运转".into(),
            });
        }

        goals
    }

    /// 3. 执行自主目标
    pub async fn execute_goal(
        &mut self,
        goal: &AutonomousGoal,
        evolution: &mut EvolutionEngine,
    ) -> AutonomousResult {
        let start = std::time::Instant::now();

        // 如果有建议能力，直接调用
        if !goal.suggested_capabilities.is_empty() {
            let cap_name = &goal.suggested_capabilities[0];

            // 获取能力的第一个 action
            if let Some(genome) = evolution.genomes().get(cap_name) {
                if !genome.actions.is_empty() {
                    let action = &genome.actions[0];
                    let input = self.generate_input_for(genome).await;

                    let msg = crate::message::Message::builder()
                        .from("autonomous")
                        .to(cap_name)
                        .action(&action.name)
                        .payload(input)
                        .build();

                    match self.bus.send(msg).await {
                        Ok(resp) => {
                            let success = resp
                                .payload
                                .get("success")
                                .and_then(|s| s.as_bool())
                                .unwrap_or(false);
                            let error = if success {
                                None
                            } else {
                                resp.payload
                                    .get("error")
                                    .and_then(|e| e.as_str())
                                    .map(String::from)
                            };
                            let elapsed = start.elapsed().as_millis() as u64;

                            // A1: 自主循环升级为真实压力源
                            //
                            // 原设计用 record_auto_test（自测试口径），理由是"自循环没有外部真实压力"。
                            // 但这导致自主进化无法真正驱动 fitness — 自测试只给 0.1 基础分，
                            // 不清零 dormant，能力永远不会被真实压力淘汰或提升。
                            //
                            // 现在改为 record_real_call（真实调用口径），并用 LLM 自主判定价值
                            // （见 autonomous_cycle 中的 auto_evaluate + record_human_signal）替代
                            // 人类反馈。这样系统完全自主：LLM 生成目标→执行→自主判定价值→驱动进化。
                            if let Some(g) = evolution.genomes_mut().get_mut(cap_name) {
                                g.fitness.record_real_call(success, elapsed as f64);
                            }

                            let result = AutonomousResult {
                                goal: goal.clone(),
                                success,
                                output: resp.payload,
                                error,
                                elapsed_ms: elapsed,
                                value_feedback: None,
                            };
                            self.add_history(result.clone());
                            return result;
                        }
                        Err(e) => {
                            let elapsed = start.elapsed().as_millis() as u64;

                            // A1: 失败也记为真实调用（真实业务中失败也是信号）
                            if let Some(g) = evolution.genomes_mut().get_mut(cap_name) {
                                g.fitness.record_real_call(false, elapsed as f64);
                            }

                            let result = AutonomousResult {
                                goal: goal.clone(),
                                success: false,
                                output: serde_json::json!({}),
                                error: Some(e.to_string()),
                                elapsed_ms: start.elapsed().as_millis() as u64,
                                value_feedback: None,
                            };
                            self.add_history(result.clone());
                            return result;
                        }
                    }
                }
            }
        }

        // 没有匹配能力，用 LLM 规划执行
        let result = AutonomousResult {
            goal: goal.clone(),
            success: false,
            output: serde_json::json!({}),
            error: Some("没有匹配的可用能力".into()),
            elapsed_ms: start.elapsed().as_millis() as u64,
            value_feedback: None,
        };
        self.add_history(result.clone());
        result
    }

    /// A1: LLM 自主价值判定 — 替代人类 👍/👎 反馈
    ///
    /// 执行目标后，调 LLM 判定结果是否对"一个真实用户"有价值。
    /// 这是去人类化的核心：LLM 自主生成目标→自主执行→自主判定价值→自主驱动进化。
    ///
    /// 判定标准（注入 prompt）：
    /// - 输出是否与目标相关且有意义（非空、非错误）
    /// - 是否真正解决了目标描述的问题
    /// - 对真实用户来说是否有用
    ///
    /// 返回 Some(true)=有价值（等价人类 👍），Some(false)=无价值（等价人类 👎）。
    /// 返回 None 表示跳过评分（LLM 不可用 / 返回无法解析），调用方不应据此记分，
    /// 避免在 LLM 故障期间把能力 fitness 永久压低（防污染）。
    async fn auto_evaluate(
        &self,
        goal: &AutonomousGoal,
        result: &AutonomousResult,
        env_changes: &[String],
    ) -> Option<bool> {
        let output_str = match &result.output {
            serde_json::Value::String(s) => s.clone(),
            v => v.to_string(),
        };

        let prompt = format!(
            r#"你是一个能力价值评估器。以下目标被执行后产生了结果，请判定这个能力是否提供了有价值的服务。

目标: {goal_desc}
目标理由: {goal_reason}
执行成功: {success}
错误信息: {error}
执行输出 (截断): {output}
环境变化: {changes}

判定标准:
- 输出是否与目标相关且有意义（非空、非纯错误信息）
- 是否真正解决了目标描述的问题（而非定义了函数却没调用、或空输出）
- 对一个真实用户来说，这个执行结果是否有用
- 环境是否有符合预期的正面变化

只回答 true 或 false，不要其他文字。"#,
            goal_desc = goal.description,
            goal_reason = goal.reason,
            success = if result.success { "是" } else { "否" },
            error = result.error.as_deref().unwrap_or("无"),
            output = safe_truncate(&output_str, 500),
            changes = if env_changes.is_empty() {
                "无变化".to_string()
            } else {
                env_changes.join("; ")
            },
        );

        match self.llm.execute(&prompt, "fast:autofeedback", None).await {
            Ok(resp) => {
                let r = resp.trim().to_lowercase();
                // 明确匹配 true/false，避免 "not true" 之类的误判
                if r.contains("false") {
                    Some(false)
                } else if r.contains("true") {
                    Some(true)
                } else {
                    // 无法解析也跳过（不误压分）
                    tracing::warn!(
                        "auto_evaluate: LLM 返回无法解析: {}",
                        safe_truncate(&resp, 100)
                    );
                    None
                }
            }
            Err(e) => {
                tracing::warn!("auto_evaluate: LLM 调用失败: {}", e);
                None // 关键防污染：失败不记分
            }
        }
    }

    /// 4. 主动实验 — 用新能力在真实环境中尝试
    pub async fn experiment_with_new_capability(
        &mut self,
        genome: &CapabilityGenome,
        env: &EnvironmentReport,
    ) -> AutonomousResult {
        let start = std::time::Instant::now();

        // 让 LLM 根据环境状态生成一个真实的测试场景
        let env_json = serde_json::to_string(env).unwrap_or_default();
        let prompt = format!(
            r#"你是一个能力的测试员。有一个新能力：
- 名称: {}
- 描述: {}
- 动作: {}

当前环境状态：
{}

请根据环境状态，生成一个真实的测试输入（JSON），让这个能力做一件实际有用的事。
只返回 JSON，不要其他文字。"#,
            genome.name,
            genome.description,
            genome
                .actions
                .first()
                .map(|a| &a.name)
                .unwrap_or(&"".to_string()),
            env_json,
        );

        let test_input = match self.llm.execute(&prompt, "fast:envtest", None).await {
            Ok(r) => serde_json::from_str::<serde_json::Value>(&r).unwrap_or(serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        };

        if genome.actions.is_empty() {
            return AutonomousResult {
                goal: AutonomousGoal {
                    description: format!("实验新能力: {}", genome.name),
                    priority: GoalPriority::Exploratory,
                    suggested_capabilities: vec![genome.name.clone()],
                    reason: "主动验证新能力".into(),
                },
                success: false,
                output: serde_json::json!({}),
                error: Some("能力没有动作".into()),
                elapsed_ms: start.elapsed().as_millis() as u64,
                value_feedback: None,
            };
        }

        let action = &genome.actions[0];

        // 注册能力到总线
        let cap = ScriptedCapability::from_genome(genome.clone())
            .with_llm(self.llm.clone())
            .with_bus(self.bus.clone());
        self.bus.register(Arc::new(cap)).await;

        let msg = crate::message::Message::builder()
            .from("autonomous")
            .to(&genome.name)
            .action(&action.name)
            .payload(test_input)
            .build();

        match self.bus.send(msg).await {
            Ok(resp) => {
                let success = resp
                    .payload
                    .get("success")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false);
                let error = if success {
                    None
                } else {
                    resp.payload
                        .get("error")
                        .and_then(|e| e.as_str())
                        .map(String::from)
                };
                let result = AutonomousResult {
                    goal: AutonomousGoal {
                        description: format!("实验新能力: {}", genome.name),
                        priority: GoalPriority::Exploratory,
                        suggested_capabilities: vec![genome.name.clone()],
                        reason: "主动验证新能力".into(),
                    },
                    success,
                    output: resp.payload,
                    error,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    value_feedback: None,
                };
                self.add_history(result.clone());
                result
            }
            Err(e) => {
                let result = AutonomousResult {
                    goal: AutonomousGoal {
                        description: format!("实验新能力: {}", genome.name),
                        priority: GoalPriority::Exploratory,
                        suggested_capabilities: vec![genome.name.clone()],
                        reason: "主动验证新能力".into(),
                    },
                    success: false,
                    output: serde_json::json!({}),
                    error: Some(e.to_string()),
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    value_feedback: None,
                };
                self.add_history(result.clone());
                result
            }
        }
    }

    /// 完整自主循环：感知 → 生成目标 → 执行 → 记录
    pub async fn autonomous_cycle(
        &mut self,
        evolution: &mut EvolutionEngine,
    ) -> Vec<AutonomousResult> {
        let mut results = vec![];

        // 1. 感知环境
        tracing::info!("自主循环: 感知环境...");
        let report = self.perceive().await;
        tracing::info!(
            "自主循环: 环境状态 — 磁盘 {:.0}%, 端口 {} 个, 进程 {} 个, git={}",
            report.disk_usage_pct,
            report.network_listening.len(),
            report.running_processes.len(),
            if report.git_status.is_some() {
                "有变更"
            } else {
                "干净"
            },
        );

        // 2. 生成目标（带思维链记录）
        let caps: Vec<String> = evolution.genomes().keys().cloned().collect();
        tracing::info!("自主循环: 生成目标 (可用能力 {} 个)...", caps.len());
        let (goals, llm_response) = self
            .generate_goals(&report, evolution, Some(evolution.memory()))
            .await;
        tracing::info!("自主循环: 生成 {} 个目标", goals.len());

        // 记录思维链 — 保存 LLM 的完整推理过程供下一轮使用
        if !llm_response.is_empty() {
            let goal_summary = goals
                .iter()
                .map(|g| g.description.clone())
                .collect::<Vec<_>>()
                .join("; ");
            evolution.record_thought_chain(crate::evolution::ThoughtChain {
                chain_type: "goal_generation".to_string(),
                reasoning: safe_truncate(&llm_response, 2000).to_string(),
                conclusion: goal_summary.clone(),
                related_capabilities: goals
                    .iter()
                    .flat_map(|g| g.suggested_capabilities.iter().cloned())
                    .collect(),
                related_goal: Some(goal_summary),
                success: !goals.is_empty(),
                timestamp: now_secs(),
            });
        }

        // 3. 执行目标（带反馈追踪）
        let env_before = serde_json::json!({
            "disk_usage": report.disk_usage_pct,
            "git_status": report.git_status.is_some(),
            "processes": report.running_processes.len(),
            "ports": report.network_listening.len(),
        });

        for goal in &goals {
            tracing::info!(
                "自主循环: 执行目标 [{:?}] {} — {}",
                goal.priority,
                goal.description,
                goal.reason
            );
            let mut result = self.execute_goal(goal, evolution).await;
            if result.success {
                tracing::info!("自主循环: ✅ 成功 ({}ms)", result.elapsed_ms);
            } else {
                tracing::warn!(
                    "自主循环: ❌ 失败 — {}",
                    result.error.as_deref().unwrap_or("未知")
                );
            }

            // 外部反馈追踪：执行后感知环境变化
            let env_after_report = self.perceive().await;
            let env_after = serde_json::json!({
                "disk_usage": env_after_report.disk_usage_pct,
                "git_status": env_after_report.git_status.is_some(),
                "processes": env_after_report.running_processes.len(),
                "ports": env_after_report.network_listening.len(),
            });

            // 检测环境变化
            let mut changes = Vec::new();
            let disk_before = env_before
                .get("disk_usage")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let disk_after = env_after
                .get("disk_usage")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if (disk_after - disk_before).abs() > 0.1 {
                changes.push(format!(
                    "磁盘使用率: {:.1}% → {:.1}%",
                    disk_before, disk_after
                ));
            }
            let git_before = env_before
                .get("git_status")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let git_after = env_after
                .get("git_status")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if git_before != git_after {
                changes.push(format!(
                    "Git 状态: {} → {}",
                    if git_before { "有变更" } else { "干净" },
                    if git_after { "有变更" } else { "干净" }
                ));
            }
            let proc_before = env_before
                .get("processes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let proc_after = env_after
                .get("processes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if proc_before != proc_after {
                changes.push(format!("进程数: {} → {}", proc_before, proc_after));
            }

            // P4b-fix: impact_score 基于执行结果质量计算，而非固定查表
            // 让真正产生有意义输出的能力获得更高 score 增量
            let impact_score = if result.success {
                let output_str = match &result.output {
                    serde_json::Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                let output_len = output_str.len();
                let has_meaningful_output = output_len > 10; // 非空输出
                let has_error = result.error.is_some();
                let base: f64 = 0.3;
                let output_bonus = if has_meaningful_output { 0.15 } else { 0.0 };
                let change_bonus = if !changes.is_empty() { 0.05 } else { 0.0 };
                let error_penalty = if has_error { -0.05 } else { 0.0 };
                (base + output_bonus + change_bonus + error_penalty)
                    .max(0.0)
                    .min(0.5)
            } else {
                -0.2 // 失败
            };

            let cap_used = goal
                .suggested_capabilities
                .first()
                .cloned()
                .unwrap_or_default();
            let cap_for_feedback = cap_used.clone();

            let feedback = FeedbackRecord {
                goal: goal.description.clone(),
                capability: cap_for_feedback,
                env_before: env_before.clone(),
                env_after,
                env_changes: changes.clone(),
                success: result.success,
                impact_score,
                timestamp: now_secs(),
            };

            tracing::info!(
                "自主循环: 反馈追踪 — 影响: {:.1}, 环境变化: {} 项",
                impact_score,
                changes.len()
            );

            // A1: LLM 自主价值判定 → 替代人类反馈
            //
            // 原设计用 impact_score * 0.1 手动调 fitness.score，这绕过了 fitness 的正规记录通道，
            // 且没有"有用性"信号（human_score）。现在用 LLM 判定结果价值，调 record_human_signal
            // 注入真实价值信号（权重 0.95，主导 score），完全替代人类 👍/👎。
            //
            // 这样系统完全自主：LLM 自主判定 → record_human_signal → recompute_score →
            // 有价值的能力 score 飙升，无价值的 score 下跌被淘汰。
            let is_valuable = self.auto_evaluate(goal, &result, &changes).await;
            match is_valuable {
                Some(v) => {
                    tracing::info!(
                        "自主循环: LLM 自主判定 — {}",
                        if v { "✅ 有价值" } else { "❌ 无价值" }
                    );
                    if !cap_used.is_empty() {
                        if let Some(g) = evolution.genomes_mut().get_mut(&cap_used) {
                            // record_human_signal 已经内部调 recompute_score，不需要再手动调 score
                            g.fitness.record_human_signal(v);
                        }
                    }
                    result.value_feedback = Some(v);
                }
                None => {
                    tracing::info!("自主循环: ⏸ 跳过评分（LLM 不可用）");
                }
            }

            self.feedback_records.push(feedback);
            // 保留最近 100 条反馈
            if self.feedback_records.len() > 100 {
                self.feedback_records.remove(0);
            }

            results.push(result);
        }

        results
    }

    /// 获取反馈追踪记录
    pub fn feedback_records(&self) -> &[FeedbackRecord] {
        &self.feedback_records
    }

    /// 获取历史记录
    pub fn history(&self) -> &[AutonomousResult] {
        &self.history
    }

    /// 获取成功/失败统计
    pub fn stats(&self) -> (usize, usize) {
        let successes = self.history.iter().filter(|r| r.success).count();
        let failures = self.history.len() - successes;
        (successes, failures)
    }

    fn add_history(&mut self, result: AutonomousResult) {
        self.history.push(result);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    async fn generate_input_for(&self, genome: &CapabilityGenome) -> serde_json::Value {
        if genome.actions.is_empty() {
            return serde_json::json!({});
        }
        let action = &genome.actions[0];
        let cap_name = &genome.name;
        let cap_desc = &genome.description;
        let action_name = &action.name;
        let action_desc = &action.description;
        let schema = &action.input_schema;

        // 用 LLM 生成真实输入
        let schema_str = serde_json::to_string_pretty(schema).unwrap_or_default();
        let prompt = format!(
            r#"为能力测试生成真实合理的输入数据。

能力: {cap_name} — {cap_desc}
动作: {action_name} — {action_desc}
输入 Schema: {schema_str}

请生成一个真实场景下的测试输入，确保数据有意义、能触发核心逻辑。
返回严格 JSON（直接可用的输入对象，不要包裹在其他结构中）:"#
        );

        match self.llm.execute(&prompt, "fast:autoinput", None).await {
            Ok(text) => {
                let json_str = extract_json(&text);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    if v.is_object() {
                        return v;
                    }
                }
                // LLM 返回无效 JSON，回退到占位值
                self.fallback_input(action)
            }
            Err(_) => {
                // LLM 调用失败，回退到占位值
                self.fallback_input(action)
            }
        }
    }

    fn fallback_input(&self, action: &crate::genome::ActionGene) -> serde_json::Value {
        let mut test = serde_json::Map::new();
        if let Some(props) = action
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            for (key, schema) in props {
                let value = match schema.get("type").and_then(|t| t.as_str()) {
                    Some("string") => serde_json::json!("test"),
                    Some("integer") | Some("number") => serde_json::json!(42),
                    Some("boolean") => serde_json::json!(true),
                    Some("array") => serde_json::json!([]),
                    Some("object") => serde_json::json!({}),
                    _ => serde_json::json!("test"),
                };
                test.insert(key.clone(), value);
            }
        }
        serde_json::Value::Object(test)
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

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

fn extract_json(text: &str) -> String {
    let start = text.find('{');
    let end = text.rfind('}');
    match (start, end) {
        (Some(s), Some(e)) if e > s => text[s..=e].to_string(),
        _ => text.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_report_default() {
        let report = EnvironmentReport::default();
        assert_eq!(report.disk_usage_pct, 0.0);
        assert!(report.recent_files.is_empty());
    }

    #[test]
    fn test_goal_priority_display() {
        assert_eq!(GoalPriority::Critical.to_string(), "Critical");
        assert_eq!(GoalPriority::High.to_string(), "High");
        assert_eq!(GoalPriority::Medium.to_string(), "Medium");
        assert_eq!(GoalPriority::Low.to_string(), "Low");
        assert_eq!(GoalPriority::Exploratory.to_string(), "Exploratory");
    }

    #[test]
    fn test_goal_priority_serialization() {
        let goal = AutonomousGoal {
            description: "test".into(),
            priority: GoalPriority::High,
            suggested_capabilities: vec!["cap".into()],
            reason: "because".into(),
        };
        let json = serde_json::to_string(&goal).unwrap();
        let decoded: AutonomousGoal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.priority, GoalPriority::High);
        assert_eq!(decoded.description, "test");
    }

    #[test]
    fn test_environment_report_serialization() {
        let report = EnvironmentReport {
            disk_usage_pct: 75.5,
            cpu_usage_pct: 30.0,
            memory_usage_pct: 60.0,
            uptime_secs: 3600,
            recent_files: vec!["file.rs".into()],
            git_status: Some("M file.rs".into()),
            running_processes: vec!["cargo".into()],
            network_listening: vec![8080],
            log_anomalies: vec!["error".into()],
            timestamp: 1234567890,
            ..Default::default()
        };
        let json = serde_json::to_string(&report).unwrap();
        let decoded: EnvironmentReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.disk_usage_pct, 75.5);
        assert_eq!(decoded.network_listening, vec![8080]);
    }

    #[test]
    fn test_autonomous_result_serialization() {
        let result = AutonomousResult {
            goal: AutonomousGoal {
                description: "do thing".into(),
                priority: GoalPriority::Medium,
                suggested_capabilities: vec![],
                reason: "test".into(),
            },
            success: true,
            output: serde_json::json!({"ok": true}),
            error: None,
            elapsed_ms: 500,
            value_feedback: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: AutonomousResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.success);
        assert_eq!(decoded.elapsed_ms, 500);
    }

    #[test]
    fn test_extract_json_found() {
        let text = r#"prefix {"key": "val"} suffix"#;
        let result = extract_json(text);
        assert!(result.contains("\"key\""));
    }

    #[test]
    fn test_extract_json_not_found() {
        let text = "no json here";
        let result = extract_json(text);
        assert_eq!(result, "no json here");
    }

    #[tokio::test]
    async fn test_perceive_returns_report() {
        let llm = Arc::new(crate::genome::LlmExecutor::new("dummy", "http://localhost"));
        let bus = Arc::new(MessageBus::new());
        let platform = Platform::detect();
        let rt = AutonomousRuntime::new(llm, bus, platform);
        let report = rt.perceive().await;
        assert!(report.timestamp > 0);
    }

    #[tokio::test]
    async fn test_autonomous_runtime_new() {
        let llm = Arc::new(crate::genome::LlmExecutor::new("dummy", "http://localhost"));
        let bus = Arc::new(MessageBus::new());
        let platform = Platform::detect();
        let rt = AutonomousRuntime::new(llm, bus, platform);
        assert_eq!(rt.max_history, 100);
        assert!(rt.history.is_empty());
    }
}

/// 判断目标是否与近期历史重复或属于低价值维护类。
///
/// 规则:
/// 1. 维护关键词黑名单 — 目标描述含这些词 → 直接判重复
///    （"提交变更"/"清理临时文件"/"检查日志" 这类不调用能力的维护操作）
/// 2. 核心意图重复 — 只比较目标描述前 40 字（去掉代码细节/文件路径），
///    如果与近期目标的前 40 字高度相似（包含关系）→ 判重复
fn is_duplicate_goal(desc: &str, recent_descs: &[String]) -> bool {
    let normalized = desc.to_lowercase();

    // 维护关键词黑名单（这些词出现 = 维护类目标，一律拦截）
    const MAINTENANCE_KEYWORDS: &[&str] = &[
        "提交变更",
        "commit",
        "清理临时",
        "清理磁盘",
        "检查日志",
        "分析异常日志",
        "释放磁盘空间",
    ];
    if MAINTENANCE_KEYWORDS
        .iter()
        .any(|kw| normalized.contains(kw))
    {
        return true;
    }

    // 只取描述前 40 字作为"核心意图"比较，忽略后面的代码细节/文件路径
    // 这样 "验证 X 能力：[100字代码]" 和 "验证 Y 能力：[100字代码]" 不会误判重复
    let core = take_chars(&normalized, 40);

    for recent in recent_descs {
        let recent_core = take_chars(&recent.to_lowercase(), 40);
        // 核心意图包含关系（一方前40字包含另一方前40字）
        if recent_core.len() >= 8 && (core.contains(&recent_core) || recent_core.contains(&core)) {
            return true;
        }
    }

    false
}

/// 取字符串前 n 个字符（按 char，非字节）
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
