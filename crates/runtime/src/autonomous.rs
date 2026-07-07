use crate::evolution::EvolutionEngine;
use crate::genome::{CapabilityGenome, LlmExecutor, ScriptedCapability};
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
}

/// 自主运行时 — 从"被动器官"到"主动生物体"
///
/// 三层自主性：
/// 1. 环境感知 — 主动观测环境状态变化
/// 2. 目标生成 — 基于环境+能力+记忆，自主决定"该做什么"
/// 3. 主动实验 — 用新能力在真实环境中尝试
pub struct AutonomousRuntime {
    llm: Arc<LlmExecutor>,
    bus: Arc<MessageBus>,
    platform: Platform,
    history: Vec<AutonomousResult>,
    max_history: usize,
}

impl AutonomousRuntime {
    pub fn new(llm: Arc<LlmExecutor>, bus: Arc<MessageBus>, platform: Platform) -> Self {
        Self {
            llm, bus, platform,
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// 1. 环境感知 — 通过已有能力观测环境
    pub async fn perceive(&self) -> EnvironmentReport {
        let mut report = EnvironmentReport::default();
        report.timestamp = now_secs();

        // 磁盘使用率
        if let Ok(out) = tokio::process::Command::new("df")
            .args(&["-h", "/"])
            .output().await
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
            .args(&["aux", "--sort=-%cpu"])
            .output().await
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
            .args(&["-iTCP", "-sTCP:LISTEN", "-P", "-n"])
            .output().await
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
            .args(&["status", "--short"])
            .output().await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            if !s.trim().is_empty() {
                report.git_status = Some(s.lines().take(20).collect::<Vec<_>>().join("\n"));
            }
        }

        // 最近修改的文件
        if let Ok(out) = tokio::process::Command::new("find")
            .args(&[".", "-name", "*.rs", "-o", "-name", "*.py", "-o", "-name", "*.ts"])
            .arg("-mtime").arg("-1")
            .output().await
        {
            let s = String::from_utf8_lossy(&out.stdout);
            report.recent_files = s.lines().take(20).map(String::from).collect();
        }

        report
    }

    /// 2. 自主目标生成 — LLM 根据环境+能力生成"该做的事"
    pub async fn generate_goals(
        &self,
        report: &EnvironmentReport,
        capabilities: &[String],
    ) -> Vec<AutonomousGoal> {
        let report_json = serde_json::to_string_pretty(report).unwrap_or_default();
        let caps_str = capabilities.join(", ");

        let prompt = format!(
r#"你是自主运行时的目标生成器。

当前环境状态：
{}

可用能力：{}

请根据环境状态和可用能力，生成 1-5 个"值得做的事"。
优先级规则：
- 磁盘 >90% 或异常进程 → Critical
- 磁盘 >80% 或有未提交的重要变更 → High
- 有新文件需要检查 → Medium
- 可以尝试新能力 → Exploratory

返回 JSON 数组：
[
  {{
    "description": "具体要做的事",
    "priority": "High",
    "suggested_capabilities": ["能力名"],
    "reason": "为什么需要做这件事"
  }}
]
只返回 JSON。"#,
            report_json, caps_str
        );

        let response = match self.llm.execute(&prompt, "auto", None).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("自主目标生成: LLM 调用失败: {}", e);
                return vec![];
            }
        };

        if let Ok(goals) = serde_json::from_str::<Vec<AutonomousGoal>>(&response) {
            return goals;
        }
        // 尝试提取 JSON 数组
        if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                if let Ok(goals) = serde_json::from_str::<Vec<AutonomousGoal>>(&response[start..=end]) {
                    return goals;
                }
            }
        }
        vec![]
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
                    let input = self.generate_input_for(genome);

                    let msg = crate::message::Message::builder()
                        .from("autonomous")
                        .to(cap_name)
                        .action(&action.name)
                        .payload(input)
                        .build();

                    match self.bus.send(msg).await {
                        Ok(resp) => {
                            let success = resp.payload.get("success")
                                .and_then(|s| s.as_bool())
                                .unwrap_or(false);
                            let error = if success { None } else {
                                resp.payload.get("error")
                                    .and_then(|e| e.as_str())
                                    .map(String::from)
                            };
                            let result = AutonomousResult {
                                goal: goal.clone(),
                                success,
                                output: resp.payload,
                                error,
                                elapsed_ms: start.elapsed().as_millis() as u64,
                            };
                            self.add_history(result.clone());
                            return result;
                        }
                        Err(e) => {
                            let result = AutonomousResult {
                                goal: goal.clone(),
                                success: false,
                                output: serde_json::json!({}),
                                error: Some(e.to_string()),
                                elapsed_ms: start.elapsed().as_millis() as u64,
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
        };
        self.add_history(result.clone());
        result
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
            genome.actions.first().map(|a| &a.name).unwrap_or(&"".to_string()),
            env_json,
        );

        let test_input = match self.llm.execute(&prompt, "auto", None).await {
            Ok(r) => {
                serde_json::from_str::<serde_json::Value>(&r).unwrap_or(serde_json::json!({}))
            }
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
                let success = resp.payload.get("success")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false);
                let error = if success { None } else {
                    resp.payload.get("error")
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
            if report.git_status.is_some() { "有变更" } else { "干净" },
        );

        // 2. 生成目标
        let caps: Vec<String> = evolution.genomes().keys().cloned().collect();
        tracing::info!("自主循环: 生成目标 (可用能力 {} 个)...", caps.len());
        let goals = self.generate_goals(&report, &caps).await;
        tracing::info!("自主循环: 生成 {} 个目标", goals.len());

        // 3. 执行目标
        for goal in &goals {
            tracing::info!(
                "自主循环: 执行目标 [{:?}] {} — {}",
                goal.priority, goal.description, goal.reason
            );
            let result = self.execute_goal(goal, evolution).await;
            if result.success {
                tracing::info!("自主循环: ✅ 成功 ({}ms)", result.elapsed_ms);
            } else {
                tracing::warn!(
                    "自主循环: ❌ 失败 — {}",
                    result.error.as_deref().unwrap_or("未知")
                );
            }
            results.push(result);
        }

        results
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

    fn generate_input_for(&self, genome: &CapabilityGenome) -> serde_json::Value {
        if genome.actions.is_empty() {
            return serde_json::json!({});
        }
        let action = &genome.actions[0];
        let mut test = serde_json::Map::new();
        if let Some(props) = action.input_schema.get("properties").and_then(|p| p.as_object()) {
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
