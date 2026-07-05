use crate::message::{Message, MessageError};
use crate::message_bus::MessageBus;
use crate::workflow::{ErrorStrategy, Step, StepCondition, StepEntry, Workflow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

/// 编排引擎（Orchestrator）— 按工作流定义编排能力
///
/// 编排引擎接收一个工作流定义，按步骤依次调用对应能力，
/// 将每一步的输出存入上下文，后续步骤可引用上下文中的数据。
pub struct Orchestrator {
    bus: Arc<MessageBus>,
}

/// 编排执行结果
#[derive(Debug, Clone)]
pub struct OrchestratorResult {
    /// 工作流名称
    pub workflow: String,
    /// 执行的步骤数
    pub steps_executed: usize,
    /// 步骤跳过数（因条件不满足）
    pub steps_skipped: usize,
    /// 步骤失败数
    pub steps_failed: usize,
    /// 步骤重试次数
    pub steps_retried: usize,
    /// 最终上下文
    pub context: HashMap<String, serde_json::Value>,
    /// 每步的输出
    pub outputs: Vec<StepOutput>,
    /// 执行是否完全成功
    pub success: bool,
}

/// 单步执行输出
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub step: String,
    pub capability: String,
    pub action: String,
    pub result: Result<serde_json::Value, String>,
    /// 重试次数
    pub retries: u32,
    /// 是否为并行组内的步骤
    pub parallel_group: Option<String>,
}

/// 编排构建器
pub struct OrchestratorBuilder {
    bus: Option<Arc<MessageBus>>,
}

impl OrchestratorBuilder {
    pub fn new() -> Self {
        Self { bus: None }
    }

    pub fn with_bus(mut self, bus: MessageBus) -> Self {
        self.bus = Some(Arc::new(bus));
        self
    }

    pub fn with_bus_arc(mut self, bus: Arc<MessageBus>) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn build(self) -> Orchestrator {
        Orchestrator {
            bus: self.bus.unwrap_or_else(|| Arc::new(MessageBus::new())),
        }
    }
}

impl Default for OrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    pub fn new(bus: MessageBus) -> Self {
        Self {
            bus: Arc::new(bus),
        }
    }

    /// 执行工作流
    pub async fn run(&self, workflow: &Workflow) -> Result<OrchestratorResult, MessageError> {
        tracing::info!("开始执行工作流: {} ({} 条目)", workflow.name, workflow.steps.len());

        let mut context: HashMap<String, serde_json::Value> = HashMap::new();
        let mut outputs = Vec::new();
        let mut steps_executed = 0;
        let mut steps_skipped = 0;
        let mut steps_failed = 0;
        let mut steps_retried = 0usize;
        let mut success = true;

        for entry in &workflow.steps {
            match entry {
                StepEntry::Single(step) => {
                    // 检查条件
                    if let Some(condition) = &step.condition {
                        if !self.evaluate_condition(condition, &context) {
                            tracing::info!("步骤 '{}' 条件不满足，跳过", step.name);
                            steps_skipped += 1;
                            continue;
                        }
                    }

                    let (output, retried, _failed) = self.execute_step(step, &workflow.name, &context).await;
                    steps_retried += retried as usize;

                    let mut should_stop = false;
                    let error_detail: Option<String> = match &output.result {
                        Ok(payload) => {
                            context.insert(step.name.clone(), payload.clone());
                            steps_executed += 1;
                            None
                        }
                        Err(e) => {
                            steps_failed += 1;
                            success = false;
                            match step.on_error {
                                ErrorStrategy::Stop => {
                                    tracing::error!("步骤 '{}' 失败，停止工作流: {}", step.name, e);
                                    should_stop = true;
                                    Some(e.clone())
                                }
                                ErrorStrategy::Continue => {
                                    tracing::warn!("步骤 '{}' 失败，跳过继续: {}", step.name, e);
                                    steps_skipped += 1;
                                    None
                                }
                                ErrorStrategy::Record => {
                                    tracing::warn!("步骤 '{}' 失败，记录错误继续: {}", step.name, e);
                                    context.insert(
                                        step.name.clone(),
                                        serde_json::json!({"error": e, "step": &step.name}),
                                    );
                                    steps_executed += 1;
                                    None
                                }
                            }
                        }
                    };
                    outputs.push(output);
                    if should_stop {
                        return Err(MessageError::Internal {
                            capability: step.capability.clone(),
                            detail: format!("步骤 '{}' 失败: {}", step.name, error_detail.unwrap()),
                        });
                    }
                }
                StepEntry::Parallel(group) => {
                    tracing::info!("执行并行组: '{}' ({} 步)", group.name, group.parallel.len());

                    let mut join_set = tokio::task::JoinSet::new();
                    let bus = self.bus.clone();

                    for step in &group.parallel {
                        let resolved_input = self.resolve_variables(&step.input, &context);
                        let step = step.clone();
                        let wf_name = workflow.name.clone();
                        let bus = bus.clone();

                        join_set.spawn(async move {
                            execute_step_with_retry(&bus, &step, &wf_name, resolved_input).await
                        });
                    }

                    let mut group_outputs = Vec::new();
                    let mut group_failed = 0;
                    let mut group_retried = 0;

                    while let Some(res) = join_set.join_next().await {
                        let (output, retried, failed) = res.unwrap();
                        group_retried += retried;
                        if failed > 0 {
                            group_failed += 1;
                        }
                        group_outputs.push(output);
                    }

                    steps_retried += group_retried as usize;

                    for output in &group_outputs {
                        if let Ok(payload) = &output.result {
                            context.insert(output.step.clone(), payload.clone());
                        }
                    }

                    if group_failed > 0 {
                        steps_failed += group_failed;
                        success = false;
                        tracing::warn!("并行组 '{}' 有 {} 步失败", group.name, group_failed);
                    }

                    steps_executed += group_outputs.len();

                    // 将并行组结果汇总存入上下文
                    let group_summary: serde_json::Value = serde_json::to_value(
                        group_outputs.iter().map(|o| {
                            serde_json::json!({
                                "step": &o.step,
                                "result": match &o.result {
                                    Ok(v) => v.clone(),
                                    Err(e) => serde_json::json!({"error": e}),
                                }
                            })
                        }).collect::<Vec<_>>(),
                    ).unwrap_or(serde_json::Value::Null);
                    context.insert(group.name.clone(), group_summary);

                    outputs.extend(group_outputs);
                }
            }
        }

        tracing::info!(
            "工作流 '{}' 完成: {} 步执行, {} 步跳过, {} 步失败, {} 步重试",
            workflow.name, steps_executed, steps_skipped, steps_failed, steps_retried
        );

        Ok(OrchestratorResult {
            workflow: workflow.name.clone(),
            steps_executed,
            steps_skipped,
            steps_failed,
            steps_retried,
            context,
            outputs,
            success,
        })
    }

    /// 执行单个步骤（含重试和超时）
    async fn execute_step(
        &self,
        step: &Step,
        workflow_name: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> (StepOutput, u32, usize) {
        let resolved_input = self.resolve_variables(&step.input, context);
        execute_step_with_retry(&self.bus, step, workflow_name, resolved_input).await
    }

    /// 解析变量引用 `${step_name}` 或 `${step_name.field}`
    fn resolve_variables(
        &self,
        value: &serde_json::Value,
        context: &HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with("${") && s.ends_with('}') {
                    let ref_path = &s[2..s.len() - 1];
                    return self.resolve_ref(ref_path, context).unwrap_or(value.clone());
                }
                value.clone()
            }
            serde_json::Value::Object(map) => {
                let new_map: serde_json::Map<String, serde_json::Value> = map
                    .iter()
                    .map(|(k, v)| (k.clone(), self.resolve_variables(v, context)))
                    .collect();
                serde_json::Value::Object(new_map)
            }
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(
                    arr.iter()
                        .map(|v| self.resolve_variables(v, context))
                        .collect(),
                )
            }
            _ => value.clone(),
        }
    }

    /// 解析引用路径
    fn resolve_ref(
        &self,
        path: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> Option<serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let key = parts.first()?;
        let mut current = context.get(*key)?.clone();

        for part in parts.iter().skip(1) {
            if let serde_json::Value::Object(map) = &current {
                current = map.get(*part)?.clone();
            } else if let serde_json::Value::Array(arr) = &current {
                let idx: usize = part.parse().ok()?;
                current = arr.get(idx)?.clone();
            } else {
                return None;
            }
        }

        Some(current)
    }

    /// 评估条件表达式
    ///
    /// 支持格式：
    /// - `context.key == "value"`
    /// - `context.key != null`
    /// - `context.key == true`
    /// - `context.key > 5` / `>=` / `<` / `<=`
    fn evaluate_condition(
        &self,
        condition: &StepCondition,
        context: &HashMap<String, serde_json::Value>,
    ) -> bool {
        let StepCondition::Expr(expr) = condition;

        // 简单条件解析
        if let Some((left, op, right)) = parse_condition(expr) {
            let left_val = self.resolve_ref(&left, context);

            match op.as_str() {
                "==" => {
                    if right == "null" {
                        return left_val.is_none();
                    }
                    match &left_val {
                        Some(serde_json::Value::String(s)) => {
                            let right_trimmed = right.trim_matches('"');
                            s == right_trimmed
                        }
                        Some(serde_json::Value::Bool(b)) => {
                            right == "true" && *b || right == "false" && !*b
                        }
                        Some(v) => v.to_string() == right,
                        None => false,
                    }
                }
                "!=" => {
                    if right == "null" {
                        return left_val.is_some();
                    }
                    !self.evaluate_condition(
                        &StepCondition::Expr(format!("{left} == {right}")),
                        context,
                    )
                }
                ">" | "<" | ">=" | "<=" => {
                    let left_num = left_val.as_ref().and_then(|v| v.as_f64());
                    let right_num: Option<f64> = right.parse().ok();
                    match (left_num, right_num) {
                        (Some(l), Some(r)) => match op.as_str() {
                            ">" => l > r,
                            "<" => l < r,
                            ">=" => l >= r,
                            "<=" => l <= r,
                            _ => false,
                        },
                        _ => false,
                    }
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// 获取消息总线引用
    pub fn bus(&self) -> &Arc<MessageBus> {
        &self.bus
    }

    /// 能力自省 — 列出所有已注册能力及其动作
    pub async fn introspect(&self) -> Vec<CapabilityInfo> {
        self.bus.introspect().await
    }

    /// 动态执行 — AI Agent 可通过此接口动态执行单步
    ///
    /// AI Agent 可以根据上一步结果决定下一步执行什么，
    /// 而不需要预定义工作流。
    pub async fn execute_dynamic(
        &self,
        step: &Step,
        context: &HashMap<String, serde_json::Value>,
    ) -> (StepOutput, u32, usize) {
        self.execute_step(step, "dynamic", context).await
    }

    /// 从 JSON 动态构建步骤并执行
    ///
    /// 适用于 AI Agent 返回 JSON 指令直接执行的场景
    pub async fn execute_json(
        &self,
        json: serde_json::Value,
        context: &HashMap<String, serde_json::Value>,
    ) -> Result<(StepOutput, u32, usize), serde_json::Error> {
        let step: Step = serde_json::from_value(json)?;
        Ok(self.execute_step(&step, "dynamic", context).await)
    }
}

/// 能力自省信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct CapabilityInfo {
    pub name: String,
    pub version: String,
    pub actions: Vec<String>,
    pub description: String,
}

/// 执行单步（含重试和超时）— 自由函数，可用于并行 spawn
async fn execute_step_with_retry(
    bus: &Arc<MessageBus>,
    step: &Step,
    workflow_name: &str,
    resolved_input: serde_json::Value,
) -> (StepOutput, u32, usize) {
    tracing::info!(
        "执行步骤: {} -> {}:{}",
        step.name,
        step.capability,
        step.action
    );

    let retry_policy = step.retry.clone().unwrap_or_default();
    let mut retries = 0u32;
    let mut last_error = String::new();

    let max_attempts = retry_policy.max_retries + 1;

    for attempt in 0..max_attempts {
        if attempt > 0 {
            let delay = retry_policy.delay_ms as f64
                * retry_policy.backoff_multiplier.powi(attempt as i32 - 1);
            tracing::info!(
                "步骤 '{}' 第 {} 次重试 (延迟 {}ms)",
                step.name,
                attempt,
                delay as u64
            );
            tokio::time::sleep(Duration::from_millis(delay as u64)).await;
            retries += 1;
        }

        let msg = Message::builder()
            .from("orchestrator")
            .to(&step.capability)
            .action(&step.action)
            .payload(resolved_input.clone())
            .metadata("workflow", workflow_name)
            .metadata("step", &step.name)
            .metadata("attempt", &attempt.to_string())
            .build();

        let send_future = bus.send(msg);

        let result = if let Some(ms) = step.timeout_ms {
            match timeout(Duration::from_millis(ms), send_future).await {
                Ok(r) => r,
                Err(_) => Err(MessageError::Internal {
                    capability: step.capability.clone(),
                    detail: format!("步骤 '{}' 超时 ({}ms)", step.name, ms),
                }),
            }
        } else {
            send_future.await
        };

        match result {
            Ok(response) => {
                return (
                    StepOutput {
                        step: step.name.clone(),
                        capability: step.capability.clone(),
                        action: step.action.clone(),
                        result: Ok(response.payload.clone()),
                        retries,
                        parallel_group: None,
                    },
                    retries,
                    0,
                );
            }
            Err(e) => {
                last_error = e.to_string();
                tracing::warn!(
                    "步骤 '{}' 第 {} 次尝试失败: {}",
                    step.name,
                    attempt + 1,
                    e
                );
            }
        }
    }

    (
        StepOutput {
            step: step.name.clone(),
            capability: step.capability.clone(),
            action: step.action.clone(),
            result: Err(last_error),
            retries,
            parallel_group: None,
        },
        retries,
        1,
    )
}

/// 解析条件表达式为 (left, op, right)
fn parse_condition(expr: &str) -> Option<(String, String, String)> {
    let expr = expr.trim();

    for op in &[">=", "<=", "==", "!=", ">", "<"] {
        if let Some(idx) = expr.find(op) {
            let left = expr[..idx].trim().to_string();
            let right = expr[idx + op.len()..].trim().to_string();
            // 去掉 context. 前缀
            let left = left.strip_prefix("context.").unwrap_or(&left).to_string();
            return Some((left, op.to_string(), right));
        }
    }

    None
}
