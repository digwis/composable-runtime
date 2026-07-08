use serde::{Deserialize, Serialize};

/// 工作流（Workflow）— 编排多个能力完成复杂任务
///
/// 工作流由一系列步骤组成，每个步骤调用一个能力的某个动作。
/// 步骤之间通过上下文（Context）传递数据，
/// 支持条件分支、变量引用、并行组和重试策略。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// 工作流名称
    pub name: String,
    /// 工作流描述
    #[serde(default)]
    pub description: String,
    /// 工作流步骤（支持并行组）
    pub steps: Vec<StepEntry>,
}

/// 工作流步骤条目 — 可以是单步或并行组
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StepEntry {
    /// 单个步骤
    Single(Step),
    /// 并行执行组
    Parallel(ParallelGroup),
}

/// 并行执行组 — 组内所有步骤同时执行
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    /// 组名称
    pub name: String,
    /// 并行执行的步骤列表
    pub parallel: Vec<Step>,
}

/// 工作流步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// 步骤名称
    pub name: String,
    /// 目标能力名称
    pub capability: String,
    /// 调用的动作
    pub action: String,
    /// 输入负载（JSON）
    #[serde(default)]
    pub input: serde_json::Value,
    /// 条件表达式（可选）
    ///
    /// 支持语法：`context.key == "value"`、`context.key != null`、`context.key > 5`
    #[serde(default)]
    pub condition: Option<StepCondition>,
    /// 重试策略（可选）
    #[serde(default)]
    pub retry: Option<RetryPolicy>,
    /// 超时（毫秒，可选）
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// 错误处理策略
    #[serde(default)]
    pub on_error: ErrorStrategy,
}

/// 重试策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// 最大重试次数
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// 重试间隔（毫秒）
    #[serde(default = "default_retry_delay")]
    pub delay_ms: u64,
    /// 指数退避倍数（1 = 固定间隔，2 = 指数退避）
    #[serde(default = "default_backoff")]
    pub backoff_multiplier: f64,
}

fn default_max_retries() -> u32 {
    3
}
fn default_retry_delay() -> u64 {
    100
}
fn default_backoff() -> f64 {
    2.0
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            delay_ms: default_retry_delay(),
            backoff_multiplier: default_backoff(),
        }
    }
}

/// 错误处理策略
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ErrorStrategy {
    /// 出错即停止工作流（默认）
    #[default]
    Stop,
    /// 出错后跳过该步骤，继续执行
    Continue,
    /// 出错后将错误存入上下文，继续执行
    Record,
}

/// 步骤条件
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StepCondition {
    /// 简单字符串表达式
    Expr(String),
}

impl Workflow {
    /// 从 YAML 字符串解析工作流
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// 从文件加载工作流
    pub fn from_file(path: &str) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content).map_err(Into::into)
    }
}

impl StepEntry {
    /// 获取步骤条目的名称
    pub fn name(&self) -> &str {
        match self {
            StepEntry::Single(s) => &s.name,
            StepEntry::Parallel(g) => &g.name,
        }
    }
}

impl ParallelGroup {
    /// 创建新的并行组
    pub fn new(name: impl Into<String>, steps: Vec<Step>) -> Self {
        Self {
            name: name.into(),
            parallel: steps,
        }
    }
}

impl Step {
    /// 创建新的步骤
    pub fn new(
        name: impl Into<String>,
        capability: impl Into<String>,
        action: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            capability: capability.into(),
            action: action.into(),
            input,
            condition: None,
            retry: None,
            timeout_ms: None,
            on_error: ErrorStrategy::default(),
        }
    }

    /// 设置条件
    pub fn with_condition(mut self, expr: impl Into<String>) -> Self {
        self.condition = Some(StepCondition::Expr(expr.into()));
        self
    }

    /// 设置重试策略
    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }

    /// 设置超时
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// 设置错误策略
    pub fn on_error(mut self, strategy: ErrorStrategy) -> Self {
        self.on_error = strategy;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_new() {
        let step = Step::new("s1", "compute", "add", serde_json::json!({"a": 1, "b": 2}));
        assert_eq!(step.name, "s1");
        assert_eq!(step.capability, "compute");
        assert_eq!(step.action, "add");
        assert_eq!(step.input["a"], 1);
        assert!(step.condition.is_none());
        assert!(step.retry.is_none());
        assert!(step.timeout_ms.is_none());
        assert!(matches!(step.on_error, ErrorStrategy::Stop));
    }

    #[test]
    fn test_step_with_condition() {
        let step = Step::new("s1", "compute", "add", serde_json::json!({}))
            .with_condition("step-1.result != null");
        assert!(step.condition.is_some());
    }

    #[test]
    fn test_step_with_retry() {
        let step = Step::new("s1", "compute", "add", serde_json::json!({}))
            .with_retry(RetryPolicy {
                max_retries: 5,
                delay_ms: 200,
                backoff_multiplier: 3.0,
            });
        let retry = step.retry.unwrap();
        assert_eq!(retry.max_retries, 5);
        assert_eq!(retry.delay_ms, 200);
        assert_eq!(retry.backoff_multiplier, 3.0);
    }

    #[test]
    fn test_step_with_timeout() {
        let step = Step::new("s1", "compute", "add", serde_json::json!({}))
            .with_timeout(5000);
        assert_eq!(step.timeout_ms, Some(5000));
    }

    #[test]
    fn test_step_on_error_continue() {
        let step = Step::new("s1", "compute", "add", serde_json::json!({}))
            .on_error(ErrorStrategy::Continue);
        assert!(matches!(step.on_error, ErrorStrategy::Continue));
    }

    #[test]
    fn test_parallel_group_new() {
        let group = ParallelGroup::new("grp", vec![
            Step::new("a", "compute", "add", serde_json::json!({})),
            Step::new("b", "greet", "hello", serde_json::json!({})),
        ]);
        assert_eq!(group.name, "grp");
        assert_eq!(group.parallel.len(), 2);
    }

    #[test]
    fn test_step_entry_name() {
        let single = StepEntry::Single(Step::new("s1", "cap", "act", serde_json::json!({})));
        assert_eq!(single.name(), "s1");

        let parallel = StepEntry::Parallel(ParallelGroup::new("grp", vec![]));
        assert_eq!(parallel.name(), "grp");
    }

    #[test]
    fn test_workflow_from_yaml() {
        let yaml = r#"
name: test-wf
description: 测试工作流
steps:
  - name: step-1
    capability: greet
    action: hello
    input:
      name: "世界"
"#;
        let wf = Workflow::from_yaml(yaml).unwrap();
        assert_eq!(wf.name, "test-wf");
        assert_eq!(wf.description, "测试工作流");
        assert_eq!(wf.steps.len(), 1);
    }

    #[test]
    fn test_workflow_from_yaml_with_parallel() {
        let yaml = r#"
name: parallel-wf
steps:
  - name: grp
    parallel:
      - name: a
        capability: compute
        action: add
        input: { a: 1, b: 2 }
      - name: b
        capability: greet
        action: hello
        input: { name: "x" }
"#;
        let wf = Workflow::from_yaml(yaml).unwrap();
        assert_eq!(wf.steps.len(), 1);
        match &wf.steps[0] {
            StepEntry::Parallel(g) => {
                assert_eq!(g.name, "grp");
                assert_eq!(g.parallel.len(), 2);
            }
            _ => panic!("应为并行组"),
        }
    }

    #[test]
    fn test_workflow_from_yaml_with_retry() {
        let yaml = r#"
name: retry-wf
steps:
  - name: risky
    capability: compute
    action: divide
    input: { a: 10, b: 0 }
    retry:
      max_retries: 3
      delay_ms: 100
      backoff_multiplier: 2.0
    on_error: continue
"#;
        let wf = Workflow::from_yaml(yaml).unwrap();
        match &wf.steps[0] {
            StepEntry::Single(s) => {
                let retry = s.retry.as_ref().unwrap();
                assert_eq!(retry.max_retries, 3);
                assert!(matches!(s.on_error, ErrorStrategy::Continue));
            }
            _ => panic!("应为单步"),
        }
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.delay_ms, 100);
        assert_eq!(policy.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_workflow_from_invalid_yaml() {
        let result = Workflow::from_yaml("not valid yaml: {{{");
        assert!(result.is_err());
    }
}
