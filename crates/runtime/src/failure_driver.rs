use crate::ab_test::{ABTester, ABTestRecommendation, ABTestResult};
use crate::genome::{CapabilityGenome, ActionImpl, LlmExecutor};
use crate::sandbox::Sandbox;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 失败事件 — 从任务执行失败中提取
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureEvent {
    pub task: String,
    pub capability: String,
    pub action: String,
    pub input: serde_json::Value,
    pub error: String,
    pub timestamp: String,
}

/// 能力缺口 — 分析失败后识别出的缺失
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityGap {
    pub description: String,
    pub suggested_name: String,
    pub suggested_actions: Vec<String>,
    pub related_failures: Vec<FailureEvent>,
}

/// 失败驱动进化器
///
/// 核心循环：
/// 1. 收集失败事件
/// 2. LLM 分析失败原因 → 识别能力缺口
/// 3. LLM 合成新能力代码
/// 4. 沙箱验证
/// 5. A/B 测试（如果已有类似能力）
/// 6. 通过则注册
pub struct FailureDriver {
    llm: Arc<LlmExecutor>,
    sandbox: Sandbox,
    ab_tester: ABTester,
    failures: Vec<FailureEvent>,
}

impl FailureDriver {
    pub fn new(llm: Arc<LlmExecutor>) -> Self {
        let sandbox = Sandbox::with_defaults();
        let ab_tester = ABTester::new(Sandbox::with_defaults());
        Self {
            llm,
            sandbox,
            ab_tester,
            failures: vec![],
        }
    }

    pub fn with_sandbox(llm: Arc<LlmExecutor>, sandbox: Sandbox) -> Self {
        let ab_tester = ABTester::new(sandbox.clone());
        Self {
            llm,
            sandbox,
            ab_tester,
            failures: vec![],
        }
    }

    /// 记录失败事件
    pub fn record_failure(&mut self, event: FailureEvent) {
        tracing::warn!(
            "失败驱动: 记录失败 — 任务='{}', 能力='{}', 错误='{}'",
            event.task, event.capability, event.error
        );
        self.failures.push(event);
    }

    /// 是否有未处理的失败事件
    pub fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// 分析失败，识别能力缺口
    ///
    /// 用 LLM 分析失败模式，判断是"现有能力不够好"还是"缺少能力"
    pub async fn analyze_gaps(&self) -> Vec<CapabilityGap> {
        if self.failures.is_empty() {
            return vec![];
        }

        let failures_json = serde_json::to_string_pretty(&self.failures).unwrap_or_default();

        let prompt = format!(
            r#"你是能力运行时的进化分析器。以下是系统运行中收集的失败事件：

{}

请分析这些失败，识别能力缺口。对于每个缺口，返回 JSON 数组：
[
  {{
    "description": "缺口的描述",
    "suggested_name": "建议的新能力名称（snake_case）",
    "suggested_actions": ["建议的 action 名称列表"],
    "related_failures": [相关的失败事件索引]
  }}
]

只返回 JSON，不要其他文字。如果没有缺口，返回 []。"#,
            failures_json
        );

        let response = match self.llm.execute(&prompt, "auto", None).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("失败驱动: LLM 调用失败: {}", e);
                return vec![];
            }
        };

        match serde_json::from_str::<Vec<CapabilityGap>>(&response) {
            Ok(gaps) => gaps,
            Err(e) => {
                tracing::warn!("失败驱动: LLM 返回解析失败: {}", e);
                // 尝试提取 JSON 数组
                self.extract_json_array(&response)
                    .and_then(|s| serde_json::from_str::<Vec<CapabilityGap>>(&s).ok())
                    .unwrap_or_default()
            }
        }
    }

    /// 合成新能力基因组
    ///
    /// 用 LLM 根据缺口描述生成完整的能力基因组
    pub async fn synthesize_capability(
        &self,
        gap: &CapabilityGap,
    ) -> Option<CapabilityGenome> {
        let prompt = format!(
            r#"你是能力运行时的能力合成器。需要创建一个新能力来填补以下缺口：

缺口描述: {}
建议名称: {}
建议动作: {}

请生成一个完整的能力基因组 JSON，格式如下：
{{
  "name": "{}",
  "description": "能力描述",
  "category": "synthesized",
  "actions": [
    {{
      "name": "动作名",
      "description": "动作描述",
      "input_schema": {{
        "type": "object",
        "properties": {{
          "字段名": {{ "type": "string", "description": "字段描述" }}
        }},
        "required": ["字段名"]
      }},
      "implementation": {{
        "type": "script",
        "language": "python",
        "code": "import json, os\n# 实现代码\nprint(json.dumps({{\"success\": true, \"result\": ...}}))"
      }}
    }}
  ]
}}

要求：
1. 代码必须是可执行的 Python3
2. 输出必须是 JSON 且包含 success 字段
3. 代码要处理异常情况，失败时返回 {{"success": false, "error": "..."}}
4. 不要依赖外部 API 或网络（除非必要）
5. 只返回 JSON，不要其他文字"#,
            gap.description,
            gap.suggested_name,
            gap.suggested_actions.join(", "),
            gap.suggested_name,
        );

        let response = match self.llm.execute(&prompt, "auto", None).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("失败驱动: LLM 调用失败: {}", e);
                return None;
            }
        };

        // 尝试直接解析
        if let Ok(genome) = serde_json::from_str::<CapabilityGenome>(&response) {
            return Some(genome);
        }

        // 尝试提取 JSON 对象
        if let Some(json_str) = self.extract_json_object(&response) {
            if let Ok(genome) = serde_json::from_str::<CapabilityGenome>(&json_str) {
                return Some(genome);
            }
        }

        tracing::warn!("失败驱动: 能力合成失败 — LLM 返回无法解析");
        None
    }

    /// 验证新能力：沙箱测试 + A/B 测试
    pub async fn validate_capability(
        &self,
        genome: &CapabilityGenome,
        existing_genome: Option<&CapabilityGenome>,
        test_input: &serde_json::Value,
    ) -> ValidationResult {
        if genome.actions.is_empty() {
            return ValidationResult {
                passed: false,
                sandbox_result: None,
                ab_test_result: None,
                reason: "能力没有动作".into(),
            };
        }

        let action = &genome.actions[0];
        let code = match &action.implementation {
            ActionImpl::Script { code, .. } => code.as_str(),
            _ => {
                return ValidationResult {
                    passed: false,
                    sandbox_result: None,
                    ab_test_result: None,
                    reason: "能力实现不是脚本类型".into(),
                };
            }
        };

        // 1. 沙箱验证
        let sandbox_result = self.sandbox.execute_python(code, test_input).await;

        if !sandbox_result.success {
            let reason = format!(
                "沙箱验证失败: {}",
                sandbox_result.validation_errors.join("; ")
            );
            return ValidationResult {
                passed: false,
                sandbox_result: Some(sandbox_result),
                ab_test_result: None,
                reason,
            };
        }

        // 2. 对抗测试
        let adversarial_results = self
            .sandbox
            .adversarial_test(code, &action.input_schema)
            .await;
        let adv_failures = adversarial_results.iter().filter(|r| !r.success).count();
        if adv_failures > adversarial_results.len() / 2 {
            return ValidationResult {
                passed: false,
                sandbox_result: Some(sandbox_result),
                ab_test_result: None,
                reason: format!("对抗测试失败率过高: {}/{}", adv_failures, adversarial_results.len()),
            };
        }

        // 3. A/B 测试（如果存在旧版本）
        if let Some(old) = existing_genome {
            if !old.actions.is_empty() {
                if let ActionImpl::Script { code: old_code, .. } = &old.actions[0].implementation {
                    let test_suite = self
                        .ab_tester
                        .generate_test_suite(&action.input_schema, test_input);
                    let ab_result = self
                        .ab_tester
                        .run_test(old_code, code, &test_suite)
                        .await;

                    let passed = match &ab_result.recommendation {
                        ABTestRecommendation::Promote => true,
                        ABTestRecommendation::Keep => true,  // 新版本不比旧版差，可以保留
                        ABTestRecommendation::Rollback => false,
                        ABTestRecommendation::InsufficientData => true, // 数据不足时保守通过
                    };

                    return ValidationResult {
                        passed,
                        sandbox_result: Some(sandbox_result),
                        ab_test_result: Some(ab_result),
                        reason: if passed {
                            "通过验证".into()
                        } else {
                            "A/B 测试结果不如旧版本".into()
                        },
                    };
                }
            }
        }

        ValidationResult {
            passed: true,
            sandbox_result: Some(sandbox_result),
            ab_test_result: None,
            reason: "通过沙箱验证（无旧版本对比）".into(),
        }
    }

    /// 完整的失败驱动进化循环
    ///
    /// 分析失败 → 识别缺口 → 合成能力 → 验证 → 返回可注册的能力
    pub async fn evolve_from_failures(
        &mut self,
    ) -> Vec<EvolutionOutcome> {
        let mut outcomes = vec![];

        // 1. 分析缺口
        let gaps = self.analyze_gaps().await;
        tracing::info!("失败驱动: 识别到 {} 个能力缺口", gaps.len());

        for gap in &gaps {
            // 2. 合成能力
            if let Some(genome) = self.synthesize_capability(gap).await {
                // 3. 生成测试输入
                let test_input = self.generate_test_input_for(&genome);

                // 4. 验证
                let validation = self.validate_capability(&genome, None, &test_input).await;

                outcomes.push(EvolutionOutcome {
                    gap: gap.clone(),
                    genome,
                    validation,
                });
            }
        }

        // 清除已处理的失败
        self.failures.clear();

        outcomes
    }

    /// 为能力生成测试输入
    fn generate_test_input_for(&self, genome: &CapabilityGenome) -> serde_json::Value {
        if genome.actions.is_empty() {
            return serde_json::json!({});
        }

        let action = &genome.actions[0];
        let mut test = serde_json::Map::new();

        if let Some(props) = action.input_schema.get("properties").and_then(|p| p.as_object()) {
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
                        if hint.contains("host") {
                            serde_json::json!("127.0.0.1")
                        } else if hint.contains("url") || hint.contains("endpoint") {
                            serde_json::json!("https://httpbin.org/get")
                        } else if hint.contains("path") || hint.contains("file") {
                            serde_json::json!("/tmp")
                        } else if hint.contains("command") || hint.contains("cmd") {
                            serde_json::json!("echo hello")
                        } else if hint.contains("sql") || hint.contains("query") {
                            serde_json::json!("SELECT 1")
                        } else {
                            serde_json::json!("test")
                        }
                    }
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

    /// 从文本中提取 JSON 数组
    fn extract_json_array(&self, text: &str) -> Option<String> {
        let start = text.find('[')?;
        let end = text.rfind(']')?;
        if end > start {
            Some(text[start..=end].to_string())
        } else {
            None
        }
    }

    /// 从文本中提取 JSON 对象
    fn extract_json_object(&self, text: &str) -> Option<String> {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        if end > start {
            Some(text[start..=end].to_string())
        } else {
            None
        }
    }
}

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub sandbox_result: Option<crate::sandbox::SandboxResult>,
    pub ab_test_result: Option<ABTestResult>,
    pub reason: String,
}

/// 进化结果
#[derive(Debug, Clone)]
pub struct EvolutionOutcome {
    pub gap: CapabilityGap,
    pub genome: CapabilityGenome,
    pub validation: ValidationResult,
}

impl EvolutionOutcome {
    pub fn is_passing(&self) -> bool {
        self.validation.passed
    }
}
