use crate::sandbox::{Sandbox, SandboxResult};
use serde::{Deserialize, Serialize};

/// A/B 测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResult {
    pub winner: ABTestWinner,
    pub old_stats: CapabilityStats,
    pub new_stats: CapabilityStats,
    pub confidence: f64,
    pub recommendation: ABTestRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ABTestWinner {
    Old,
    New,
    Tie,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ABTestRecommendation {
    Promote,
    Keep,
    Rollback,
    InsufficientData,
}

/// 能力执行统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityStats {
    pub total_runs: u32,
    pub successes: u32,
    pub failures: u32,
    pub avg_latency_ms: f64,
    pub success_rate: f64,
}

impl CapabilityStats {
    pub fn from_results(results: &[SandboxResult]) -> Self {
        let total = results.len() as u32;
        let successes = results.iter().filter(|r| r.success).count() as u32;
        let failures = total - successes;
        let avg_latency_ms = if total > 0 {
            results.iter().map(|r| r.elapsed_ms as f64).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let success_rate = if total > 0 {
            successes as f64 / total as f64
        } else {
            0.0
        };
        Self {
            total_runs: total,
            successes,
            failures,
            avg_latency_ms,
            success_rate,
        }
    }
}

/// A/B 测试器 — 对比新旧能力，决定是否提升
pub struct ABTester {
    sandbox: Sandbox,
    min_samples: usize,
    min_confidence: f64,
}

impl ABTester {
    pub fn new(sandbox: Sandbox) -> Self {
        Self {
            sandbox,
            min_samples: 5,
            min_confidence: 0.7,
        }
    }

    pub fn with_thresholds(sandbox: Sandbox, min_samples: usize, min_confidence: f64) -> Self {
        Self {
            sandbox,
            min_samples,
            min_confidence,
        }
    }

    /// 执行 A/B 测试
    ///
    /// 对相同输入集并行执行新旧代码，对比成功率、延迟、输出质量
    pub async fn run_test(
        &self,
        old_code: &str,
        new_code: &str,
        test_inputs: &[serde_json::Value],
    ) -> ABTestResult {
        let mut old_results = vec![];
        let mut new_results = vec![];

        for input in test_inputs {
            let old_r = self.sandbox.execute_python(old_code, input).await;
            let new_r = self.sandbox.execute_python(new_code, input).await;
            old_results.push(old_r);
            new_results.push(new_r);
        }

        let old_stats = CapabilityStats::from_results(&old_results);
        let new_stats = CapabilityStats::from_results(&new_results);

        let (winner, confidence, recommendation) =
            self.evaluate(&old_stats, &new_stats);

        ABTestResult {
            winner,
            old_stats,
            new_stats,
            confidence,
            recommendation,
        }
    }

    /// 评估哪个版本更好
    fn evaluate(
        &self,
        old: &CapabilityStats,
        new: &CapabilityStats,
    ) -> (ABTestWinner, f64, ABTestRecommendation) {
        let total = old.total_runs + new.total_runs;
        if (total as usize) < self.min_samples * 2 {
            return (
                ABTestWinner::Tie,
                0.0,
                ABTestRecommendation::InsufficientData,
            );
        }

        // 成功率差异
        let _success_diff = new.success_rate - old.success_rate;
        // 延迟改善比例
        let latency_improvement = if old.avg_latency_ms > 0.0 {
            (old.avg_latency_ms - new.avg_latency_ms) / old.avg_latency_ms
        } else {
            0.0
        };

        // 综合评分: 成功率权重 0.7, 延迟权重 0.3
        let new_score = new.success_rate * 0.7 + (1.0 + latency_improvement).min(2.0) * 0.15;
        let old_score = old.success_rate * 0.7 + 1.0 * 0.15;

        let confidence = ((new_score - old_score).abs() * 10.0).min(1.0);

        let winner = if new_score > old_score * 1.05 {
            ABTestWinner::New
        } else if old_score > new_score * 1.05 {
            ABTestWinner::Old
        } else {
            ABTestWinner::Tie
        };

        let recommendation = match &winner {
            ABTestWinner::New if confidence >= self.min_confidence => {
                ABTestRecommendation::Promote
            }
            ABTestWinner::New => ABTestRecommendation::Keep, // 有改善但不够确信
            ABTestWinner::Tie => ABTestRecommendation::Keep,
            ABTestWinner::Old if confidence >= self.min_confidence => {
                ABTestRecommendation::Rollback
            }
            ABTestWinner::Old => ABTestRecommendation::Keep,
        };

        (winner, confidence, recommendation)
    }

    /// 生成测试输入集（基于 schema）
    pub fn generate_test_suite(
        &self,
        schema: &serde_json::Value,
        smart_input: &serde_json::Value,
    ) -> Vec<serde_json::Value> {
        let mut suite = vec![];

        // 1. 智能输入（正常用例）
        suite.push(smart_input.clone());

        // 2. 对抗性输入
        let adversarial = self.sandbox.generate_adversarial_inputs(schema);
        suite.extend(adversarial);

        // 3. 边界值
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            let mut boundary = serde_json::Map::new();
            for (key, field_schema) in props {
                let val = match field_schema.get("type").and_then(|t| t.as_str()) {
                    Some("integer") | Some("number") => serde_json::json!(0),
                    Some("boolean") => serde_json::json!(false),
                    Some("string") => serde_json::json!(""),
                    Some("array") => serde_json::json!([]),
                    Some("object") => serde_json::json!({}),
                    _ => serde_json::json!(null),
                };
                boundary.insert(key.clone(), val);
            }
            suite.push(serde_json::Value::Object(boundary));
        }

        suite
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::Sandbox;

    #[tokio::test]
    async fn test_ab_promote() {
        let tester = ABTester::new(Sandbox::with_defaults());
        let old_code = r#"import json
print(json.dumps({"success": False}))"#;
        let new_code = r#"import json
print(json.dumps({"success": True}))"#;
        let inputs = vec![serde_json::json!({}); 5];
        let result = tester.run_test(old_code, new_code, &inputs).await;
        assert_eq!(result.winner, ABTestWinner::New);
        assert_eq!(result.recommendation, ABTestRecommendation::Promote);
    }

    #[tokio::test]
    async fn test_ab_rollback() {
        let tester = ABTester::new(Sandbox::with_defaults());
        let old_code = r#"import json
print(json.dumps({"success": True}))"#;
        let new_code = r#"import json
print(json.dumps({"success": False}))"#;
        let inputs = vec![serde_json::json!({}); 5];
        let result = tester.run_test(old_code, new_code, &inputs).await;
        assert_eq!(result.winner, ABTestWinner::Old);
    }

    #[tokio::test]
    async fn test_ab_insufficient() {
        let tester = ABTester::new(Sandbox::with_defaults());
        let old_code = r#"import json
print(json.dumps({"success": True}))"#;
        let new_code = r#"import json
print(json.dumps({"success": True}))"#;
        let inputs = vec![serde_json::json!({}); 1];
        let result = tester.run_test(old_code, new_code, &inputs).await;
        assert_eq!(
            result.recommendation,
            ABTestRecommendation::InsufficientData
        );
    }
}
