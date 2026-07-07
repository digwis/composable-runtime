use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// 沙箱验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub elapsed_ms: u64,
    pub timed_out: bool,
    pub validation_errors: Vec<String>,
}

/// 沙箱配置
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub timeout: Duration,
    pub max_memory_mb: Option<u64>,
    pub env_whitelist: Vec<String>,
    pub forbidden_commands: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_memory_mb: Some(256),
            env_whitelist: vec![
                "PATH".into(), "HOME".into(), "TMPDIR".into(),
                "LANG".into(), "LC_ALL".into(),
                "__EXECUTOR_INPUT__".into(),
            ],
            forbidden_commands: vec![
                "rm -rf /".into(), "mkfs".into(), "dd if=".into(),
                "shutdown".into(), "reboot".into(),
            ],
        }
    }
}

/// 沙箱执行器 — 隔离环境下验证能力代码
#[derive(Clone)]
pub struct Sandbox {
    config: SandboxConfig,
}

impl Sandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SandboxConfig::default())
    }

    /// 安全检查：代码是否包含危险操作
    pub fn check_safety(&self, code: &str) -> Vec<String> {
        let mut errors = vec![];
        for forbidden in &self.config.forbidden_commands {
            if code.contains(forbidden) {
                errors.push(format!("禁止的操作: {}", forbidden));
            }
        }
        // Python 特有检查
        if code.contains("os.system(") && !code.contains("subprocess") {
            errors.push("建议使用 subprocess 替代 os.system".into());
        }
        if code.contains("__import__('os')") {
            errors.push("动态导入 os 模块可能不安全".into());
        }
        errors
    }

    /// 在沙箱中执行 Python 脚本
    pub async fn execute_python(
        &self,
        code: &str,
        input: &serde_json::Value,
    ) -> SandboxResult {
        // 安全检查
        let safety_errors = self.check_safety(code);
        if !safety_errors.is_empty() {
            return SandboxResult {
                success: false,
                stdout: String::new(),
                stderr: safety_errors.join("\n"),
                exit_code: None,
                elapsed_ms: 0,
                timed_out: false,
                validation_errors: safety_errors,
            };
        }

        // 构建环境变量
        let mut env_vars: HashMap<String, String> = HashMap::new();
        for key in &self.config.env_whitelist {
            if let Ok(val) = std::env::var(key) {
                env_vars.insert(key.clone(), val);
            }
        }
        let input_str = serde_json::to_string(input).unwrap_or_else(|_| "{}".into());
        env_vars.insert("__EXECUTOR_INPUT__".into(), input_str);

        let start = Instant::now();

        // 执行
        let mut cmd = Command::new("python3");
        cmd.arg("-c").arg(code);
        for (k, v) in &env_vars {
            cmd.env(k, v);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return SandboxResult {
                    success: false,
                    stdout: String::new(),
                    stderr: format!("启动失败: {}", e),
                    exit_code: None,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    timed_out: false,
                    validation_errors: vec![format!("spawn error: {}", e)],
                };
            }
        };

        let output = match tokio::time::timeout(
            self.config.timeout,
            child.wait_with_output(),
        ).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return SandboxResult {
                    success: false,
                    stdout: String::new(),
                    stderr: format!("执行错误: {}", e),
                    exit_code: None,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    timed_out: false,
                    validation_errors: vec![e.to_string()],
                };
            }
            Err(_) => {
                return SandboxResult {
                    success: false,
                    stdout: String::new(),
                    stderr: format!("超时 ({}s)", self.config.timeout.as_secs()),
                    exit_code: None,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    timed_out: true,
                    validation_errors: vec!["timeout".into()],
                };
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code();
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // 验证输出是否为有效 JSON 且包含 success 字段
        let mut validation_errors = vec![];
        let success = if exit_code == Some(0) {
            match serde_json::from_str::<serde_json::Value>(&stdout.trim()) {
                Ok(v) => {
                    if let Some(s) = v.get("success").and_then(|s| s.as_bool()) {
                        s
                    } else {
                        validation_errors.push("输出缺少 success 字段".into());
                        false
                    }
                }
                Err(_) => {
                    validation_errors.push("输出不是有效 JSON".into());
                    false
                }
            }
        } else {
            false
        };

        SandboxResult {
            success,
            stdout,
            stderr,
            exit_code,
            elapsed_ms,
            timed_out: false,
            validation_errors,
        }
    }

    /// 对抗测试：用边界输入验证能力健壮性
    pub async fn adversarial_test(
        &self,
        code: &str,
        schema: &serde_json::Value,
    ) -> Vec<SandboxResult> {
        let inputs = self.generate_adversarial_inputs(schema);
        let mut results = vec![];
        for input in &inputs {
            let r = self.execute_python(code, input).await;
            results.push(r);
        }
        results
    }

    /// 生成对抗性测试输入
    pub fn generate_adversarial_inputs(&self, schema: &serde_json::Value) -> Vec<serde_json::Value> {
        let mut inputs = vec![];

        // 空输入
        inputs.push(serde_json::json!({}));

        // 各字段为空字符串
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            let mut empty_obj = serde_json::Map::new();
            for (key, _) in props {
                empty_obj.insert(key.clone(), serde_json::Value::String("".into()));
            }
            inputs.push(serde_json::Value::Object(empty_obj));

            // 各字段为 null
            let mut null_obj = serde_json::Map::new();
            for (key, _) in props {
                null_obj.insert(key.clone(), serde_json::Value::Null);
            }
            inputs.push(serde_json::Value::Object(null_obj));

            // 超长字符串
            let mut long_obj = serde_json::Map::new();
            for (key, _) in props {
                long_obj.insert(key.clone(), serde_json::Value::String("A".repeat(10000)));
            }
            inputs.push(serde_json::Value::Object(long_obj));
        }

        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sandbox_success() {
        let sandbox = Sandbox::with_defaults();
        let code = r#"import json
print(json.dumps({"success": True, "result": 42}))"#;
        let result = sandbox.execute_python(code, &serde_json::json!({})).await;
        assert!(result.success);
        assert!(result.elapsed_ms < 5000);
    }

    #[tokio::test]
    async fn test_sandbox_timeout() {
        let mut config = SandboxConfig::default();
        config.timeout = Duration::from_millis(100);
        let sandbox = Sandbox::new(config);
        let code = r#"import time
time.sleep(10)"#;
        let result = sandbox.execute_python(code, &serde_json::json!({})).await;
        assert!(result.timed_out);
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_sandbox_safety() {
        let sandbox = Sandbox::with_defaults();
        let errors = sandbox.check_safety("os.system('rm -rf /')");
        assert!(!errors.is_empty());
    }

    #[tokio::test]
    async fn test_sandbox_missing_success() {
        let sandbox = Sandbox::with_defaults();
        let code = r#"print("hello")"#;
        let result = sandbox.execute_python(code, &serde_json::json!({})).await;
        assert!(!result.success);
        assert!(!result.validation_errors.is_empty());
    }
}
