use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
    #[serde(default)]
    pub isolation: String,
    #[serde(default)]
    pub worker_pid: Option<u32>,
    #[serde(default)]
    pub sandbox_backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CapabilityWorkerRequest {
    language: String,
    code: String,
    input: serde_json::Value,
    timeout_ms: u64,
    env_vars: HashMap<String, String>,
    forbidden_commands: Vec<String>,
    allowed_paths: Vec<PathBuf>,
    allow_network: bool,
    require_json_success: bool,
    stdin_data: Option<String>,
    working_dir: Option<PathBuf>,
}

/// 路径守卫 — 硬约束 AI agent 的文件操作范围
///
/// 当 `allowed_paths` 非空时，所有写操作（write/mkdir/delete/move）
/// 的目标路径必须落在某个 allowed_path 之下。
/// 空列表表示不限制（向后兼容）。
#[derive(Debug, Clone, Default)]
pub struct PathGuard {
    /// 允许写入的根目录列表（原始路径，用于直接比较）
    raw_roots: Vec<PathBuf>,
    /// 允许写入的根目录列表（canonicalized，用于符号链接解析后比较）
    canonical_roots: Vec<PathBuf>,
}

impl PathGuard {
    /// 创建无限制的 PathGuard（默认行为，向后兼容）
    pub fn unrestricted() -> Self {
        Self {
            raw_roots: vec![],
            canonical_roots: vec![],
        }
    }

    /// 创建受限制的 PathGuard，只允许在指定根目录下操作
    pub fn new(allowed_paths: &[PathBuf]) -> Self {
        let mut raw = vec![];
        let mut canonical = vec![];
        for p in allowed_paths {
            raw.push(p.clone());
            match p.canonicalize() {
                Ok(c) => canonical.push(c),
                Err(_) => {
                    // 路径可能还不存在，用父目录 canonicalize 再拼接
                    if let Some(parent) = p.parent() {
                        if let Ok(c) = parent.canonicalize() {
                            canonical.push(c.join(p.file_name().unwrap_or_default()));
                        }
                    }
                }
            }
        }
        Self {
            raw_roots: raw,
            canonical_roots: canonical,
        }
    }

    /// 检查路径是否在允许范围内
    pub fn check(&self, path: &str) -> Result<(), String> {
        if self.raw_roots.is_empty() {
            return Ok(()); // 无限制模式
        }

        let target = Path::new(path);

        // 1. 用原始路径直接比较（快速路径，处理符号链接不一致问题）
        for root in &self.raw_roots {
            if target.starts_with(root) {
                return Ok(());
            }
        }

        // 2. 用 canonicalized 路径比较（处理符号链接和 . / .. 解析）
        let canonical = match target.canonicalize() {
            Ok(c) => c,
            Err(_) => {
                // 文件可能不存在（新建场景），尝试 canonicalize 父目录
                if let Some(parent) = target.parent() {
                    match parent.canonicalize() {
                        Ok(c) => c.join(target.file_name().unwrap_or_default()),
                        Err(_) => target.to_path_buf(),
                    }
                } else {
                    target.to_path_buf()
                }
            }
        };

        for root in &self.canonical_roots {
            if canonical.starts_with(root) {
                return Ok(());
            }
        }

        Err(format!(
            "路径 '{}' 不在允许的写入范围内。允许的目录: {:?}",
            path, self.raw_roots
        ))
    }

    /// 是否处于限制模式
    pub fn is_restricted(&self) -> bool {
        !self.raw_roots.is_empty()
    }

    /// 获取允许的根目录列表
    pub fn allowed_roots(&self) -> &[PathBuf] {
        &self.raw_roots
    }
}

/// 沙箱配置
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub timeout: Duration,
    pub max_memory_mb: Option<u64>,
    pub env_whitelist: Vec<String>,
    pub forbidden_commands: Vec<String>,
    /// 进化能力默认不需要访问网络；需要时必须由调用方显式开启。
    pub allow_network: bool,
    /// AI agent 允许写入的路径白名单
    /// 空列表 = 不限制（默认）
    /// 非空 = 只允许在这些目录下创建/修改/删除文件
    pub allowed_paths: Vec<PathBuf>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_memory_mb: Some(256),
            env_whitelist: vec![
                "PATH".into(),
                "HOME".into(),
                "TMPDIR".into(),
                "LANG".into(),
                "LC_ALL".into(),
                "__EXECUTOR_INPUT__".into(),
            ],
            forbidden_commands: vec![
                "rm -rf /".into(),
                "mkfs".into(),
                "dd if=".into(),
                "shutdown".into(),
                "reboot".into(),
            ],
            allow_network: false,
            allowed_paths: vec![],
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
        check_code_safety(code, &self.config.forbidden_commands)
    }

    /// 在独立 Capability Worker 中执行 Python；非 orch 测试二进制回退到直接子进程。
    pub async fn execute_python(&self, code: &str, input: &serde_json::Value) -> SandboxResult {
        self.execute_script_with_io("python", code, input, HashMap::new(), None, None, true)
            .await
    }

    /// 通过统一 Capability Worker 执行 Python、Node 或 Shell 脚本。
    pub async fn execute_script(
        &self,
        language: &str,
        code: &str,
        input: &serde_json::Value,
        extra_env: HashMap<String, String>,
    ) -> SandboxResult {
        self.execute_script_with_io(language, code, input, extra_env, None, None, false)
            .await
    }

    /// 带 stdin 和工作目录的统一脚本入口，供能力包等文件型能力使用。
    pub async fn execute_script_with_io(
        &self,
        language: &str,
        code: &str,
        input: &serde_json::Value,
        extra_env: HashMap<String, String>,
        stdin_data: Option<String>,
        working_dir: Option<PathBuf>,
        require_json_success: bool,
    ) -> SandboxResult {
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
                isolation: "rejected_before_execution".into(),
                worker_pid: None,
                sandbox_backend: "not_started".into(),
            };
        }

        let language = match language.trim().to_ascii_lowercase().as_str() {
            "python" | "py" => "python",
            "node" | "js" | "javascript" => "node",
            "shell" | "sh" | "bash" => "shell",
            other => {
                return validation_error(
                    format!("不支持的沙箱脚本语言: {}", other),
                    Instant::now(),
                    "rejected_before_execution",
                    None,
                )
            }
        };

        let mut env_vars = self.filtered_environment(input);
        env_vars.extend(extra_env);

        let request = CapabilityWorkerRequest {
            language: language.into(),
            code: code.to_string(),
            input: input.clone(),
            timeout_ms: self.config.timeout.as_millis().min(u64::MAX as u128) as u64,
            env_vars,
            forbidden_commands: self.config.forbidden_commands.clone(),
            allowed_paths: self.config.allowed_paths.clone(),
            allow_network: self.config.allow_network,
            require_json_success,
            stdin_data,
            working_dir,
        };
        if let Some(worker) = capability_worker_executable() {
            return execute_via_capability_worker(&worker, &request).await;
        }
        execute_python_process(&request, "python_subprocess", None).await
    }

    fn filtered_environment(&self, input: &serde_json::Value) -> HashMap<String, String> {
        let mut env_vars = HashMap::new();
        for key in &self.config.env_whitelist {
            if let Ok(value) = std::env::var(key) {
                env_vars.insert(key.clone(), value);
            }
        }
        env_vars.insert(
            "__EXECUTOR_INPUT__".into(),
            serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
        );
        env_vars.insert("PYTHONDONTWRITEBYTECODE".into(), "1".into());
        env_vars.insert(
            "ORCH_INPUT".into(),
            serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
        );
        env_vars
    }

    /*
     * The remaining methods generate adversarial inputs and intentionally use
     * execute_python, so every evolved capability follows the same boundary.
     */

    /// 对抗测试：用边界输入验证能力健壮性
    pub async fn adversarial_test(
        &self,
        code: &str,
        schema: &serde_json::Value,
    ) -> Vec<SandboxResult> {
        let inputs = self.generate_adversarial_inputs(schema);
        let mut results = vec![];
        for input in &inputs {
            let result = self.execute_python(code, input).await;
            results.push(result);
        }
        results
    }

    /// 生成对抗性测试输入
    pub fn generate_adversarial_inputs(
        &self,
        schema: &serde_json::Value,
    ) -> Vec<serde_json::Value> {
        let mut inputs = vec![];
        inputs.push(serde_json::json!({}));
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            let mut empty_obj = serde_json::Map::new();
            for (key, _) in props {
                empty_obj.insert(key.clone(), serde_json::Value::String("".into()));
            }
            inputs.push(serde_json::Value::Object(empty_obj));
            let mut null_obj = serde_json::Map::new();
            for (key, _) in props {
                null_obj.insert(key.clone(), serde_json::Value::Null);
            }
            inputs.push(serde_json::Value::Object(null_obj));
            let mut long_obj = serde_json::Map::new();
            for (key, _) in props {
                long_obj.insert(key.clone(), serde_json::Value::String("A".repeat(10000)));
            }
            inputs.push(serde_json::Value::Object(long_obj));
        }
        inputs
    }
}

fn check_code_safety(code: &str, forbidden_commands: &[String]) -> Vec<String> {
    let mut errors = vec![];
    for forbidden in forbidden_commands {
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

fn capability_worker_executable() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ORCH_CAPABILITY_WORKER_BIN") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    let executable = std::env::current_exe().ok()?;
    (executable.file_stem().and_then(|name| name.to_str()) == Some("orch")).then_some(executable)
}

async fn execute_via_capability_worker(
    executable: &Path,
    request: &CapabilityWorkerRequest,
) -> SandboxResult {
    let start = Instant::now();
    let mut child = match Command::new(executable)
        .arg("capability-worker")
        .kill_on_drop(true)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return process_error(
                "启动 Capability Worker 失败",
                error,
                start,
                "worker_process",
                None,
            )
        }
    };
    let worker_pid = child.id();
    let payload = match serde_json::to_vec(request) {
        Ok(payload) => payload,
        Err(error) => {
            return validation_error(
                format!("序列化 Worker 请求失败: {}", error),
                start,
                "worker_process",
                worker_pid,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(error) = stdin.write_all(&payload).await {
            return process_error(
                "写入 Capability Worker 失败",
                error,
                start,
                "worker_process",
                worker_pid,
            );
        }
    }
    let timeout =
        Duration::from_millis(request.timeout_ms.max(1)).saturating_add(Duration::from_secs(2));
    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return process_error(
                "Capability Worker 执行错误",
                error,
                start,
                "worker_process",
                worker_pid,
            )
        }
        Err(_) => return timeout_error(timeout, start, "worker_process", worker_pid),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    match serde_json::from_str::<SandboxResult>(stdout.trim()) {
        Ok(mut result) => {
            result.isolation = "worker_process".into();
            result.worker_pid = worker_pid;
            result.elapsed_ms = start.elapsed().as_millis() as u64;
            result
        }
        Err(error) => validation_error(
            format!(
                "Capability Worker 返回无效结果: {}; stderr: {}",
                error,
                String::from_utf8_lossy(&output.stderr)
            ),
            start,
            "worker_process",
            worker_pid,
        ),
    }
}

async fn execute_python_process(
    request: &CapabilityWorkerRequest,
    isolation: &str,
    worker_pid: Option<u32>,
) -> SandboxResult {
    let start = Instant::now();
    let safety_errors = check_code_safety(&request.code, &request.forbidden_commands);
    if !safety_errors.is_empty() {
        return SandboxResult {
            success: false,
            stdout: String::new(),
            stderr: safety_errors.join("\n"),
            exit_code: None,
            elapsed_ms: 0,
            timed_out: false,
            validation_errors: safety_errors,
            isolation: isolation.into(),
            worker_pid,
            sandbox_backend: "not_started".into(),
        };
    }
    let (mut command, sandbox_backend) = script_command(request);
    command.kill_on_drop(true);
    command.env_clear();
    for (key, value) in &request.env_vars {
        command.env(key, value);
    }
    if let Some(working_dir) = &request.working_dir {
        command.current_dir(working_dir);
    }
    if request.stdin_data.is_some() {
        command.stdin(std::process::Stdio::piped());
    }
    command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return process_error("启动 Python 失败", error, start, isolation, worker_pid)
        }
    };
    if let Some(data) = request.stdin_data.as_deref() {
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(error) = stdin.write_all(data.as_bytes()).await {
                return process_error("写入脚本 stdin 失败", error, start, isolation, worker_pid);
            }
        }
    }
    let timeout = Duration::from_millis(request.timeout_ms.max(1));
    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return process_error("Python 执行错误", error, start, isolation, worker_pid)
        }
        Err(_) => return timeout_error(timeout, start, isolation, worker_pid),
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code();
    let mut validation_errors = vec![];
    let process_succeeded = exit_code == Some(0);
    let parsed_json = serde_json::from_str::<serde_json::Value>(stdout.trim());
    let success = if !process_succeeded {
        false
    } else if request.require_json_success {
        match &parsed_json {
            Ok(value) => value
                .get("success")
                .and_then(|value| value.as_bool())
                .unwrap_or_else(|| {
                    validation_errors.push("输出缺少 success 字段".into());
                    false
                }),
            Err(_) => {
                validation_errors.push("输出不是有效 JSON".into());
                false
            }
        }
    } else if parsed_json.as_ref().is_ok_and(json_output_reports_failure) {
        validation_errors.push("输出 JSON 明确报告失败".into());
        false
    } else {
        true
    };
    SandboxResult {
        success,
        stdout,
        stderr,
        exit_code,
        elapsed_ms: start.elapsed().as_millis() as u64,
        timed_out: false,
        validation_errors,
        isolation: isolation.into(),
        worker_pid,
        sandbox_backend,
    }
}

fn script_command(request: &CapabilityWorkerRequest) -> (Command, String) {
    let input_arg = serde_json::to_string(&request.input).unwrap_or_else(|_| "{}".into());
    let (program, args): (&str, Vec<String>) = match request.language.as_str() {
        "python" => (
            "python3",
            vec!["-c".into(), request.code.clone(), input_arg.clone()],
        ),
        "node" => (
            "node",
            vec!["-e".into(), request.code.clone(), input_arg.clone()],
        ),
        "shell" => (
            "bash",
            vec!["-c".into(), request.code.clone(), "--".into(), input_arg],
        ),
        _ => unreachable!("language validated before worker dispatch"),
    };
    let os_sandbox_disabled = std::env::var("ORCH_OS_SANDBOX")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false"
            )
        })
        .unwrap_or(false);
    if cfg!(target_os = "macos")
        && !os_sandbox_disabled
        && Path::new("/usr/bin/sandbox-exec").is_file()
    {
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command
            .arg("-p")
            .arg(macos_sandbox_profile(request))
            .arg(program)
            .args(&args);
        return (command, "macos_sandbox_exec".into());
    }
    let mut command = Command::new(program);
    command.args(&args);
    (command, "process_only".into())
}

fn macos_sandbox_profile(request: &CapabilityWorkerRequest) -> String {
    let mut profile = String::from("(version 1)\n(allow default)\n");
    if !request.allow_network {
        profile.push_str("(deny network*)\n");
    }
    profile.push_str(
        "(deny file-write*)\n(allow file-write* (literal \"/dev/null\") (subpath \"/private/tmp\")",
    );
    for path in &request.allowed_paths {
        let normalized = path.canonicalize().unwrap_or_else(|_| path.clone());
        profile.push_str(&format!(
            " (subpath \"{}\")",
            escape_sandbox_literal(&normalized.to_string_lossy())
        ));
    }
    profile.push_str(")\n");
    profile
}

fn json_output_reports_failure(value: &serde_json::Value) -> bool {
    if value.get("success").and_then(|item| item.as_bool()) == Some(false) {
        return true;
    }
    if let Some(error) = value.get("error") {
        let present = match error {
            serde_json::Value::Null => false,
            serde_json::Value::String(message) => !message.trim().is_empty(),
            _ => true,
        };
        if present {
            return true;
        }
    }
    value
        .get("status")
        .and_then(|item| item.as_str())
        .is_some_and(|status| {
            matches!(
                status.to_ascii_lowercase().as_str(),
                "error" | "failed" | "failure"
            )
        })
}

fn escape_sandbox_literal(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn process_error(
    prefix: &str,
    error: std::io::Error,
    start: Instant,
    isolation: &str,
    worker_pid: Option<u32>,
) -> SandboxResult {
    validation_error(
        format!("{}: {}", prefix, error),
        start,
        isolation,
        worker_pid,
    )
}

fn validation_error(
    message: String,
    start: Instant,
    isolation: &str,
    worker_pid: Option<u32>,
) -> SandboxResult {
    SandboxResult {
        success: false,
        stdout: String::new(),
        stderr: message.clone(),
        exit_code: None,
        elapsed_ms: start.elapsed().as_millis() as u64,
        timed_out: false,
        validation_errors: vec![message],
        isolation: isolation.into(),
        worker_pid,
        sandbox_backend: "not_started".into(),
    }
}

fn timeout_error(
    timeout: Duration,
    start: Instant,
    isolation: &str,
    worker_pid: Option<u32>,
) -> SandboxResult {
    SandboxResult {
        success: false,
        stdout: String::new(),
        stderr: format!("超时 ({}ms)", timeout.as_millis()),
        exit_code: None,
        elapsed_ms: start.elapsed().as_millis() as u64,
        timed_out: true,
        validation_errors: vec!["timeout".into()],
        isolation: isolation.into(),
        worker_pid,
        sandbox_backend: "not_started".into(),
    }
}

/// Hidden CLI entrypoint used by `orch capability-worker`.
pub async fn run_capability_worker_stdio() -> Result<(), String> {
    let mut payload = Vec::new();
    tokio::io::stdin()
        .read_to_end(&mut payload)
        .await
        .map_err(|error| error.to_string())?;
    let request: CapabilityWorkerRequest = serde_json::from_slice(&payload)
        .map_err(|error| format!("解析 Worker 请求失败: {}", error))?;
    let result = execute_python_process(
        &request,
        "worker_python_subprocess",
        Some(std::process::id()),
    )
    .await;
    let output = serde_json::to_vec(&result).map_err(|error| error.to_string())?;
    tokio::io::stdout()
        .write_all(&output)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_guard_unrestricted() {
        let guard = PathGuard::unrestricted();
        assert!(!guard.is_restricted());
        assert!(guard.check("/anywhere/random").is_ok());
    }

    #[test]
    fn test_path_guard_restricted_allowed() {
        let tmp = std::env::temp_dir();
        let guard = PathGuard::new(&[tmp.clone()]);
        assert!(guard.is_restricted());
        let allowed = tmp.join("subdir/file.txt");
        assert!(guard.check(allowed.to_str().unwrap()).is_ok());
    }

    #[test]
    fn test_path_guard_restricted_blocked() {
        let tmp = std::env::temp_dir();
        let guard = PathGuard::new(&[tmp.join("safe_zone")]);
        // 创建 safe_zone 以确保 canonicalize 成功
        std::fs::create_dir_all(tmp.join("safe_zone")).ok();
        assert!(guard.check("/etc/passwd").is_err());
        assert!(guard.check("/Users/hacker/evil").is_err());
    }

    #[tokio::test]
    async fn test_sandbox_success() {
        let sandbox = Sandbox::with_defaults();
        let code = r#"import json
print(json.dumps({"success": True, "result": 42}))"#;
        let result = sandbox.execute_python(code, &serde_json::json!({})).await;
        assert!(result.success);
        assert!(result.elapsed_ms < 5000);
        assert_eq!(result.isolation, "python_subprocess");
        if cfg!(target_os = "macos") {
            assert_eq!(result.sandbox_backend, "macos_sandbox_exec");
        }
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

    #[tokio::test]
    async fn non_strict_script_rejects_explicit_json_error() {
        let sandbox = Sandbox::with_defaults();
        let result = sandbox
            .execute_script(
                "python",
                "import json; print(json.dumps({'error': 'missing input'}))",
                &serde_json::json!({}),
                HashMap::new(),
            )
            .await;
        assert!(!result.success);
        assert!(result
            .validation_errors
            .iter()
            .any(|message| message.contains("明确报告失败")));
    }

    #[tokio::test]
    async fn python_legacy_argv_input_remains_compatible() {
        let sandbox = Sandbox::with_defaults();
        let result = sandbox
            .execute_script(
                "python",
                "import json, sys; print(json.dumps({'received': json.loads(sys.argv[1])['value']}))",
                &serde_json::json!({"value": 42}),
                HashMap::new(),
            )
            .await;
        assert!(result.success, "{}", result.stderr);
        assert!(result.stdout.contains("42"));
    }

    #[test]
    fn macos_profile_denies_network_and_unlisted_writes() {
        let request = CapabilityWorkerRequest {
            language: "python".into(),
            code: String::new(),
            input: serde_json::Value::Null,
            timeout_ms: 1000,
            env_vars: HashMap::new(),
            forbidden_commands: vec![],
            allowed_paths: vec![PathBuf::from("/tmp/allowed")],
            allow_network: false,
            require_json_success: true,
            stdin_data: None,
            working_dir: None,
        };
        let profile = macos_sandbox_profile(&request);
        assert!(profile.contains("(deny network*)"));
        assert!(profile.contains("(deny file-write*)"));
        assert!(profile.contains("/dev/null"));
        assert!(profile.contains("/tmp/allowed"));
    }

    #[test]
    fn macos_profile_can_explicitly_allow_network() {
        let request = CapabilityWorkerRequest {
            language: "python".into(),
            code: String::new(),
            input: serde_json::Value::Null,
            timeout_ms: 1000,
            env_vars: HashMap::new(),
            forbidden_commands: vec![],
            allowed_paths: vec![],
            allow_network: true,
            require_json_success: true,
            stdin_data: None,
            working_dir: None,
        };
        assert!(!macos_sandbox_profile(&request).contains("(deny network*)"));
    }
}
