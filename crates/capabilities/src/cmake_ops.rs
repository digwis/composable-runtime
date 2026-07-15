use runtime::{Capability, Message, MessageError, MessageResult};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// CMake 能力 — 执行 CMake 构建操作
///
/// 基于 CMake 的构建系统操作能力，支持配置、构建、清洁、测试和安装。提供可跨平台编译C/C++项目的标准化接口，
/// 是构建自动化和包管理的基础。
pub struct CMakeOpsCapability;

#[derive(Deserialize)]
struct CMakeConfigureInput {
    source_dir: String,
    build_dir: String,
    #[serde(default)]
    generator: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
struct CMakeBuildInput {
    build_dir: String,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    parallel: Option<u32>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
struct CMakeCleanInput {
    build_dir: String,
    #[serde(default)]
    preserve_cache: bool,
}

#[derive(Deserialize)]
struct CMakeTestInput {
    build_dir: String,
    #[serde(default)]
    ctest_args: Vec<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Deserialize)]
struct CMakeInstallInput {
    build_dir: String,
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Serialize)]
struct CMakeOutput {
    command: String,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    success: bool,
}

#[async_trait::async_trait]
impl Capability for CMakeOpsCapability {
    fn name(&self) -> &str {
        "cmake_ops"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn actions(&self) -> Vec<&str> {
        vec!["configure", "build", "clean", "test", "install"]
    }

    fn describe(&self) -> String {
        "CMake 构建操作能力，支持配置、构建、清洁、测试和安装C/C++项目".to_string()
    }

    fn is_native(&self) -> bool {
        true
    }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "configure" => {
                let input: CMakeConfigureInput = msg.payload_as()?;
                let mut cmd = Command::new("cmake");
                cmd.args(&["-S", &input.source_dir, "-B", &input.build_dir]);

                if let Some(generator) = &input.generator {
                    cmd.args(&["-G", generator]);
                }

                cmd.args(&input.args);
                cmd.stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());

                let result = if let Some(timeout) = input.timeout_secs {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout),
                        run_cmake_command(cmd),
                    )
                    .await
                    .map_err(|_| MessageError::Internal {
                        capability: "cmake_ops".into(),
                        detail: format!("配置超时 ({}s)", timeout),
                    })?
                } else {
                    run_cmake_command(cmd).await
                };

                let (stdout, stderr, exit_code, success) = result?;

                let output = CMakeOutput {
                    command: format!("cmake -S {} -B {}", input.source_dir, input.build_dir),
                    stdout,
                    stderr,
                    exit_code,
                    success,
                };

                Ok(Message::builder()
                    .from("cmake_ops")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("configure.response")
                    .payload(serde_json::to_value(&output).unwrap_or_default())
                    .build())
            }

            "build" => {
                let input: CMakeBuildInput = msg.payload_as()?;
                let mut cmd = Command::new("cmake");
                cmd.args(&["--build", &input.build_dir]);

                if let Some(target) = &input.target {
                    cmd.args(["--target", target]);
                }

                if let Some(parallel) = input.parallel {
                    cmd.args(["--parallel", &parallel.to_string()]);
                }

                cmd.args(&input.args);
                cmd.stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());

                let result = if let Some(timeout) = input.timeout_secs {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout),
                        run_cmake_command(cmd),
                    )
                    .await
                    .map_err(|_| MessageError::Internal {
                        capability: "cmake_ops".into(),
                        detail: format!("构建超时 ({}s)", timeout),
                    })?
                } else {
                    run_cmake_command(cmd).await
                };

                let (stdout, stderr, exit_code, success) = result?;

                let output = CMakeOutput {
                    command: format!("cmake --build {}", input.build_dir),
                    stdout,
                    stderr,
                    exit_code,
                    success,
                };

                Ok(Message::builder()
                    .from("cmake_ops")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("build.response")
                    .payload(serde_json::to_value(&output).unwrap_or_default())
                    .build())
            }

            "clean" => {
                let input: CMakeCleanInput = msg.payload_as()?;
                let mut cmd = Command::new("cmake");
                cmd.args(&["-E", "rmrf", &input.build_dir]);

                cmd.stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());

                let result = run_cmake_command(cmd).await?;
                let (stdout, stderr, exit_code, success) = result;

                Ok(Message::builder()
                    .from("cmake_ops")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("clean.response")
                    .payload(serde_json::json!({
                        "command": format!("cmake -E rmrf {}", input.build_dir),
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "success": success,
                        "preserved_cache": input.preserve_cache,
                    }))
                    .build())
            }

            "test" => {
                let input: CMakeTestInput = msg.payload_as()?;
                let mut cmd = Command::new("ctest");
                cmd.args([&input.build_dir, "--output-on-failure"]);

                cmd.args(&input.ctest_args);
                cmd.stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());

                let result = if let Some(timeout) = input.timeout_secs {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout),
                        run_cmake_command(cmd),
                    )
                    .await
                    .map_err(|_| MessageError::Internal {
                        capability: "cmake_ops".into(),
                        detail: format!("测试超时 ({}s)", timeout),
                    })?
                } else {
                    run_cmake_command(cmd).await
                };

                let (stdout, stderr, exit_code, success) = result?;

                let output = CMakeOutput {
                    command: format!("ctest --output-on-failure {}", input.build_dir),
                    stdout,
                    stderr,
                    exit_code,
                    success,
                };

                Ok(Message::builder()
                    .from("cmake_ops")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("test.response")
                    .payload(serde_json::to_value(&output).unwrap_or_default())
                    .build())
            }

            "install" => {
                let input: CMakeInstallInput = msg.payload_as()?;
                let mut cmd = Command::new("cmake");
                cmd.args(&["--install", &input.build_dir]);

                if let Some(prefix) = &input.prefix {
                    cmd.args(["--prefix", prefix]);
                }

                cmd.args(&input.args);
                cmd.stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());

                let result = if let Some(timeout) = input.timeout_secs {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout),
                        run_cmake_command(cmd),
                    )
                    .await
                    .map_err(|_| MessageError::Internal {
                        capability: "cmake_ops".into(),
                        detail: format!("安装超时 ({}s)", timeout),
                    })?
                } else {
                    run_cmake_command(cmd).await
                };

                let (stdout, stderr, exit_code, success) = result?;

                let output = CMakeOutput {
                    command: format!("cmake --install {}", input.build_dir),
                    stdout,
                    stderr,
                    exit_code,
                    success,
                };

                Ok(Message::builder()
                    .from("cmake_ops")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("install.response")
                    .payload(serde_json::to_value(&output).unwrap_or_default())
                    .build())
            }

            _ => Err(MessageError::UnsupportedAction {
                capability: "cmake_ops".into(),
                action: msg.action.clone(),
            }),
        }
    }
}

async fn run_cmake_command(
    mut cmd: Command,
) -> Result<(String, String, Option<i32>, bool), MessageError> {
    let child = cmd.spawn().map_err(|e| MessageError::Internal {
        capability: "cmake_ops".into(),
        detail: format!("启动 {} 失败: {}", "cmake".to_string(), e),
    })?;

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| MessageError::Internal {
            capability: "cmake_ops".into(),
            detail: format!("等待失败: {}", e),
        })?;

    Ok((
        String::from_utf8_lossy(&result.stdout).to_string(),
        String::from_utf8_lossy(&result.stderr).to_string(),
        result.status.code(),
        result.status.success(),
    ))
}
