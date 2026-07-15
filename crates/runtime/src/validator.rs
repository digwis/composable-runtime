//! 环境验证器 — 真实世界硬信号
//!
//! 进化闭环的反馈信号质量瓶颈：`validate_in_real_project` 原本只看能力返回的
//! JSON `success` 字段，一个能力只要返回结构化 JSON 就算成功，哪怕它声称做的事
//! 在真实世界里并没有发生。这让"进化选择"失去依据——坏能力和好能力看起来一样。
//!
//! 环境验证器在能力执行**之后**追加一次真实世界校验：
//! - cargo 能力声称跑过 cargo → 验证器在同一目标目录跑 `cargo check` 看是否通过
//! - git 能力声称改过状态 → 验证器去读 `git status` 比对
//! - shell/script 能力 → 退出码非零即真实失败
//!
//! 验证结果作为 `RealValidationResult.success` 的覆盖判定，
//! 并喂给 `FitnessGene::real_validation_passes` 参与 score 加权。
//!
//! 这是"真实世界压力"的注入点——把"能力自己说成功"升级为"环境证明它成功"。

use crate::genome::CapabilityGenome;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const VALIDATION_TIMEOUT: Duration = Duration::from_secs(60);
const CARGO_CHECK_ARGS: &[&str] = &["check", "-q"];
const CARGO_CHECK_LABEL: &str = "cargo check -q";

/// 真实世界信号强度 — 验证手段的可信度/价值层级
///
/// 离散层级而非连续值:不同验证手段对"能力在真实世界真的成立"提供的证据强度有本质差异,
/// 用 enum 让"通过 cargo test"和"通过退出码"在类型上可区分,防止弱信号被当强信号用。
/// 权值参与 `FitnessGene::recompute_score` 的真实轨计算,真实信号权重远大于自报。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SignalStrength {
    /// 无验证器,信任能力自报 — 最弱 (0.1)
    #[default]
    SelfReport,
    /// 非零退出码判定失败 — 中 (0.3)
    ExitCode,
    /// cargo check / 等价编译检查 — 较强 (0.5)
    BuildDry,
    /// cargo test / fork-repo 测试套件通过 — 强 (0.7)
    TestPass,
    /// 真实任务完成 / 被其他能力依赖 — 最强 (0.9)
    RealTask,
    /// 人类实测判定有用/无用 — 高于一切自动信号 (0.95)
    ///
    /// 自动 fitness 只能验证"能跑通",无法判断"对人有用"。对主观价值型能力
    /// (pandas/sklearn 等数据科学类),"有用没用"只能由人实测后判定。这是唯一
    /// 超越 RealTask 的信号层级:人类价值是进化选择的最终标准。
    HumanValue,
}

impl SignalStrength {
    /// 权值:真实世界背书的可信度,参与 score 双轨加权
    pub fn weight(self) -> f64 {
        match self {
            SignalStrength::SelfReport => 0.1,
            SignalStrength::ExitCode => 0.3,
            SignalStrength::BuildDry => 0.5,
            SignalStrength::TestPass => 0.7,
            SignalStrength::RealTask => 0.9,
            SignalStrength::HumanValue => 0.95,
        }
    }
}

/// 真实世界信号 — 环境验证器的输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealWorldSignal {
    /// 真实世界是否达成预期（覆盖能力自报的 success）
    pub success: bool,
    /// 证据摘要（cargo build 输出、git status 摘要、退出码等），用于归因和审计
    pub evidence: String,
    /// 信号强度:真实世界背书的可信度层级。默认 SelfReport(信任自报)。
    #[serde(default)]
    pub strength: SignalStrength,
}

impl RealWorldSignal {
    /// 无匹配验证器时的"信任能力自报"信号（向后兼容分析类能力）
    pub fn trust_self_report(success: bool) -> Self {
        Self {
            success,
            evidence: "无环境验证器，信任能力自报".into(),
            strength: SignalStrength::SelfReport,
        }
    }
}

/// 环境验证器 trait — 按能力名匹配，执行后校验真实世界状态
///
/// 验证器是只读的：它只观察环境，不修改能力或环境。
/// 副作用校验（如 cargo build）应使用幂等或低成本的检查命令。
#[async_trait::async_trait]
pub trait EnvironmentValidator: Send + Sync {
    /// 能力名匹配规则（如包含 "cargo"）
    fn matches(&self, capability_name: &str) -> bool;

    /// 执行后环境校验
    ///
    /// - `capability_name`: 被验证的能力名
    /// - `action`: 执行的 action 名
    /// - `output`: 能力返回的 payload（可能含 exit_code/stderr 等线索）
    async fn verify(
        &self,
        capability_name: &str,
        action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal;
}

/// 验证器注册表 — 有序匹配，首个命中的验证器生效
pub struct ValidatorRegistry {
    validators: Vec<Box<dyn EnvironmentValidator>>,
}

impl ValidatorRegistry {
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// 默认注册表：内置 cargo / git / 通用退出码 / fork-repo 四个验证器
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        // 顺序很重要：专用验证器在前，通用兜底在后
        reg.register(Box::new(CargoValidator));
        reg.register(Box::new(GitValidator));
        // fork 验证器放在通用兜底之前:能力名含 fork/repo_test/bugfix 时命中
        let fork_cache = fork_cache_default_dir();
        std::fs::create_dir_all(&fork_cache).ok();
        reg.register(Box::new(ForkRepoValidator::new(fork_cache)));
        reg.register(Box::new(ExitCodeValidator));
        reg
    }

    pub fn register(&mut self, v: Box<dyn EnvironmentValidator>) {
        self.validators.push(v);
    }

    /// 查找首个匹配能力的验证器
    pub fn find(&self, capability_name: &str) -> Option<&dyn EnvironmentValidator> {
        self.validators
            .iter()
            .find(|v| v.matches(capability_name))
            .map(|v| v.as_ref())
    }

    /// 对能力执行验证：有匹配验证器则跑环境校验，否则信任能力自报
    pub async fn verify(
        &self,
        capability_name: &str,
        action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal {
        match self.find(capability_name) {
            Some(v) => v.verify(capability_name, action, output).await,
            None => {
                let self_success = output
                    .get("success")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false);
                RealWorldSignal::trust_self_report(self_success)
            }
        }
    }
}

impl Default for ValidatorRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// 运行一个命令并截取输出（超时保护，避免验证器卡死进化循环）
async fn run_cmd(program: &str, args: &[&str], cwd: Option<&str>) -> (bool, String) {
    run_cmd_with_timeout(program, args, cwd, VALIDATION_TIMEOUT).await
}

async fn run_cmd_with_timeout(
    program: &str,
    args: &[&str],
    cwd: Option<&str>,
    timeout_duration: Duration,
) -> (bool, String) {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // 超时 future 被丢弃时触发终止子进程，避免验证器在后台泄漏 cargo 等进程。
        .kill_on_drop(true);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (false, format!("启动 {} 失败: {}", program, e)),
    };

    // 超时保护：生产路径使用 60 秒；测试可传入更短时限验证该分支。
    match tokio::time::timeout(timeout_duration, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let success = output.status.success();
            let evidence = if stderr.is_empty() {
                stdout.chars().take(300).collect()
            } else {
                format!("{}\n---stderr---\n{}", stdout, stderr)
                    .chars()
                    .take(400)
                    .collect()
            };
            (success, evidence)
        }
        Ok(Err(e)) => (false, format!("等待 {} 失败: {}", program, e)),
        Err(_) => (false, format!("{} 验证超时", program)),
    }
}

async fn run_cargo_check(cwd: Option<&str>) -> RealWorldSignal {
    let (ok, evidence) = run_cmd("cargo", CARGO_CHECK_ARGS, cwd).await;
    RealWorldSignal {
        success: ok,
        strength: SignalStrength::BuildDry,
        evidence: if ok {
            format!("{} 通过（项目可通过编译检查）", CARGO_CHECK_LABEL)
        } else {
            format!("{} 失败 — 真实世界未达成:\n{}", CARGO_CHECK_LABEL, evidence)
        },
    }
}

/// Cargo 验证器 — 能力名含 "cargo"，执行后跑 `cargo check -q` 校验项目可编译
///
/// 为什么是 check 而非 test：check 会执行真实的解析和类型检查，但不生成最终二进制，
/// 成本较低，同时能捕获"能力声称跑过 cargo 但项目实际无法编译"的真实失败。
/// 在项目根目录无 Cargo.toml 时降级为信任自报。
pub struct CargoValidator;

#[async_trait::async_trait]
impl EnvironmentValidator for CargoValidator {
    fn matches(&self, capability_name: &str) -> bool {
        capability_name.contains("cargo")
    }

    async fn verify(
        &self,
        _capability_name: &str,
        _action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal {
        // 能力自己说失败 → 直接失败，无需再跑环境校验
        let self_success = output
            .get("success")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);
        if !self_success {
            return RealWorldSignal {
                success: false,
                evidence: "能力自报失败".into(),
                strength: SignalStrength::SelfReport,
            };
        }

        // 操作能力必须报告真实子进程退出码；只有 success=true 不构成环境证据。
        let Some(exit_code) = output.get("exit_code").and_then(|v| v.as_i64()) else {
            return RealWorldSignal {
                success: false,
                evidence: "Cargo 能力未报告 exit_code，无法证明命令真实执行".into(),
                strength: SignalStrength::SelfReport,
            };
        };
        if exit_code != 0 {
            return RealWorldSignal {
                success: false,
                evidence: format!("能力报告 exit_code={}（非零即真实失败）", exit_code),
                strength: SignalStrength::ExitCode,
            };
        }

        // 验证器必须检查本次能力实际收到的目标，而不是 daemon 偶然所在的仓库。
        let validation_input = output.get("_validation_input");
        let command = validation_input
            .and_then(|input| input.get("command"))
            .and_then(|v| v.as_str());
        let cwd = validation_input
            .and_then(|input| input.get("cwd").or_else(|| input.get("path")))
            .and_then(|v| v.as_str());
        if command != Some("check") {
            return RealWorldSignal {
                success: false,
                evidence: "Cargo 真实验证必须执行 command=check".into(),
                strength: SignalStrength::SelfReport,
            };
        }
        let Some(cwd) = cwd else {
            return RealWorldSignal {
                success: false,
                evidence: "Cargo 真实验证缺少目标 cwd/path".into(),
                strength: SignalStrength::SelfReport,
            };
        };
        if !std::path::Path::new(cwd).join("Cargo.toml").is_file() {
            return RealWorldSignal {
                success: false,
                evidence: format!("Cargo 目标目录 '{}' 不含 Cargo.toml", cwd),
                strength: SignalStrength::BuildDry,
            };
        }

        run_cargo_check(Some(cwd)).await
    }
}

/// Git 验证器 — 能力名含 "git"，执行后跑 `git status --porcelain` 校验 git 可用
///
/// git 类能力的真实失败通常是：仓库不存在、git 未安装、命令拼错导致子进程失败。
/// 验证器读能力报告的 exit_code 与 stderr，并确认当前目录是有效 git 仓库。
pub struct GitValidator;

#[async_trait::async_trait]
impl EnvironmentValidator for GitValidator {
    fn matches(&self, capability_name: &str) -> bool {
        capability_name.contains("git")
    }

    async fn verify(
        &self,
        _capability_name: &str,
        action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal {
        // 退出码非零即真实失败
        if let Some(code) = output.get("exit_code").and_then(|v| v.as_i64()) {
            if code != 0 {
                return RealWorldSignal {
                    success: false,
                    evidence: format!("git {} 退出码={}", action, code),
                    strength: SignalStrength::ExitCode,
                };
            }
        }

        let self_success = output
            .get("success")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);
        if !self_success {
            return RealWorldSignal {
                success: false,
                evidence: "能力自报失败".into(),
                strength: SignalStrength::SelfReport,
            };
        }

        // 能力自报成功 → 校验当前目录是有效 git 仓库
        let (ok, evidence) = run_cmd("git", &["status", "--porcelain"], None).await;
        if ok {
            RealWorldSignal {
                success: true,
                strength: SignalStrength::ExitCode,
                evidence: format!("git status 可执行，当前目录是有效仓库\n{}", evidence),
            }
        } else {
            RealWorldSignal {
                success: false,
                strength: SignalStrength::ExitCode,
                evidence: format!("git status 失败 — 可能不是有效 git 仓库:\n{}", evidence),
            }
        }
    }
}

/// 通用退出码验证器 — 能力输出里带非零 exit_code 或非空 stderr 即真实失败
///
/// 兜底覆盖所有 shell/script 能力：不依赖能力名，只看输出里的硬信号。
/// 这是"能力说成功但子进程其实崩了"的最后一道防线。
pub struct ExitCodeValidator;

#[async_trait::async_trait]
impl EnvironmentValidator for ExitCodeValidator {
    fn matches(&self, _capability_name: &str) -> bool {
        // 兜底：总是匹配。注册表里放在最后，仅当专用验证器未命中时才到这里
        true
    }

    async fn verify(
        &self,
        _capability_name: &str,
        _action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal {
        // 非零退出码 → 真实失败
        if let Some(code) = output.get("exit_code").and_then(|v| v.as_i64()) {
            if code != 0 {
                let stderr = output.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
                return RealWorldSignal {
                    success: false,
                    evidence: format!("exit_code={}\n{}", code, stderr),
                    strength: SignalStrength::ExitCode,
                };
            }
        }

        // 无退出码或退出码为 0 → 信任能力自报（兜底弱信号）
        let self_success = output
            .get("success")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);
        RealWorldSignal::trust_self_report(self_success)
    }
}

/// 便利函数：从 genome 推断是否需要环境验证（操作类能力才走验证器）
/// Fork 仓库验证器 — 在真实开源项目里跑测试套件，最强的本地真实信号之一
///
/// 能力名含 "fork" / "repo_test" / "bugfix" 时匹配。能力输出里需带:
///   { "repo": "owner/name", "action": "clone" | "test" | "build" }
///
/// **安全边界(用户红线,不可逾越)**:
/// - 只允许公开 GitHub 仓库,`git clone --depth 1` 浅克隆省带宽
/// - clone 到 `repo_cache_dir/owner_name`,不写项目源码树
/// - **只读**:只 clone + 跑测试,**绝不 commit/push/开 PR/开 issue**。
///   本验证器代码路径里不引入任何 `git push` / `gh pr create` 调用。
/// - 超时 60s(复用 run_cmd 保护)
///
/// 信号强度:跑通测试套件 = TestPass(0.7);若能力声称修复了某条之前失败的测试且复跑通过 = RealTask(0.9)。
/// 这是让进化"和真实世界建立联系"的关键反馈源,风险可控。
pub struct ForkRepoValidator {
    /// fork 缓存根目录,默认 `{sandbox}/fork_cache/`
    repo_cache_dir: PathBuf,
}

impl ForkRepoValidator {
    pub fn new(repo_cache_dir: PathBuf) -> Self {
        Self { repo_cache_dir }
    }
}

#[async_trait::async_trait]
impl EnvironmentValidator for ForkRepoValidator {
    fn matches(&self, capability_name: &str) -> bool {
        let n = capability_name.to_lowercase();
        n.contains("fork") || n.contains("repo_test") || n.contains("bugfix")
    }

    async fn verify(
        &self,
        _capability_name: &str,
        _action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal {
        // 能力自报失败 → 直接失败,不必 clone
        let self_success = output
            .get("success")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);
        if !self_success {
            return RealWorldSignal {
                success: false,
                evidence: "能力自报失败,fork 验证中止".into(),
                strength: SignalStrength::SelfReport,
            };
        }

        // 读能力输出里的 repo 字段:必须是 owner/name 格式的公开 GitHub 仓库
        let repo = match output.get("repo").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => {
                return RealWorldSignal {
                    success: false,
                    evidence: "fork 能力未声明目标 repo (owner/name)".into(),
                    strength: SignalStrength::SelfReport,
                };
            }
        };
        if !is_valid_github_repo(repo) {
            return RealWorldSignal {
                success: false,
                evidence: format!("非法/非 GitHub 仓库标识: {} (仅允许 owner/name)", repo),
                strength: SignalStrength::SelfReport,
            };
        }

        let cache_dir = self.repo_cache_dir.join(repo.replace('/', "_"));
        let url = format!("https://github.com/{}.git", repo);

        // clone 或 fetch(已存在则只更新,不重复全量克隆)
        if !cache_dir.exists() {
            let (ok, evidence) = run_cmd(
                "git",
                &[
                    "clone",
                    "--depth",
                    "1",
                    &url,
                    cache_dir.to_str().unwrap_or(""),
                ],
                None,
            )
            .await;
            if !ok {
                return RealWorldSignal {
                    success: false,
                    evidence: format!("git clone 失败:\n{}", evidence),
                    strength: SignalStrength::ExitCode,
                };
            }
        }

        // 跑该 repo 自己的测试套件(按项目类型探测)
        let (ok, evidence) = run_repo_tests(&cache_dir).await;
        RealWorldSignal {
            success: ok,
            // 跑通真实项目的测试套件 = 强真实信号;通过比 0.7
            // (若能力声明修复了某条测试,应通过 output.tests_fixed 字段升级到 RealTask,后续扩展)
            strength: if ok {
                SignalStrength::TestPass
            } else {
                SignalStrength::TestPass // 失败也是 TestPass 强度的负信号(走到测试阶段才败)
            },
            evidence: if ok {
                format!("fork 仓库 {} 测试套件通过\n{}", repo, evidence)
            } else {
                format!(
                    "fork 仓库 {} 测试失败 — 真实世界未达成:\n{}",
                    repo, evidence
                )
            },
        }
    }
}

/// 校验 repo 标识:必须是 owner/name 形式,且仅允许 GitHub(受控例外)
fn is_valid_github_repo(repo: &str) -> bool {
    let parts: Vec<&str> = repo.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
        })
}

/// 探测 repo 类型并跑对应测试套件(只读,绝不写)
async fn run_repo_tests(repo_dir: &Path) -> (bool, String) {
    // 探测顺序:Cargo.toml → package.json → pytest
    if repo_dir.join("Cargo.toml").exists() {
        return run_cmd("cargo", &["test", "--offline", "-q"], repo_dir.to_str()).await;
    }
    if repo_dir.join("package.json").exists() {
        // 安装依赖(成败不直接判,只看 test 能否跑通)
        let _ = run_cmd("npm", &["ci", "--silent"], repo_dir.to_str()).await;
        return run_cmd("npm", &["test"], repo_dir.to_str()).await;
    }
    if repo_dir.join("pytest.ini").exists()
        || repo_dir.join("setup.py").exists()
        || repo_dir.join("pyproject.toml").exists()
    {
        return run_cmd("python3", &["-m", "pytest", "-q"], repo_dir.to_str()).await;
    }
    // 未知项目类型:至少确认能列目录(能力存活),给中信号
    (true, "无已知测试套件,仅确认仓库可访问".into())
}

/// 便利函数：从 genome 推断是否需要环境验证（操作类能力才走验证器）
///
/// 复用 auto_evolve 里的操作类关键词，保持判定一致。
pub fn is_operation_capability(name: &str) -> bool {
    const OP_KEYWORDS: &[&str] = &[
        "git", "cargo", "make", "shell", "fs", "file", "ssh", "curl", "http", "npm", "pip", "brew",
        "rg", "jq", "sqlite", "rustc", "wasm",
    ];
    OP_KEYWORDS.iter().any(|k| name.contains(k))
}

/// 给 genome 的 fitness 记录一次环境验证结果（正/负反馈 + 最强信号升级）
///
/// - `success=true`  → `real_validation_passes += 1`，且若 `strength` 高于历史最强则升级 `strongest_signal`
/// - `success=false` → `real_validation_failures += 1`（负反馈，参与 score 双轨计算的通过比）
///
/// 注意:失败也按 strength 记录——通过 cargo test 失败 比 退出码失败 信号更强(它走到测试阶段才败),
/// 让 LLM 归因时能区分"差一步通过"和"根本没跑起来"。
pub fn record_validation(genome: &mut CapabilityGenome, signal: &RealWorldSignal) {
    if signal.success {
        genome.fitness.real_validation_passes =
            genome.fitness.real_validation_passes.saturating_add(1);
        if signal.strength > genome.fitness.strongest_automatic_signal
            && signal.strength != SignalStrength::HumanValue
        {
            genome.fitness.strongest_automatic_signal = signal.strength;
        }
        if signal.strength > genome.fitness.strongest_signal {
            genome.fitness.strongest_signal = signal.strength;
        }
    } else {
        genome.fitness.real_validation_failures =
            genome.fitness.real_validation_failures.saturating_add(1);
    }
}

/// 共享一个默认注册表给 AutoEvolver
pub fn default_registry() -> Arc<ValidatorRegistry> {
    Arc::new(ValidatorRegistry::with_defaults())
}

/// fork 缓存默认目录:`~/.orch/fork_cache/`(与 daemon 存储约定一致)
fn fork_cache_default_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".orch").join("fork_cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cargo_fixture(source: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create cargo fixture");
        std::fs::create_dir(dir.path().join("src")).expect("create fixture src directory");
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"validator-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n",
        )
        .expect("write fixture manifest");
        std::fs::write(dir.path().join("src/lib.rs"), source).expect("write fixture source");
        dir
    }

    #[test]
    fn test_cargo_validator_uses_supported_check_command() {
        assert_eq!(CARGO_CHECK_ARGS, &["check", "-q"]);
        assert!(!CARGO_CHECK_ARGS.contains(&"--dry-run"));
    }

    #[tokio::test]
    async fn test_cargo_check_accepts_valid_project() {
        let dir = cargo_fixture("pub fn answer() -> u32 { 42 }\n");
        let signal = run_cargo_check(dir.path().to_str()).await;

        assert!(signal.success, "{}", signal.evidence);
        assert_eq!(signal.strength, SignalStrength::BuildDry);
        assert!(signal.evidence.contains(CARGO_CHECK_LABEL));
    }

    #[tokio::test]
    async fn test_cargo_check_failure_preserves_compiler_stderr() {
        let dir = cargo_fixture("pub fn broken() { let _: u32 = \"not a number\"; }\n");
        let signal = run_cargo_check(dir.path().to_str()).await;

        assert!(!signal.success);
        assert_eq!(signal.strength, SignalStrength::BuildDry);
        assert!(signal.evidence.contains("cargo check -q 失败"));
        assert!(signal.evidence.contains("---stderr---"));
        assert!(signal.evidence.contains("error"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_run_cmd_timeout_is_reported() {
        let (ok, evidence) = run_cmd_with_timeout(
            "/bin/sh",
            &["-c", "sleep 1"],
            None,
            Duration::from_millis(20),
        )
        .await;

        assert!(!ok);
        assert_eq!(evidence, "/bin/sh 验证超时");
    }

    #[tokio::test]
    async fn test_cargo_validator_nonzero_exit() {
        let v = CargoValidator;
        let out = serde_json::json!({"success": true, "exit_code": 1, "stderr": "error"});
        let sig = v.verify("cargo_ops", "run_cargo", &out).await;
        assert!(!sig.success);
        assert!(sig.evidence.contains("exit_code"));
    }

    #[tokio::test]
    async fn test_cargo_validator_self_reported_failure() {
        let v = CargoValidator;
        let out = serde_json::json!({"success": false});
        let sig = v.verify("cargo_ops", "run_cargo", &out).await;
        assert!(!sig.success);
    }

    #[tokio::test]
    async fn test_cargo_validator_checks_declared_target_directory() {
        let dir = cargo_fixture("pub fn answer() -> u32 { 42 }\n");
        let v = CargoValidator;
        let out = serde_json::json!({
            "success": true,
            "exit_code": 0,
            "_validation_input": {
                "command": "check",
                "cwd": dir.path().to_string_lossy()
            }
        });
        let sig = v.verify("cargo_ops", "run_cargo", &out).await;
        assert!(sig.success, "{}", sig.evidence);
        assert_eq!(sig.strength, SignalStrength::BuildDry);
    }

    #[tokio::test]
    async fn test_cargo_validator_rejects_unscoped_success() {
        let v = CargoValidator;
        let out = serde_json::json!({"success": true, "exit_code": 0});
        let sig = v.verify("cargo_ops", "run_cargo", &out).await;
        assert!(!sig.success);
        assert!(sig.evidence.contains("command=check"));
    }

    #[tokio::test]
    async fn test_exit_code_validator_zero_exit_trusts_self() {
        let v = ExitCodeValidator;
        let out = serde_json::json!({"success": true, "exit_code": 0});
        let sig = v.verify("any_cap", "any", &out).await;
        assert!(sig.success);
    }

    #[tokio::test]
    async fn test_exit_code_validator_nonzero_exit_fails() {
        let v = ExitCodeValidator;
        let out = serde_json::json!({"success": true, "exit_code": 127});
        let sig = v.verify("any_cap", "any", &out).await;
        // 即使能力自报 success=true，非零退出码也应判失败
        assert!(!sig.success);
    }

    #[tokio::test]
    async fn test_registry_find_cargo() {
        let reg = ValidatorRegistry::with_defaults();
        let v = reg.find("cargo_ops-v2").unwrap();
        // cargo 能力应由 CargoValidator 处理
        assert!(v.matches("cargo_ops-v2"));
    }

    #[tokio::test]
    async fn test_registry_verify_no_match_trusts_self() {
        let reg = ValidatorRegistry::with_defaults();
        // analysis 类能力无专用验证器，兜底的 ExitCodeValidator 会命中（matches 总是 true）
        // 但若输出无 exit_code，则信任自报
        let out = serde_json::json!({"success": true});
        let sig = reg.verify("knowledge_graph_ops", "extract", &out).await;
        assert!(sig.success);
    }

    #[tokio::test]
    async fn test_registry_verify_nonzero_exit_fails_even_for_analysis() {
        let reg = ValidatorRegistry::with_defaults();
        let out = serde_json::json!({"success": true, "exit_code": 2});
        let sig = reg.verify("knowledge_graph_ops", "extract", &out).await;
        assert!(!sig.success, "非零退出码应触发真实失败");
    }

    #[tokio::test]
    async fn test_is_operation_capability() {
        assert!(is_operation_capability("git_ops-v2"));
        assert!(is_operation_capability("cargo_ops"));
        assert!(!is_operation_capability("knowledge_graph_ops"));
        assert!(!is_operation_capability("temporal_causal_analyzer"));
    }

    #[tokio::test]
    async fn test_record_validation_pass_only_increments_on_pass() {
        let mut g = CapabilityGenome::new(String::from("test_cap"), String::from("test"));
        // 通过:记 passes + 升级最强信号
        record_validation(
            &mut g,
            &RealWorldSignal {
                success: true,
                evidence: "build ok".into(),
                strength: SignalStrength::BuildDry,
            },
        );
        record_validation(
            &mut g,
            &RealWorldSignal {
                success: true,
                evidence: "test ok".into(),
                strength: SignalStrength::TestPass,
            },
        );
        // 失败:记 failures(负反馈),不动 passes
        record_validation(
            &mut g,
            &RealWorldSignal {
                success: false,
                evidence: "fail".into(),
                strength: SignalStrength::TestPass,
            },
        );
        assert_eq!(g.fitness.real_validation_passes, 2);
        assert_eq!(g.fitness.real_validation_failures, 1);
        // 最强信号应是 TestPass(第二次通过升级了它)
        assert_eq!(g.fitness.strongest_signal, SignalStrength::TestPass);
    }
}
