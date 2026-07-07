use serde::{Deserialize, Serialize};

/// 运行平台信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Platform {
    /// 操作系统
    pub os: String,
    /// CPU 架构
    pub arch: String,
    /// 平台标识（用于能力基因组匹配）
    pub id: String,
    /// 是否支持进程 spawn
    pub supports_process: bool,
    /// 是否支持文件系统
    pub supports_fs: bool,
    /// 是否支持网络
    pub supports_network: bool,
    /// 平台特定的环境信息
    pub env: std::collections::HashMap<String, String>,
}

impl Platform {
    /// 检测当前运行平台
    pub fn detect() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        let (id, supports_process, supports_fs, supports_network) = match os.as_str() {
            "macos" => ("macos".to_string(), true, true, true),
            "linux" => ("linux".to_string(), true, true, true),
            "windows" => ("windows".to_string(), true, true, true),
            "android" => ("android".to_string(), false, true, true),
            "ios" => ("ios".to_string(), false, true, true),
            _ => ("unknown".to_string(), false, false, false),
        };

        let mut env = std::collections::HashMap::new();
        env.insert("os".to_string(), os.clone());
        env.insert("arch".to_string(), arch.clone());

        if let Ok(home) = std::env::var("HOME") {
            env.insert("home".to_string(), home);
        }
        if let Ok(user) = std::env::var("USER") {
            env.insert("user".to_string(), user);
        }
        if let Ok(shell) = std::env::var("SHELL") {
            env.insert("shell".to_string(), shell);
        }

        // 检测可用的运行时和工具
        for tool in &[
            "python3", "node", "git", "docker",
            "curl", "wget", "sqlite3", "jq", "ffmpeg",
            "rg", "fd", "fzf", "tmux", "ssh",
            "make", "cmake", "rustc", "cargo",
            "brew", "pip3", "npm",
            "wasmtime",
        ] {
            if which(tool) {
                env.insert(format!("has_{}", tool), "true".to_string());
            }
        }

        // 检测 rustc 的 wasm32-wasi target（兼容新旧名称）
        if which("rustc") {
            let has_wasi = std::process::Command::new("rustc")
                .args(&["--print", "target-list"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .any(|line| line == "wasm32-wasi" || line == "wasm32-wasip1")
                })
                .unwrap_or(false);
            if has_wasi {
                env.insert("has_wasm32_wasi".to_string(), "true".to_string());
            }
        }

        // 检测可用的 Python 包
        if which("python3") {
            for pkg in &["numpy", "pandas", "requests", "matplotlib", "sympy", "networkx", "sklearn"] {
                let check = std::process::Command::new("python3")
                    .args(&["-c", &format!("import {}", pkg)])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if check {
                    env.insert(format!("has_py_{}", pkg), "true".to_string());
                }
            }
        }

        Self {
            os,
            arch,
            id,
            supports_process,
            supports_fs,
            supports_network,
            env,
        }
    }

    /// 平台描述（给 LLM 看）
    pub fn describe(&self) -> String {
        let mut desc = format!(
            "运行平台: {} ({})\n",
            self.os, self.arch
        );
        desc.push_str(&format!(
            "  文件系统: {} | 进程: {} | 网络: {}\n",
            if self.supports_fs { "✅" } else { "❌" },
            if self.supports_process { "✅" } else { "❌" },
            if self.supports_network { "✅" } else { "❌" }
        ));

        let tools: Vec<&str> = self.env.iter()
            .filter(|(k, _)| k.starts_with("has_"))
            .filter(|(_, v)| v.as_str() == "true")
            .map(|(k, _)| k.strip_prefix("has_").unwrap_or(k))
            .collect();

        if !tools.is_empty() {
            desc.push_str(&format!("  可用工具: {}\n", tools.join(", ")));
        }

        desc
    }

    /// 判断能力基因组是否兼容当前平台
    pub fn is_compatible(&self, genome: &crate::genome::CapabilityGenome) -> bool {
        for action in &genome.actions {
            match &action.implementation {
                crate::genome::ActionImpl::Composite { steps } => {
                    // 组合能力：检查所有子步骤是否兼容
                    // 这里只做简单检查，实际兼容性在运行时验证
                    if steps.is_empty() {
                        return false;
                    }
                }
                crate::genome::ActionImpl::Native { .. } => {
                    // 原生委托需要进程支持
                    if !self.supports_process {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    /// 获取平台特定的存储目录
    pub fn storage_dir(&self) -> String {
        let base = self.env.get("home").cloned().unwrap_or_else(|| ".".to_string());
        format!("{}/.evolution", base)
    }
}

fn which(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
