//! Portable capability package format.
//!
//! The manifest is language-neutral; Python is the default runtime because it
//! is fast to evolve and has broad automation/data tooling. Packages remain
//! versionable in Git and can be validated before local registration.

use crate::sandbox::{Sandbox, SandboxConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CapabilityRuntime {
    Python,
    Rust,
    Shell,
    Typescript,
    Wasm,
}

impl Default for CapabilityRuntime {
    fn default() -> Self {
        Self::Python
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPermissions {
    #[serde(default = "default_filesystem_permission")]
    pub filesystem: String,
    #[serde(default)]
    pub network: Vec<String>,
    #[serde(default)]
    pub commands: Vec<String>,
}

impl Default for CapabilityPermissions {
    fn default() -> Self {
        Self {
            filesystem: default_filesystem_permission(),
            network: Vec::new(),
            commands: Vec::new(),
        }
    }
}

fn default_filesystem_permission() -> String {
    "read".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPackageManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub version: String,
    pub description: String,
    #[serde(default)]
    pub runtime: CapabilityRuntime,
    pub entrypoint: String,
    #[serde(default = "empty_object")]
    pub input_schema: Value,
    #[serde(default = "empty_object")]
    pub output_schema: Value,
    #[serde(default)]
    pub permissions: CapabilityPermissions,
    #[serde(default)]
    pub tests: Vec<String>,
    #[serde(default)]
    pub evals: Vec<String>,
    #[serde(default)]
    pub source_repository: Option<String>,
}

fn default_schema_version() -> u32 {
    1
}

fn empty_object() -> Value {
    serde_json::json!({"type": "object"})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPackage {
    pub root: PathBuf,
    pub manifest: CapabilityPackageManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPackageOutput {
    pub success: bool,
    pub output: Value,
    pub stderr: String,
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub isolation: String,
    #[serde(default)]
    pub worker_pid: Option<u32>,
    #[serde(default)]
    pub sandbox_backend: String,
}

impl CapabilityPackage {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref();
        let manifest_path = root.join("capability.yaml");
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|error| format!("读取 {} 失败: {}", manifest_path.display(), error))?;
        let manifest = serde_yaml::from_str::<CapabilityPackageManifest>(&raw)
            .map_err(|error| format!("能力包 Manifest 解析失败: {}", error))?;
        let package = Self {
            root: root.to_path_buf(),
            manifest,
        };
        package.validate()?;
        Ok(package)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.manifest.schema_version != 1 {
            return Err(format!(
                "不支持能力包 schema_version {}",
                self.manifest.schema_version
            ));
        }
        if self.manifest.id.is_empty()
            || !self
                .manifest
                .id
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        {
            return Err("能力 ID 只能包含 ASCII 字母、数字、-、_、.".into());
        }
        if self.manifest.version.trim().is_empty() {
            return Err("能力版本不能为空".into());
        }
        let entrypoint = Path::new(&self.manifest.entrypoint);
        if entrypoint.is_absolute()
            || entrypoint
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err("能力入口必须位于能力包目录内".into());
        }
        if !self.root.join(entrypoint).is_file() {
            return Err(format!("能力入口不存在: {}", self.manifest.entrypoint));
        }
        if !matches!(
            self.manifest.permissions.filesystem.as_str(),
            "none" | "read" | "write"
        ) {
            return Err("filesystem 权限必须是 none、read 或 write".into());
        }
        Ok(())
    }

    pub async fn execute_python(
        &self,
        input: &Value,
        timeout_secs: u64,
    ) -> Result<CapabilityPackageOutput, String> {
        if self.manifest.runtime != CapabilityRuntime::Python {
            return Err("该能力包不是 Python runtime".into());
        }
        self.validate()?;
        let code = std::fs::read_to_string(self.root.join(&self.manifest.entrypoint))
            .map_err(|error| format!("读取能力入口失败: {}", error))?;
        let mut config = SandboxConfig::default();
        config.timeout = Duration::from_secs(timeout_secs.max(1));
        config.allow_network = !self.manifest.permissions.network.is_empty();
        if self.manifest.permissions.filesystem == "write" {
            config.allowed_paths.push(self.root.clone());
        }
        let mut env = HashMap::new();
        env.insert("ORCH_CAPABILITY_PACKAGE".into(), self.manifest.id.clone());
        let stdin = format!(
            "{}\n",
            serde_json::to_string(input).map_err(|error| error.to_string())?
        );
        let result = Sandbox::new(config)
            .execute_script_with_io(
                "python",
                &code,
                input,
                env,
                Some(stdin),
                Some(self.root.clone()),
                false,
            )
            .await;
        let stdout = result.stdout.trim().to_string();
        let stderr: String = result.stderr.chars().take(4000).collect();
        let value = if stdout.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&stdout)
                .map_err(|error| format!("Python 能力 stdout 不是合法 JSON: {}", error))?
        };
        Ok(CapabilityPackageOutput {
            success: result.success,
            output: value,
            stderr,
            exit_code: result.exit_code,
            isolation: result.isolation,
            worker_pid: result.worker_pid,
            sandbox_backend: result.sandbox_backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_manifest_is_language_neutral_and_python_by_default() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::write(directory.path().join("src/main.py"), "print('{}')\n").unwrap();
        std::fs::write(
            directory.path().join("capability.yaml"),
            "id: project-audit\nversion: 1.0.0\ndescription: audit\nentrypoint: src/main.py\n",
        )
        .unwrap();
        let package = CapabilityPackage::load(directory.path()).unwrap();
        assert_eq!(package.manifest.runtime, CapabilityRuntime::Python);
    }

    #[test]
    fn package_rejects_entrypoint_escape() {
        let package = CapabilityPackage {
            root: PathBuf::from("/tmp/package"),
            manifest: CapabilityPackageManifest {
                schema_version: 1,
                id: "bad".into(),
                version: "1".into(),
                description: "bad".into(),
                runtime: CapabilityRuntime::Python,
                entrypoint: "../escape.py".into(),
                input_schema: empty_object(),
                output_schema: empty_object(),
                permissions: CapabilityPermissions::default(),
                tests: Vec::new(),
                evals: Vec::new(),
                source_repository: None,
            },
        };
        assert!(package.validate().is_err());
    }

    #[tokio::test]
    async fn package_python_uses_unified_sandbox_runtime() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("main.py"),
            "import json, sys\npayload=json.loads(sys.stdin.readline())\nprint(json.dumps({'seen': payload.get('value')}))\n",
        )
        .unwrap();
        std::fs::write(
            directory.path().join("capability.yaml"),
            "id: sandboxed-package\nversion: 1.0.0\ndescription: test\nentrypoint: main.py\n",
        )
        .unwrap();
        let package = CapabilityPackage::load(directory.path()).unwrap();
        let output = package
            .execute_python(&serde_json::json!({"value": 42}), 5)
            .await
            .unwrap();
        assert!(output.success);
        assert_eq!(output.output["seen"], 42);
        if cfg!(target_os = "macos") {
            assert_eq!(output.sandbox_backend, "macos_sandbox_exec");
        }
    }
}
