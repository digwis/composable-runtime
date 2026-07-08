use crate::evolution::EvolutionEngine;
use crate::genome::{ActionImpl, LlmExecutor};
use crate::message_bus::MessageBus;
use crate::platform::Platform;
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// 运行时规范基因组（RuntimeSpec）— 元基因
///
/// 这是"进化的进化规则"：描述运行时本身支持哪些执行器类型，
/// 以及每种执行器的参数 schema。
///
/// 当系统发现现有 5 种 ActionImpl 不足以表达某种执行方式时，
/// MetaEvolver 可以创造新的执行器类型并注册到 ExecutorRegistry，
/// 同时更新 RuntimeSpec 记录这一元进化事件。
///
/// 生物学类比：ActionImpl 的 5 种类型是"遗传密码"（像 A/T/C/G 四个碱基），
/// RuntimeSpec 是"密码表"（tRNA），ExecutorRegistry 是"翻译系统"。
/// 元进化 = 进化密码表本身。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSpec {
    /// 版本
    pub version: String,
    /// 内置执行器类型（编译期固定，不可元进化）
    pub builtin_executors: Vec<String>,
    /// 动态注册的执行器类型（元进化产物）
    #[serde(default)]
    pub custom_executors: Vec<CustomExecutorSpec>,
    /// 元进化历史
    #[serde(default)]
    pub meta_history: Vec<MetaEvolutionEvent>,
    /// 元进化统计
    #[serde(default)]
    pub meta_stats: MetaEvolveStats,
}

/// 自定义执行器规范 — 描述一个动态注册的执行器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomExecutorSpec {
    /// 执行器类型名（唯一标识，如 "wasm", "cached_script", "pipeline"）
    pub type_name: String,
    /// 人类可读描述
    pub description: String,
    /// 参数 schema（JSON Schema 风格，描述 implementation 对象需要哪些字段）
    pub params_schema: serde_json::Value,
    /// 执行器实现（Python 脚本）
    ///
    /// 执行器脚本接收一个 JSON 对象作为输入，包含：
    /// - params: 该执行器的参数（来自 ActionImpl::Custom 的 params）
    /// - input: 动作的输入参数
    /// - context: 运行时上下文（能力名、动作名等）
    ///
    /// 执行器脚本必须输出 JSON 对象到 stdout，包含 success 字段。
    pub executor_code: String,
    /// 执行器脚本语言
    pub language: String,
    /// 执行超时（秒）
    #[serde(default = "default_executor_timeout")]
    pub timeout_secs: u64,
    /// 创建时间
    pub created_at: String,
    /// 谱系（元进化也记录谱系）
    #[serde(default)]
    pub lineage: ExecutorLineage,
}

/// 执行器谱系
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutorLineage {
    /// 来源：原生 / 元进化生成 / 元进化变异
    #[serde(default)]
    pub origin: ExecutorOrigin,
    /// 父代执行器名（变异时）
    #[serde(default)]
    pub parent: Option<String>,
    /// 变异代数
    #[serde(default)]
    pub generation: u32,
}

/// 执行器来源
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ExecutorOrigin {
    #[default]
    Native,
    MetaGenerated,
    MetaMutated,
    #[serde(other)]
    Other,
}

/// 元进化事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaEvolutionEvent {
    pub event_type: String,
    pub executor_name: String,
    pub description: String,
    pub timestamp: String,
}

/// 元进化统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaEvolveStats {
    /// 元自省次数
    pub meta_introspections: u32,
    /// 执行器创造次数
    pub executors_created: u32,
    /// 执行器变异次数
    pub executors_mutated: u32,
    /// 执行器淘汰次数
    pub executors_eliminated: u32,
    /// 使用自定义执行器的能力数
    pub capabilities_using_custom: u32,
}

fn default_executor_timeout() -> u64 {
    60
}

impl RuntimeSpec {
    /// 创建初始运行时规范（仅包含内置执行器）
    pub fn initial() -> Self {
        Self {
            version: "0.1.0".into(),
            builtin_executors: vec![
                "Llm".into(),
                "Rule".into(),
                "Composite".into(),
                "Native".into(),
                "Script".into(),
            ],
            custom_executors: Vec::new(),
            meta_history: Vec::new(),
            meta_stats: MetaEvolveStats::default(),
        }
    }

    /// 注册自定义执行器
    pub fn register_executor(&mut self, spec: CustomExecutorSpec) {
        let name = spec.type_name.clone();
        // 替换同名执行器
        self.custom_executors.retain(|e| e.type_name != name);
        self.custom_executors.push(spec);
        self.meta_history.push(MetaEvolutionEvent {
            event_type: "executor_register".into(),
            executor_name: name,
            description: "注册自定义执行器".into(),
            timestamp: now_string(),
        });
    }

    /// 获取自定义执行器规范
    pub fn get_executor(&self, type_name: &str) -> Option<&CustomExecutorSpec> {
        self.custom_executors
            .iter()
            .find(|e| e.type_name == type_name)
    }

    /// 列出所有执行器类型（内置 + 自定义）
    pub fn all_executor_types(&self) -> Vec<String> {
        let mut all = self.builtin_executors.clone();
        all.extend(self.custom_executors.iter().map(|e| e.type_name.clone()));
        all
    }

    /// 序列化为 JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// 从 JSON 加载
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// 执行器注册表 — 运行时执行器的动态注册中心
///
/// 内置执行器（Llm/Rule/Composite/Native/Script）由 ScriptedCapability 硬编码处理。
/// 自定义执行器通过 ExecutorRegistry 动态注册和执行。
///
/// 执行器是一个 Python 脚本，接收统一的输入格式，输出统一的输出格式。
/// 这使得系统可以在运行时"发明"新的执行方式，而不需要修改 Rust 代码。
pub struct ExecutorRegistry {
    /// 执行器规范（可序列化的部分）
    spec: RwLock<RuntimeSpec>,
    /// 持久化路径
    storage_path: std::path::PathBuf,
    /// 原生插件缓存：type_name → (源码哈希, 已加载动态库)
    native_plugins: Mutex<HashMap<String, NativePlugin>>,
}

impl ExecutorRegistry {
    /// 创建执行器注册表
    pub fn new(storage_dir: impl Into<std::path::PathBuf>) -> Self {
        let storage_dir = storage_dir.into();
        let storage_path = storage_dir.join("runtime_spec.json");
        let mut registry = Self {
            spec: RwLock::new(RuntimeSpec::initial()),
            storage_path,
            native_plugins: Mutex::new(HashMap::new()),
        };
        registry.load();
        registry
    }

    /// 注册自定义执行器
    pub async fn register(&self, spec: CustomExecutorSpec) {
        let name = spec.type_name.clone();
        {
            let mut s = self.spec.write().await;
            s.register_executor(spec);
            s.meta_stats.executors_created += 1;
        }
        self.save().await;
        tracing::info!("注册自定义执行器: {}", name);
    }

    /// 变异自定义执行器
    pub async fn mutate_executor(
        &self,
        type_name: &str,
        new_code: String,
        new_description: Option<String>,
    ) -> Result<(), String> {
        let mut s = self.spec.write().await;
        let executor = s
            .custom_executors
            .iter_mut()
            .find(|e| e.type_name == type_name)
            .ok_or_else(|| format!("执行器 '{}' 不存在", type_name))?;

        executor.executor_code = new_code;
        if let Some(desc) = new_description {
            executor.description = desc;
        }
        executor.lineage.generation += 1;
        executor.lineage.origin = ExecutorOrigin::MetaMutated;
        let generation = executor.lineage.generation;

        s.meta_stats.executors_mutated += 1;
        s.meta_history.push(MetaEvolutionEvent {
            event_type: "executor_mutate".into(),
            executor_name: type_name.to_string(),
            description: format!("执行器 '{}' 变异 (代 {})", type_name, generation),
            timestamp: now_string(),
        });
        drop(s);
        self.save().await;
        Ok(())
    }

    /// 淘汰自定义执行器
    pub async fn eliminate_executor(&self, type_name: &str) -> Result<(), String> {
        let mut s = self.spec.write().await;
        let existed = s.custom_executors.iter().any(|e| e.type_name == type_name);
        if !existed {
            return Err(format!("执行器 '{}' 不存在", type_name));
        }
        s.custom_executors.retain(|e| e.type_name != type_name);
        s.meta_stats.executors_eliminated += 1;
        s.meta_history.push(MetaEvolutionEvent {
            event_type: "executor_eliminate".into(),
            executor_name: type_name.to_string(),
            description: format!("淘汰执行器 '{}'", type_name),
            timestamp: now_string(),
        });
        drop(s);
        self.save().await;
        Ok(())
    }

    /// 获取执行器规范
    pub async fn get_executor(&self, type_name: &str) -> Option<CustomExecutorSpec> {
        let s = self.spec.read().await;
        s.get_executor(type_name).cloned()
    }

    /// 获取运行时规范快照
    pub async fn spec(&self) -> RuntimeSpec {
        self.spec.read().await.clone()
    }

    /// 获取所有自定义执行器类型名
    pub async fn custom_executor_types(&self) -> Vec<String> {
        let s = self.spec.read().await;
        s.custom_executors
            .iter()
            .map(|e| e.type_name.clone())
            .collect()
    }

    /// 获取所有执行器类型（内置 + 自定义）
    pub async fn all_executor_types(&self) -> Vec<String> {
        let s = self.spec.read().await;
        s.all_executor_types()
    }

    /// 执行自定义执行器
    ///
    /// 执行器脚本接收统一输入格式，输出 JSON 到 stdout。
    /// 支持三种语言：
    /// - python: 直接执行 Python 脚本
    /// - node: 直接执行 JavaScript 脚本
    /// - rust: 编译为 WASM 后在 wasmtime 沙箱中执行（编译结果缓存）
    pub async fn execute(
        &self,
        type_name: &str,
        params: &serde_json::Value,
        input: &serde_json::Value,
        context: &ExecutorContext,
    ) -> Result<serde_json::Value, String> {
        let spec = self.spec.read().await;
        let executor = spec
            .get_executor(type_name)
            .ok_or_else(|| format!("自定义执行器 '{}' 不存在", type_name))?;

        let executor_code = executor.executor_code.clone();
        let language = executor.language.clone();
        let timeout_secs = executor.timeout_secs;
        drop(spec);

        // 构造执行器输入
        let executor_input = serde_json::json!({
            "params": params,
            "input": input,
            "context": {
                "capability": context.capability_name,
                "action": context.action_name,
            },
        });

        let input_str = serde_json::to_string(&executor_input)
            .map_err(|e| format!("执行器输入序列化失败: {}", e))?;

        match language.as_str() {
            "python" | "py" => {
                self.execute_script(
                    type_name,
                    "python3",
                    "py",
                    &executor_code,
                    &input_str,
                    timeout_secs,
                    true,
                )
                .await
            }
            "node" | "js" | "javascript" => {
                self.execute_script(
                    type_name,
                    "node",
                    "js",
                    &executor_code,
                    &input_str,
                    timeout_secs,
                    false,
                )
                .await
            }
            "rust" | "wasm" => {
                self.execute_wasm(type_name, &executor_code, &input_str, timeout_secs)
                    .await
            }
            "rust_native" | "native" => {
                self.execute_native(type_name, &executor_code, &input_str, timeout_secs)
                    .await
            }
            _ => Err(format!("不支持的执行器语言: {}", language)),
        }
    }

    /// 执行 Python/Node 脚本
    #[allow(clippy::too_many_arguments)]
    async fn execute_script(
        &self,
        type_name: &str,
        runner: &str,
        ext: &str,
        code: &str,
        input_str: &str,
        timeout_secs: u64,
        is_python: bool,
    ) -> Result<serde_json::Value, String> {
        let rendered_code = if is_python {
            render_python_code(code, input_str)
        } else {
            render_node_code(code, input_str)
        };

        let tmp = std::env::temp_dir().join(format!(
            "executor_{}_{}.{}",
            type_name,
            uuid::Uuid::new_v4(),
            ext
        ));

        tokio::fs::write(&tmp, &rendered_code)
            .await
            .map_err(|e| format!("写入执行器文件失败: {}", e))?;

        let mut cmd = tokio::process::Command::new(runner);
        cmd.arg(&tmp);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let child = cmd
            .spawn()
            .map_err(|e| format!("启动执行器 {} 失败: {}", runner, e))?;

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            child.wait_with_output(),
        )
        .await;

        let _ = tokio::fs::remove_file(&tmp).await;

        Self::process_output(output, type_name, timeout_secs)
    }

    /// 执行 Rust 代码：编译为 WASM → wasmtime 沙箱执行
    ///
    /// 编译结果通过源码哈希缓存，避免重复编译。
    /// 缓存目录: {storage_dir}/wasm_cache/{hash}.wasm
    async fn execute_wasm(
        &self,
        type_name: &str,
        code: &str,
        input_str: &str,
        timeout_secs: u64,
    ) -> Result<serde_json::Value, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // 1. 计算源码哈希，确定缓存路径
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        let hash = format!("{:016x}", hasher.finish());

        let cache_dir = self
            .storage_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("/tmp"))
            .join("wasm_cache");
        let wasm_path = cache_dir.join(format!("{}_{}.wasm", type_name, hash));

        // 2. 检查缓存，未命中则编译
        if !wasm_path.exists() {
            tokio::fs::create_dir_all(&cache_dir)
                .await
                .map_err(|e| format!("创建 WASM 缓存目录失败: {}", e))?;

            // 构造完整的 Rust 程序
            let full_code = render_rust_wasm_code(code);

            // 写入临时源文件
            let src_path = cache_dir.join(format!("{}_{}.rs", type_name, hash));
            tokio::fs::write(&src_path, &full_code)
                .await
                .map_err(|e| format!("写入 Rust 源文件失败: {}", e))?;

            // 用 rustc 编译为 WASM
            let compile_result = tokio::time::timeout(
                std::time::Duration::from_secs(120),
                tokio::process::Command::new("rustc")
                    .args([
                        "--target",
                        "wasm32-wasip1",
                        "--edition",
                        "2021",
                        "-O",
                        "-o",
                        wasm_path.to_str().unwrap(),
                        src_path.to_str().unwrap(),
                    ])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .output(),
            )
            .await;

            let _ = tokio::fs::remove_file(&src_path).await;

            match compile_result {
                Ok(Ok(out)) if out.status.success() => {
                    tracing::info!("WASM 编译成功: {} (hash: {})", type_name, &hash[..8]);
                }
                Ok(Ok(out)) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    return Err(format!(
                        "Rust 编译失败 (执行器 '{}'): {}",
                        type_name,
                        safe_truncate(&stderr, 500),
                    ));
                }
                Ok(Err(e)) => return Err(format!("rustc 启动失败: {}", e)),
                Err(_) => return Err("Rust 编译超时 (120s)".to_string()),
            }
        }

        // 3. 用 wasmtime 执行 WASM，通过 stdin 传入输入
        let mut cmd = tokio::process::Command::new("wasmtime");
        cmd.arg(&wasm_path);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("启动 wasmtime 失败: {}", e))?;

        // 写入 stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(input_str.as_bytes()).await;
            let _ = stdin.shutdown().await;
        }

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            child.wait_with_output(),
        )
        .await;

        Self::process_output(output, type_name, timeout_secs)
    }

    /// 处理子进程输出，解析为 JSON
    fn process_output(
        output: Result<Result<std::process::Output, std::io::Error>, tokio::time::error::Elapsed>,
        type_name: &str,
        timeout_secs: u64,
    ) -> Result<serde_json::Value, String> {
        match output {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let exit_code = out.status.code();
                let success = out.status.success();

                if !success {
                    return Err(format!(
                        "执行器 '{}' 执行失败 (exit {:?}): {}",
                        type_name, exit_code, stderr
                    ));
                }

                let trimmed = stdout.trim();
                // 先尝试整体解析
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if v.is_object() {
                        return Ok(v);
                    }
                }
                // 多行输出：取第一个有效 JSON 行
                for line in trimmed.lines() {
                    let line = line.trim();
                    if line.starts_with('{') {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                            if v.is_object() {
                                return Ok(v);
                            }
                        }
                    }
                }
                // 无法解析为 JSON，返回原始输出
                Ok(serde_json::json!({
                    "success": true,
                    "result": stdout,
                    "stderr": stderr,
                }))
            }
            Ok(Err(e)) => Err(format!("执行器执行失败: {}", e)),
            Err(_) => Err(format!("执行器执行超时 ({}s)", timeout_secs)),
        }
    }

    /// 执行 Rust 原生代码：编译为动态库 → 热加载执行
    ///
    /// 这是最高级别的执行方式：LLM 生成的 Rust 代码被编译为
    /// 平台原生动态库（.dylib/.so/.dll），通过 libloading 加载
    /// 并直接调用 C ABI 函数。性能与原生代码完全一致。
    ///
    /// 与 WASM 方式的区别：
    /// - WASM: 沙箱隔离，安全但受限（只能用标准库，无网络/文件系统）
    /// - Native: 无沙箱，完全能力（可以用任何 crate，访问文件系统/网络）
    ///
    /// 编译结果通过源码哈希缓存，代码不变时跳过编译。
    /// 热替换：当代码变化时，卸载旧库，加载新库。
    async fn execute_native(
        &self,
        type_name: &str,
        code: &str,
        input_str: &str,
        _timeout_secs: u64,
    ) -> Result<serde_json::Value, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // 1. 计算源码哈希
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        let hash = format!("{:016x}", hasher.finish());

        // 2. 检查是否已加载且哈希匹配
        {
            let plugins = self.native_plugins.lock().await;
            if let Some(plugin) = plugins.get(type_name) {
                if plugin.hash == hash {
                    // 已加载且代码未变，直接执行
                    return plugin.call(input_str);
                }
            }
        }

        // 3. 编译新版本
        let cache_dir = self
            .storage_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("/tmp"))
            .join("native_cache");
        tokio::fs::create_dir_all(&cache_dir)
            .await
            .map_err(|e| format!("创建原生缓存目录失败: {}", e))?;

        let ext = match std::env::consts::OS {
            "macos" => "dylib",
            "linux" => "so",
            "windows" => "dll",
            _ => "so",
        };
        let lib_path = cache_dir.join(format!("{}_{}.{}", type_name, &hash[..16], ext));

        if !lib_path.exists() {
            // 构造完整 Rust 源码（C ABI）
            let full_code = render_rust_native_code(code);
            let src_path = cache_dir.join(format!("{}_{}.rs", type_name, &hash[..16]));
            tokio::fs::write(&src_path, &full_code)
                .await
                .map_err(|e| format!("写入 Rust 源文件失败: {}", e))?;

            let compile_result = tokio::time::timeout(
                std::time::Duration::from_secs(120),
                tokio::process::Command::new("rustc")
                    .args([
                        "--edition",
                        "2021",
                        "-O",
                        "--crate-type",
                        "cdylib",
                        "-o",
                        lib_path.to_str().unwrap(),
                        src_path.to_str().unwrap(),
                    ])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .output(),
            )
            .await;

            let _ = tokio::fs::remove_file(&src_path).await;

            match compile_result {
                Ok(Ok(out)) if out.status.success() => {
                    tracing::info!("原生插件编译成功: {} (hash: {})", type_name, &hash[..8]);
                }
                Ok(Ok(out)) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    return Err(format!(
                        "Rust 原生编译失败 (执行器 '{}'): {}",
                        type_name,
                        safe_truncate(&stderr, 500),
                    ));
                }
                Ok(Err(e)) => return Err(format!("rustc 启动失败: {}", e)),
                Err(_) => return Err("Rust 原生编译超时 (120s)".to_string()),
            }
        }

        // 4. 加载动态库
        let lib =
            unsafe { Library::new(&lib_path) }.map_err(|e| format!("加载动态库失败: {}", e))?;

        let plugin = NativePlugin {
            hash: hash.clone(),
            library: lib,
            lib_path: lib_path.clone(),
        };

        // 5. 注册到缓存（热替换旧版本）
        {
            let mut plugins = self.native_plugins.lock().await;
            if let Some(old) = plugins.insert(type_name.to_string(), plugin) {
                tracing::info!(
                    "热替换原生插件: {} (旧 hash: {})",
                    type_name,
                    &old.hash[..8]
                );
            }
        }

        // 6. 执行
        let plugins = self.native_plugins.lock().await;
        if let Some(plugin) = plugins.get(type_name) {
            plugin.call(input_str)
        } else {
            Err("插件加载后未找到".to_string())
        }
    }

    /// 保存到磁盘
    async fn save(&self) {
        let s = self.spec.read().await;
        let json = s.to_json();
        let tmp = self.storage_path.with_extension("tmp");
        if std::fs::write(&tmp, &json).is_ok() {
            let _ = std::fs::rename(&tmp, &self.storage_path);
        }
    }

    /// 从磁盘加载
    fn load(&mut self) {
        if let Ok(content) = std::fs::read_to_string(&self.storage_path) {
            if let Ok(spec) = RuntimeSpec::from_json(&content) {
                tracing::info!(
                    "从磁盘加载运行时规范: {} 个自定义执行器, {} 个元进化事件",
                    spec.custom_executors.len(),
                    spec.meta_history.len()
                );
                // 直接用 blocking_write，避免在 tokio runtime 内嵌套 block_on
                let mut s = self.spec.blocking_write();
                *s = spec;
            }
        }
    }
}

/// 执行器上下文 — 传递给执行器的运行时信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorContext {
    pub capability_name: String,
    pub action_name: String,
}

/// 渲染 Python 执行器代码：将输入 JSON 注入到脚本中
///
/// 执行器脚本通过 `__EXECUTOR_INPUT__` 全局变量获取输入。
pub fn render_python_code(code: &str, input_json: &str) -> String {
    let escaped = input_json.replace('\\', "\\\\").replace('\'', "\\'");

    if code.contains("__EXECUTOR_INPUT__") {
        format!(
            "import json, sys\n__EXECUTOR_INPUT__ = json.loads('''{}''')\n{}",
            escaped, code
        )
    } else {
        format!(
            "import json, sys\n__EXECUTOR_INPUT__ = json.loads(sys.stdin.read())\n{}",
            code
        )
    }
}

/// 渲染 Node.js 执行器代码
pub fn render_node_code(code: &str, input_json: &str) -> String {
    let escaped = input_json
        .replace('\\', "\\\\")
        .replace('`', "\\`")
        .replace('$', "\\$");

    if code.contains("__EXECUTOR_INPUT__") {
        format!(
            "const __EXECUTOR_INPUT__ = JSON.parse(`{}`);\n{}",
            escaped, code
        )
    } else {
        format!(
            "const __EXECUTOR_INPUT__ = JSON.parse(require('fs').readFileSync('/dev/stdin', 'utf8'));\n{}",
            code
        )
    }
}

/// 渲染 Rust WASM 执行器代码
///
/// 将 LLM 生成的 Rust 代码包装为完整的 WASM 程序。
/// 通过 stdin 读取 JSON 输入，通过 stdout 输出 JSON。
///
/// 注意：wasm32-wasip1 目标下不使用外部 crate（serde_json 等），
/// 而是用纯标准库实现简易 JSON 解析。LLM 生成的代码通过
/// `__input` 变量（&str 类型）获取输入 JSON 字符串。
pub fn render_rust_wasm_code(user_code: &str) -> String {
    let template = "use std::io::{self, Read, Write};

fn main() {
    // 读取 stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap_or_else(|e| {
        eprintln!(\"读取 stdin 失败: {}\", e);
        std::process::exit(1);
    });

    // 用户代码在此执行
    // __input 变量包含完整的输入 JSON 字符串
    let __input: &str = &input;
    let mut __output_done: bool = false;

    // 用户代码中应通过 println! 输出 JSON 到 stdout
{user_code}

    // 如果用户代码没有输出，输出默认成功响应
    if !__output_done {
        println!(\"{{\\\"success\\\": true}}\");
    }
}
";
    template.replace("{user_code}", user_code)
}

/// 已加载的原生插件
///
/// 封装了通过 libloading 加载的动态库，
/// 提供 C ABI 调用接口。
pub struct NativePlugin {
    /// 源码哈希（用于判断是否需要重新编译）
    hash: String,
    /// 已加载的动态库
    library: Library,
    /// 动态库文件路径
    #[allow(dead_code)]
    lib_path: std::path::PathBuf,
}

impl NativePlugin {
    /// 调用插件的 execute 函数
    ///
    /// 插件必须导出 C ABI 函数：
    /// ```c
    /// char* execute(const char* input);
    /// void free_string(char* ptr);
    /// ```
    fn call(&self, input: &str) -> Result<serde_json::Value, String> {
        use std::ffi::CString;
        use std::os::raw::c_char;

        // 获取函数符号
        let execute_fn: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> *mut c_char> =
            unsafe { self.library.get(b"execute\0") }
                .map_err(|e| format!("获取 execute 符号失败: {}", e))?;

        let free_fn: libloading::Symbol<unsafe extern "C" fn(*mut c_char)> =
            unsafe { self.library.get(b"free_string\0") }
                .map_err(|e| format!("获取 free_string 符号失败: {}", e))?;

        // 构造 C 字符串输入
        let input_cstr = CString::new(input).map_err(|e| format!("输入包含 null 字节: {}", e))?;

        // 调用插件
        let result_ptr = unsafe { execute_fn(input_cstr.as_ptr()) };

        if result_ptr.is_null() {
            return Err("插件返回 null 指针".to_string());
        }

        // 转换结果为 Rust String
        let result = unsafe {
            let cstr = std::ffi::CStr::from_ptr(result_ptr);
            let s = cstr.to_string_lossy().to_string();
            free_fn(result_ptr);
            s
        };

        // 解析 JSON
        let trimmed = result.trim();
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(v) if v.is_object() => Ok(v),
            _ => Ok(serde_json::json!({
                "success": true,
                "result": result,
            })),
        }
    }
}

/// 渲染 Rust 原生插件代码
///
/// 将 LLM 生成的 Rust 代码包装为完整的 cdylib 程序，
/// 导出 C ABI 函数供 libloading 调用。
///
/// 用户代码通过 `__input` 变量（&str）获取输入 JSON，
/// 通过 `__output` 变量返回输出 JSON 字符串。
pub fn render_rust_native_code(user_code: &str) -> String {
    let template = "use std::os::raw::c_char;
use std::ffi::{CStr, CString};

#[no_mangle]
pub extern \"C\" fn execute(input: *const c_char) -> *mut c_char {
    let input_str = if input.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(input) }.to_string_lossy().to_string()
    };

    let __input: &str = &input_str;

    // 用户代码在此执行
    // __input 包含完整的输入 JSON 字符串
    // 将输出 JSON 字符串赋值给 __output
    let mut __output: String = String::from(\"{\\\"success\\\": true}\");

    // ─── 用户代码开始 ───
{user_code}
    // ─── 用户代码结束 ───

    // 如果用户代码没有设置 __output，使用默认值
    let result = CString::new(__output).unwrap_or_else(|_| CString::new(\"{\\\"error\\\": \\\"output contains null byte\\\"}\").unwrap());
    result.into_raw()
}

#[no_mangle]
pub extern \"C\" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { CString::from_raw(ptr) };
    }
}
";
    template.replace("{user_code}", user_code)
}

/// 元进化器（MetaEvolver）— 进化运行时本身
///
/// 与 AutoEvolver（进化能力）不同，MetaEvolver 进化的是"进化的规则"：
/// - 分析能力库的执行模式瓶颈
/// - 创造新的执行器类型
/// - 变异现有执行器
/// - 淘汰低效执行器
///
/// 生物学类比：
/// AutoEvolver = 自然选择（变异生物个体）
/// MetaEvolver = 进化进化本身（变异遗传密码）
pub struct MetaEvolver {
    /// LLM 执行器
    llm: Arc<LlmExecutor>,
    /// 消息总线
    #[allow(dead_code)]
    bus: Arc<MessageBus>,
    /// 平台信息
    platform: Platform,
    /// 执行器注册表
    registry: Arc<ExecutorRegistry>,
}

impl MetaEvolver {
    pub fn new(
        llm: Arc<LlmExecutor>,
        bus: Arc<MessageBus>,
        platform: Platform,
        registry: Arc<ExecutorRegistry>,
    ) -> Self {
        Self {
            llm,
            bus,
            platform,
            registry,
        }
    }

    /// 运行一轮元进化
    ///
    /// 1. 元自省：分析能力库的执行模式瓶颈
    /// 2. 元创造：如果发现瓶颈，用 LLM 生成新执行器
    /// 3. 元变异：变异现有自定义执行器
    /// 4. 元淘汰：淘汰低效执行器
    pub async fn meta_evolve_once(
        &self,
        evolution: &mut EvolutionEngine,
    ) -> Result<Vec<String>, String> {
        let mut actions = Vec::new();

        // 1. 元自省
        let report = self.meta_introspect(evolution).await;
        {
            let mut spec = self.registry.spec.write().await;
            spec.meta_stats.meta_introspections += 1;
        }

        if report.bottleneck_description.is_empty() {
            println!("  🔬 元自省: 未发现执行模式瓶颈");
            self.registry.save().await;
            return Ok(actions);
        }

        println!("  🔬 元自省: 发现瓶颈 — {}", report.bottleneck_description);

        // 2. 元创造：如果有瓶颈且有提案，创造新执行器
        if let Some(proposal) = &report.executor_proposal {
            println!(
                "  🧬 元创造: 提议新执行器 '{}' — {}",
                proposal.type_name, proposal.description
            );

            let spec = CustomExecutorSpec {
                type_name: proposal.type_name.clone(),
                description: proposal.description.clone(),
                params_schema: proposal.params_schema.clone(),
                executor_code: proposal.executor_code.clone(),
                language: proposal.language.clone(),
                timeout_secs: proposal.timeout_secs.unwrap_or(60),
                created_at: now_string(),
                lineage: ExecutorLineage {
                    origin: ExecutorOrigin::MetaGenerated,
                    parent: None,
                    generation: 1,
                },
            };

            // 测试执行器
            let test_result = self.test_executor(&spec).await;
            if test_result {
                self.registry.register(spec).await;
                actions.push(format!(
                    "元创造: 执行器 '{}' (测试通过)",
                    proposal.type_name
                ));
                println!("  ✅ 元创造成功: 执行器 '{}' 已注册", proposal.type_name);

                // 统计使用自定义执行器的能力数
                let count = self.count_custom_executor_usage(evolution).await;
                {
                    let mut s = self.registry.spec.write().await;
                    s.meta_stats.capabilities_using_custom = count;
                }
            } else {
                println!(
                    "  ❌ 元创造失败: 执行器 '{}' 测试未通过",
                    proposal.type_name
                );
                actions.push(format!(
                    "元创造: 执行器 '{}' (测试失败)",
                    proposal.type_name
                ));
            }
        }

        // 3. 元变异：检查自定义执行器是否需要优化
        let custom_types = self.registry.custom_executor_types().await;
        for type_name in &custom_types {
            // 简单策略：每 5 轮检查一次
            // 实际实现中可以用更智能的策略
            if self.should_mutate_executor(type_name, evolution).await {
                if let Some(new_code) = self.generate_executor_mutation(type_name).await {
                    match self
                        .registry
                        .mutate_executor(type_name, new_code.0, Some(new_code.1))
                        .await
                    {
                        Ok(_) => {
                            actions.push(format!("元变异: 执行器 '{}'", type_name));
                            println!("  🧬 元变异: 执行器 '{}' 已优化", type_name);
                        }
                        Err(e) => {
                            println!("  ❌ 元变异失败: {}", e);
                        }
                    }
                }
            }
        }

        // 4. 元淘汰：淘汰从未被使用的自定义执行器
        let unused = self.find_unused_executors(evolution).await;
        for name in &unused {
            match self.registry.eliminate_executor(name).await {
                Ok(_) => {
                    actions.push(format!("元淘汰: 执行器 '{}'", name));
                    println!("  🗑️  元淘汰: 执行器 '{}' (从未被使用)", name);
                }
                Err(e) => println!("  ❌ 元淘汰失败: {}", e),
            }
        }

        self.registry.save().await;
        Ok(actions)
    }

    /// 元自省：分析能力库的执行模式瓶颈
    ///
    /// 让 LLM 分析当前所有能力的实现类型分布，
    /// 发现是否有任务无法用现有 5 种 ActionImpl 有效表达。
    async fn meta_introspect(&self, evolution: &EvolutionEngine) -> MetaIntrospectionReport {
        let genomes: Vec<_> = evolution.genomes().values().cloned().collect();
        let spec = self.registry.spec().await;

        // 统计执行类型分布
        let mut type_counts: HashMap<String, u32> = HashMap::new();
        for g in &genomes {
            for action in &g.actions {
                let type_name = match &action.implementation {
                    ActionImpl::Llm { .. } => "Llm",
                    ActionImpl::Rule { .. } => "Rule",
                    ActionImpl::Composite { .. } => "Composite",
                    ActionImpl::Native { .. } => "Native",
                    ActionImpl::Script { .. } => "Script",
                    ActionImpl::Custom { executor_type, .. } => {
                        type_counts
                            .entry(format!("Custom:{}", executor_type))
                            .or_insert(0);
                        type_counts
                            .entry(format!("Custom:{}", executor_type))
                            .and_modify(|c| *c += 1);
                        continue;
                    }
                };
                *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
            }
        }

        let type_dist: Vec<String> = type_counts
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect();

        let prompt = format!(
            r#"你是一个运行时元进化分析器。你的任务是分析当前能力库的执行模式瓶颈，
并判断是否需要创造新的执行器类型（超越现有的 Llm/Rule/Composite/Native/Script 五种）。

当前执行类型分布:
{type_dist}

已有自定义执行器: {custom_executors}
内置执行器: {builtin}

能力库摘要:
{caps_summary}

平台: {os} ({arch})
可用工具: {tools}

分析维度：
1. 是否有大量能力使用 Script 但实际做的是简单映射？（应该用 Rule 但可能表达力不够）
2. 是否有 Composite 能力的步骤过于复杂，实际上需要条件分支/循环？（需要控制流执行器）
3. 是否有 Script 能力反复启动 Python 进程，性能瓶颈明显？（需要缓存执行器）
4. 是否有能力的逻辑无法用现有 5 种类型优雅表达？

如果发现瓶颈，给出一个新执行器提案。执行器支持三种语言：

- language: "python" — Python 脚本，接收 __EXECUTOR_INPUT__ 变量（含 params/input/context），输出 JSON 到 stdout
  适合快速原型、数据处理、需要丰富库（numpy/requests 等）

- language: "rust" — Rust 代码，通过 __input 变量（&str）获取输入 JSON，用 println! 输出 JSON 到 stdout
  编译为 WASM 在沙箱中执行（编译结果缓存），安全隔离，只能使用标准库
  适合性能敏感的纯计算场景

- language: "rust_native" — Rust 代码，通过 __input 变量（&str）获取输入 JSON，将输出 JSON 字符串赋值给 __output 变量
  编译为原生动态库（.dylib/.so）通过 libloading 热加载执行，性能与原生代码完全一致
  可以使用外部 crate（通过 cargo 编译），可以访问文件系统、网络等全部系统能力
  适合需要完整系统能力、最高性能、或需要调用其他 Rust 库的场景
  这是最高级别的进化：系统可以生成真正的原生代码并热加载

返回严格 JSON:
{{
  "bottleneck_description": "瓶颈描述，如果无瓶颈则为空字符串",
  "bottleneck_severity": "low/medium/high",
  "executor_proposal": {{
    "type_name": "执行器类型名（如 cached_script, pipeline, conditional）",
    "description": "执行器描述",
    "params_schema": {{"properties": {{}}}},
    "language": "python, rust, 或 rust_native",
    "executor_code": "执行器代码",
    "timeout_secs": 60
  }}
}}

如果无瓶颈，bottleneck_description 设为空字符串，executor_proposal 设为 null。"#,
            type_dist = type_dist.join(", "),
            custom_executors = if spec.custom_executors.is_empty() {
                "（无）".to_string()
            } else {
                spec.custom_executors
                    .iter()
                    .map(|e| format!("{}: {} [{}]", e.type_name, e.description, e.language))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
            builtin = spec.builtin_executors.join(", "),
            caps_summary = if genomes.is_empty() {
                "（空）".to_string()
            } else {
                genomes
                    .iter()
                    .take(20)
                    .map(|g| {
                        let impl_types: Vec<String> = g
                            .actions
                            .iter()
                            .map(|a| impl_type_name(&a.implementation))
                            .collect();
                        format!("{}: {} [{}]", g.name, g.description, impl_types.join(","))
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            },
            os = self.platform.os,
            arch = self.platform.arch,
            tools = {
                let mut t: Vec<String> = self
                    .platform
                    .env
                    .iter()
                    .filter(|(k, v)| {
                        (k.starts_with("has_") || k.starts_with("has_py_")) && v.as_str() == "true"
                    })
                    .map(|(k, _)| {
                        k.strip_prefix("has_")
                            .or_else(|| k.strip_prefix("has_py_"))
                            .unwrap_or(k)
                            .to_string()
                    })
                    .collect();
                t.sort();
                t.join(", ")
            },
        );

        let result = match self.llm.execute(&prompt, "smart:meta", None).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("元自省 LLM 调用失败: {}", e);
                return MetaIntrospectionReport::default();
            }
        };

        let json_str = extract_json(&result);
        let v: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "元自省 JSON 解析失败: {} | 原始前200字符: {}",
                    e,
                    safe_truncate(&result, 200)
                );
                return MetaIntrospectionReport::default();
            }
        };

        let bottleneck = v
            .get("bottleneck_description")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();

        let proposal = v
            .get("executor_proposal")
            .filter(|p| !p.is_null())
            .and_then(|p| serde_json::from_value::<ExecutorProposal>(p.clone()).ok());

        MetaIntrospectionReport {
            bottleneck_description: bottleneck,
            executor_proposal: proposal,
        }
    }

    /// 测试执行器是否正常工作
    async fn test_executor(&self, spec: &CustomExecutorSpec) -> bool {
        let context = ExecutorContext {
            capability_name: "meta_evolver_test".into(),
            action_name: "test".into(),
        };

        // 构造测试输入
        let test_input = serde_json::json!({"test": "hello"});
        let test_params = serde_json::json!({});

        match self
            .registry
            .execute(&spec.type_name, &test_params, &test_input, &context)
            .await
        {
            Ok(result) => result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            Err(e) => {
                tracing::warn!("执行器 '{}' 测试失败: {}", spec.type_name, e);
                false
            }
        }
    }

    /// 检查执行器是否需要变异
    async fn should_mutate_executor(&self, type_name: &str, evolution: &EvolutionEngine) -> bool {
        // 检查使用此执行器的能力是否有低成功率的
        let weak: Vec<_> = evolution
            .genomes()
            .values()
            .filter(|g| {
                g.actions.iter().any(|a| {
                    if let ActionImpl::Custom { executor_type, .. } = &a.implementation {
                        executor_type == type_name
                            && g.fitness.success_rate < 0.5
                            && g.fitness.call_count > 2
                    } else {
                        false
                    }
                })
            })
            .collect();
        !weak.is_empty()
    }

    /// 生成执行器变异方案
    async fn generate_executor_mutation(&self, type_name: &str) -> Option<(String, String)> {
        let spec = self.registry.get_executor(type_name).await?;

        let prompt = format!(
            r#"你是一个执行器变异器。以下执行器可能需要优化。

执行器名: {name}
描述: {desc}
当前代码:
```python
{code}
```

请分析代码并给出优化版本。返回严格 JSON:
{{
  "new_code": "优化后的完整 Python 代码",
  "new_description": "更新后的描述"
}}"#,
            name = spec.type_name,
            desc = spec.description,
            code = spec.executor_code,
        );

        let result = self
            .llm
            .execute(&prompt, "coder:optimize", None)
            .await
            .ok()?;
        let json_str = extract_json(&result);
        let v: serde_json::Value = serde_json::from_str(json_str).ok()?;

        let new_code = v.get("new_code").and_then(|s| s.as_str())?.to_string();
        let new_desc = v
            .get("new_description")
            .and_then(|s| s.as_str())?
            .to_string();
        Some((new_code, new_desc))
    }

    /// 找出从未被使用的自定义执行器
    async fn find_unused_executors(&self, evolution: &EvolutionEngine) -> Vec<String> {
        let custom_types = self.registry.custom_executor_types().await;
        let used_types: std::collections::HashSet<String> = evolution
            .genomes()
            .values()
            .flat_map(|g| g.actions.iter())
            .filter_map(|a| {
                if let ActionImpl::Custom { executor_type, .. } = &a.implementation {
                    Some(executor_type.clone())
                } else {
                    None
                }
            })
            .collect();

        custom_types
            .into_iter()
            .filter(|t| !used_types.contains(t))
            .collect()
    }

    /// 统计使用自定义执行器的能力数
    async fn count_custom_executor_usage(&self, evolution: &EvolutionEngine) -> u32 {
        evolution
            .genomes()
            .values()
            .filter(|g| {
                g.actions
                    .iter()
                    .any(|a| matches!(a.implementation, ActionImpl::Custom { .. }))
            })
            .count() as u32
    }

    /// 生成元进化报告
    pub async fn report(&self) -> String {
        let spec = self.registry.spec().await;
        format!(
            "═══ 元进化报告 ═══\n\
             元自省次数: {}\n\
             执行器创造: {}\n\
             执行器变异: {}\n\
             执行器淘汰: {}\n\
             使用自定义执行器的能力: {}\n\
             当前自定义执行器: {}\n\
             元进化事件: {}\n",
            spec.meta_stats.meta_introspections,
            spec.meta_stats.executors_created,
            spec.meta_stats.executors_mutated,
            spec.meta_stats.executors_eliminated,
            spec.meta_stats.capabilities_using_custom,
            if spec.custom_executors.is_empty() {
                "（无）".to_string()
            } else {
                spec.custom_executors
                    .iter()
                    .map(|e| format!("{} (代{})", e.type_name, e.lineage.generation))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
            spec.meta_history.len(),
        )
    }
}

/// 元自省报告
#[derive(Debug, Clone, Default)]
struct MetaIntrospectionReport {
    bottleneck_description: String,
    executor_proposal: Option<ExecutorProposal>,
}

/// 执行器提案（LLM 生成）
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutorProposal {
    type_name: String,
    description: String,
    params_schema: serde_json::Value,
    language: String,
    executor_code: String,
    timeout_secs: Option<u64>,
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_default()
}

/// 获取 ActionImpl 的类型名（用于摘要显示）
fn impl_type_name(impl_: &crate::genome::ActionImpl) -> String {
    match impl_ {
        crate::genome::ActionImpl::Llm { .. } => "Llm".into(),
        crate::genome::ActionImpl::Rule { .. } => "Rule".into(),
        crate::genome::ActionImpl::Composite { .. } => "Composite".into(),
        crate::genome::ActionImpl::Native { .. } => "Native".into(),
        crate::genome::ActionImpl::Script { .. } => "Script".into(),
        crate::genome::ActionImpl::Custom { executor_type, .. } => {
            format!("Custom:{}", executor_type)
        }
    }
}

fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_spec_initial() {
        let spec = RuntimeSpec::initial();
        assert_eq!(spec.builtin_executors.len(), 5);
        assert!(spec.custom_executors.is_empty());
    }

    #[test]
    fn test_runtime_spec_register_executor() {
        let mut spec = RuntimeSpec::initial();
        let custom = CustomExecutorSpec {
            type_name: "cached_script".into(),
            description: "带缓存的脚本执行器".into(),
            params_schema: serde_json::json!({}),
            executor_code: "print(json.dumps({'success': True}))".into(),
            language: "python".into(),
            timeout_secs: 30,
            created_at: "123".into(),
            lineage: ExecutorLineage::default(),
        };
        spec.register_executor(custom);
        assert_eq!(spec.custom_executors.len(), 1);
        assert!(spec.get_executor("cached_script").is_some());
        assert_eq!(spec.meta_history.len(), 1);
    }

    #[test]
    fn test_runtime_spec_all_types() {
        let mut spec = RuntimeSpec::initial();
        spec.register_executor(CustomExecutorSpec {
            type_name: "pipeline".into(),
            description: "管道执行器".into(),
            params_schema: serde_json::json!({}),
            executor_code: "".into(),
            language: "python".into(),
            timeout_secs: 30,
            created_at: "123".into(),
            lineage: ExecutorLineage::default(),
        });
        let all = spec.all_executor_types();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&"pipeline".to_string()));
    }

    #[test]
    fn test_runtime_spec_serialization() {
        let mut spec = RuntimeSpec::initial();
        spec.register_executor(CustomExecutorSpec {
            type_name: "test_exec".into(),
            description: "测试".into(),
            params_schema: serde_json::json!({"properties": {"code": {"type": "string"}}}),
            executor_code: "print('hello')".into(),
            language: "python".into(),
            timeout_secs: 10,
            created_at: "123".into(),
            lineage: ExecutorLineage::default(),
        });
        let json = spec.to_json();
        let restored = RuntimeSpec::from_json(&json).unwrap();
        assert_eq!(restored.custom_executors.len(), 1);
        assert_eq!(restored.custom_executors[0].type_name, "test_exec");
    }

    #[test]
    fn test_render_executor_code_python() {
        let code = "print(json.dumps(__EXECUTOR_INPUT__))";
        let rendered = render_python_code(code, r#"{"test": true}"#);
        assert!(rendered.contains("__EXECUTOR_INPUT__"));
        assert!(rendered.contains("json.loads"));
        assert!(rendered.contains("test"));
    }

    #[test]
    fn test_render_executor_code_stdin() {
        let code = "import json; print(json.dumps({'success': True}))";
        let rendered = render_python_code(code, r#"{"test": true}"#);
        assert!(rendered.contains("sys.stdin.read()"));
    }

    #[test]
    fn test_render_rust_wasm_code() {
        let user_code = r##"println!(r#"{"success": true}"#);"##;
        let rendered = render_rust_wasm_code(user_code);
        assert!(rendered.contains("fn main()"));
        assert!(rendered.contains("read_to_string"));
        assert!(rendered.contains("__input"));
    }

    #[test]
    fn test_render_rust_native_code() {
        let user_code = r#"__output = format!("{{\"result\": \"{}\"}}", __input);"#;
        let rendered = render_rust_native_code(user_code);
        assert!(rendered.contains("#[no_mangle]"));
        assert!(rendered.contains("pub extern \"C\" fn execute"));
        assert!(rendered.contains("free_string"));
        assert!(rendered.contains("__input"));
        assert!(rendered.contains("__output"));
        assert!(rendered.contains(user_code));
    }
}
