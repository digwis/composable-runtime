use crate::auto_evolve::AutoEvolver;
use crate::autonomous::AutonomousRuntime;
use crate::evolution::EvolutionEngine;
use crate::failure_driver::FailureDriver;
use crate::genome::{LlmExecutor, ScriptedCapability};
use crate::message_bus::MessageBus;
use crate::meta_evolve::{ExecutorRegistry, MetaEvolver};
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

/// Daemon 配置
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub bin_dir: PathBuf,
    pub storage_dir: PathBuf,
    pub evolution_interval_secs: u64,
    pub max_rounds: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        Self {
            socket_path: PathBuf::from(format!("{}/.orch/socket", home)),
            bin_dir: PathBuf::from(format!("{}/.orch/bin", home)),
            storage_dir: PathBuf::from(format!("{}/.orch", home)),
            evolution_interval_secs: 300,
            max_rounds: 100,
        }
    }
}

/// Daemon 状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub pid: u32,
    pub capabilities_count: usize,
    pub total_calls: u64,
    pub total_evolutions: u64,
    pub uptime_secs: u64,
    pub socket_path: String,
    pub bin_dir: String,
}

/// 共享进化状态 — 在 socket handler 和进化循环之间共享
struct SharedState {
    evolution: EvolutionEngine,
    failure_driver: Option<FailureDriver>,
    total_evolutions: u64,
}

/// 进化运行时 Daemon — 系统级常驻服务
pub struct Daemon {
    config: DaemonConfig,
    bus: Arc<MessageBus>,
    llm: Option<Arc<LlmExecutor>>,
    platform: Platform,
    start_time: std::time::Instant,
    total_calls: u64,
    shared: Arc<Mutex<SharedState>>,
}

impl Daemon {
    pub fn new(
        config: DaemonConfig,
        bus: Arc<MessageBus>,
        evolution: EvolutionEngine,
        llm: Option<Arc<LlmExecutor>>,
        platform: Platform,
    ) -> Self {
        let failure_driver = llm.as_ref().map(|l| FailureDriver::new(l.clone()));
        let shared = Arc::new(Mutex::new(SharedState {
            evolution,
            failure_driver,
            total_evolutions: 0,
        }));
        Self {
            config,
            bus,
            llm,
            platform,
            start_time: std::time::Instant::now(),
            total_calls: 0,
            shared,
        }
    }

    /// 启动 daemon
    pub async fn run(&mut self) -> Result<(), String> {
        std::fs::create_dir_all(&self.config.storage_dir)
            .map_err(|e| format!("创建存储目录失败: {}", e))?;
        std::fs::create_dir_all(&self.config.bin_dir)
            .map_err(|e| format!("创建 bin 目录失败: {}", e))?;

        if self.config.socket_path.exists() {
            std::fs::remove_file(&self.config.socket_path).ok();
        }

        self.register_all_capabilities().await;
        self.inject_to_path().await;

        let listener = UnixListener::bind(&self.config.socket_path)
            .map_err(|e| format!("绑定 socket 失败: {}: {}", self.config.socket_path.display(), e))?;

        let cap_count = {
            let state = self.shared.lock().await;
            state.evolution.genomes().len()
        };

        tracing::info!(
            "Daemon 启动 — socket: {}, bin: {}, 能力: {}",
            self.config.socket_path.display(),
            self.config.bin_dir.display(),
            cap_count
        );

        let evolution_handle = self.spawn_evolution_loop();

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let bus = self.bus.clone();
                    let shared = self.shared.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, bus, shared).await {
                            tracing::warn!("连接处理错误: {}", e);
                        }
                    });
                    self.total_calls += 1;
                }
                Err(e) => {
                    tracing::error!("接受连接失败: {}", e);
                    break;
                }
            }
        }

        evolution_handle.abort();
        Ok(())
    }

    /// 注册所有能力到消息总线
    async fn register_all_capabilities(&self) {
        let genomes: Vec<_> = {
            let state = self.shared.lock().await;
            state.evolution.genomes().values().cloned().collect()
        };

        for genome in &genomes {
            if genome.actions.is_empty() {
                continue;
            }
            if !self.platform.is_compatible(genome) {
                continue;
            }
            let mut cap = ScriptedCapability::from_genome(genome.clone());
            if let Some(llm) = &self.llm {
                cap = cap.with_llm(llm.clone()).with_bus(self.bus.clone());
            }
            self.bus.register(Arc::new(cap)).await;
        }
    }

    /// PATH 注入 — 为每个能力创建可执行文件
    async fn inject_to_path(&self) {
        let genomes: Vec<_> = {
            let state = self.shared.lock().await;
            state.evolution.genomes().values().cloned().collect()
        };

        let mut count = 0;
        for genome in &genomes {
            if genome.actions.is_empty() {
                continue;
            }
            for action in &genome.actions {
                let filename = format!("{}.{}", genome.name, action.name);
                let filepath = self.config.bin_dir.join(&filename);
                let script = format!(
                    r#"#!/bin/sh
# Auto-generated by orch daemon
# Capability: {} / Action: {}
exec orch exec "{}" "{}" "$@"
"#,
                    genome.name, action.name, genome.name, action.name
                );
                std::fs::write(&filepath, script).ok();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&filepath, std::fs::Permissions::from_mode(0o755)).ok();
                }
                count += 1;
            }
        }
        tracing::info!("PATH 注入: {} 个可执行文件", count);
    }

    /// 后台进化循环 — 失败驱动 + 自省变异 + 淘汰 + 交叉重组
    fn spawn_evolution_loop(&self) -> tokio::task::JoinHandle<()> {
        let interval = self.config.evolution_interval_secs;
        let max_rounds = self.config.max_rounds;
        let shared = self.shared.clone();
        let bus = self.bus.clone();
        let llm = self.llm.clone();
        let platform = self.platform.clone();
        let bin_dir = self.config.bin_dir.clone();

        tokio::spawn(async move {
            let mut round = 0u32;
            loop {
                if round >= max_rounds {
                    tracing::info!("进化循环: 达到最大轮次 {}, 停止", max_rounds);
                    break;
                }
                round += 1;

                tokio::time::sleep(std::time::Duration::from_secs(interval)).await;

                tracing::info!("进化循环: 第 {} 轮", round);

                // 1. 失败驱动进化
                {
                    let mut state = shared.lock().await;
                    if let Some(fd) = &mut state.failure_driver {
                        let outcomes = match tokio::time::timeout(
                            std::time::Duration::from_secs(300),
                            fd.evolve_from_failures(),
                        ).await {
                            Ok(o) => o,
                            Err(_) => {
                                tracing::warn!("进化循环: 失败驱动超时 (300s)，跳过");
                                vec![]
                            }
                        };
                        let passed = outcomes.iter().filter(|o| o.is_passing()).count();
                        let total = outcomes.len();
                        if total > 0 {
                            tracing::info!(
                                "进化循环: 失败驱动产生 {} 个能力, {} 个通过验证",
                                total, passed
                            );
                            for outcome in &outcomes {
                                if outcome.is_passing() {
                                    state.evolution.register_genome(outcome.genome.clone());
                                    state.total_evolutions += 1;
                                }
                            }
                        } else {
                            tracing::info!("进化循环: 无失败事件或未产生新能力");
                        }
                    }
                }

                // 2. 自省 + 变异（AutoEvolver）
                if let Some(llm) = &llm {
                    let mut auto = AutoEvolver::new(
                        llm.clone(),
                        bus.clone(),
                        platform.clone(),
                    );

                    let mut state = shared.lock().await;
                    tracing::info!("进化循环: 执行自省+变异...");
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(600),
                        auto.evolve_once(&mut state.evolution),
                    ).await {
                        Ok(Ok(actions)) => {
                            if actions.is_empty() {
                                tracing::info!("进化循环: 无需进化动作");
                            } else {
                                tracing::info!(
                                    "进化循环: 进化动作: {}",
                                    actions.join(", ")
                                );
                                state.total_evolutions += actions.len() as u64;
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("进化循环: 进化失败: {}", e);
                        }
                        Err(_) => {
                            tracing::warn!("进化循环: 自省+变异超时 (600s)，跳过");
                        }
                    }
                }

                // 2.5 自主循环 — 感知环境 + 生成目标 + 主动执行
                if let Some(llm) = &llm {
                    let mut auto_runtime = AutonomousRuntime::new(
                        llm.clone(),
                        bus.clone(),
                        platform.clone(),
                    );

                    let mut state = shared.lock().await;
                    tracing::info!("自主循环: 启动...");
                    let auto_results = match tokio::time::timeout(
                        std::time::Duration::from_secs(300),
                        auto_runtime.autonomous_cycle(&mut state.evolution),
                    ).await {
                        Ok(results) => results,
                        Err(_) => {
                            tracing::warn!("自主循环: 超时 (300s)，跳过");
                            vec![]
                        }
                    };
                    let (successes, failures) = auto_runtime.stats();
                    tracing::info!(
                        "自主循环: 完成 — {} 个目标, {} 成功, {} 失败",
                        auto_results.len(), successes, failures
                    );
                }

                // 2.6 元进化 — 每 5 轮执行一次，进化执行器本身
                if round % 5 == 0 {
                    if let Some(llm) = &llm {
                        let storage_dir = std::path::PathBuf::from(
                            std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())
                        ).join(".orch");
                        let registry = Arc::new(ExecutorRegistry::new(storage_dir.clone()));
                        let meta = MetaEvolver::new(
                            llm.clone(),
                            bus.clone(),
                            platform.clone(),
                            registry.clone(),
                        );

                        let mut state = shared.lock().await;
                        tracing::info!("元进化: 启动...");
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(300),
                            meta.meta_evolve_once(&mut state.evolution),
                        ).await {
                            Ok(Ok(actions)) => {
                                if actions.is_empty() {
                                    tracing::info!("元进化: 无需进化动作");
                                } else {
                                    tracing::info!("元进化: 动作: {}", actions.join(", "));
                                }
                            }
                            Ok(Err(e)) => {
                                tracing::warn!("元进化: 失败: {}", e);
                            }
                            Err(_) => {
                                tracing::warn!("元进化: 超时 (300s)，跳过");
                            }
                        }
                    }
                }

                // 3. 保存 + 重新注入 PATH + 注册新能力
                {
                    let state = shared.lock().await;
                    state.evolution.save();
                    let genomes: Vec<_> = state.evolution.genomes().values().cloned().collect();
                    drop(state);

                    // 重新生成 PATH 可执行文件
                    for genome in &genomes {
                        if genome.actions.is_empty() {
                            continue;
                        }
                        for action in &genome.actions {
                            let filename = format!("{}.{}", genome.name, action.name);
                            let filepath = bin_dir.join(&filename);
                            let script = format!(
                                r#"#!/bin/sh
# Auto-generated by orch daemon
# Capability: {} / Action: {}
exec orch exec "{}" "{}" "$@"
"#,
                                genome.name, action.name, genome.name, action.name
                            );
                            std::fs::write(&filepath, script).ok();
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::PermissionsExt;
                                std::fs::set_permissions(&filepath, std::fs::Permissions::from_mode(0o755)).ok();
                            }
                        }
                    }

                    // 注册新能力到总线
                    let registered = bus.list_capabilities().await;
                    for genome in &genomes {
                        if genome.actions.is_empty() || !platform.is_compatible(genome) {
                            continue;
                        }
                        if registered.iter().any(|c| *c == genome.name) {
                            continue;
                        }
                        let mut cap = ScriptedCapability::from_genome(genome.clone());
                        if let Some(llm) = &llm {
                            cap = cap.with_llm(llm.clone()).with_bus(bus.clone());
                        }
                        bus.register(Arc::new(cap)).await;
                        tracing::info!("进化循环: 注册新能力: {}", genome.name);
                    }
                }

                tracing::info!("进化循环: 第 {} 轮完成", round);
            }
        })
    }

    /// 获取 daemon 状态
    pub async fn status(&self) -> DaemonStatus {
        let state = self.shared.lock().await;
        DaemonStatus {
            running: true,
            pid: std::process::id(),
            capabilities_count: state.evolution.genomes().len(),
            total_calls: self.total_calls,
            total_evolutions: state.total_evolutions,
            uptime_secs: self.start_time.elapsed().as_secs(),
            socket_path: self.config.socket_path.display().to_string(),
            bin_dir: self.config.bin_dir.display().to_string(),
        }
    }
}

/// 处理 Unix socket 连接
async fn handle_connection(
    stream: UnixStream,
    bus: Arc<MessageBus>,
    shared: Arc<Mutex<SharedState>>,
) -> Result<(), String> {
    let (reader, writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {
                let request: serde_json::Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(_) => {
                        let resp = serde_json::json!({"success": false, "error": "Invalid JSON"});
                        writer.write_all(format!("{}\n", resp).as_bytes()).await.ok();
                        writer.flush().await.ok();
                        continue;
                    }
                };

                let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");

                let response = match method {
                    "exec" => {
                        let cap = request.get("capability").and_then(|c| c.as_str()).unwrap_or("");
                        let action = request.get("action").and_then(|a| a.as_str()).unwrap_or("");
                        let input = request.get("input").cloned().unwrap_or(serde_json::json!({}));
                        let input_for_failure = input.clone();

                        let msg = crate::message::Message::builder()
                            .from("daemon")
                            .to(cap)
                            .action(action)
                            .payload(input)
                            .build();

                        match bus.send(msg).await {
                            Ok(resp) => {
                                let is_failure = resp.payload.get("success")
                                    .and_then(|s| s.as_bool())
                                    .map(|s| !s)
                                    .unwrap_or(false);

                                if is_failure {
                                    let error = resp.payload.get("error")
                                        .and_then(|e| e.as_str())
                                        .unwrap_or("unknown error");
                                    let mut state = shared.lock().await;
                                    if let Some(fd) = &mut state.failure_driver {
                                        fd.record_failure(crate::failure_driver::FailureEvent {
                                            task: "socket_exec".into(),
                                            capability: cap.into(),
                                            action: action.into(),
                                            input: input_for_failure.clone(),
                                            error: error.into(),
                                            timestamp: format!("{}", std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map(|d| d.as_secs())
                                                .unwrap_or(0)),
                                        });
                                    }
                                }

                                serde_json::json!({"success": true, "payload": resp.payload})
                            }
                            Err(e) => {
                                let mut state = shared.lock().await;
                                if let Some(fd) = &mut state.failure_driver {
                                    fd.record_failure(crate::failure_driver::FailureEvent {
                                        task: "socket_exec".into(),
                                        capability: cap.into(),
                                        action: action.into(),
                                        input: input_for_failure.clone(),
                                        error: e.to_string(),
                                        timestamp: format!("{}", std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map(|d| d.as_secs())
                                            .unwrap_or(0)),
                                    });
                                }
                                serde_json::json!({"success": false, "error": e.to_string()})
                            }
                        }
                    }
                    "list" => {
                        let caps = bus.list_capabilities().await;
                        serde_json::json!({"success": true, "capabilities": caps})
                    }
                    "introspect" => {
                        let caps = bus.introspect().await;
                        serde_json::json!({"success": true, "capabilities": caps})
                    }
                    "status" => {
                        let state = shared.lock().await;
                        serde_json::json!({
                            "success": true,
                            "status": {
                                "pid": std::process::id(),
                                "capabilities": state.evolution.genomes().len(),
                                "total_evolutions": state.total_evolutions,
                            }
                        })
                    }
                    _ => {
                        serde_json::json!({"success": false, "error": format!("Unknown method: {}", method)})
                    }
                };

                writer.write_all(format!("{}\n", response).as_bytes()).await.ok();
                writer.flush().await.ok();
            }
            Err(e) => {
                tracing::warn!("读取连接失败: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// LLM 后端自动发现
pub fn discover_llm_backends() -> Vec<DiscoveredBackend> {
    let mut backends = vec![];

    for (cmd, backend) in &[
        ("claude", "claude"),
        ("devin", "devin"),
    ] {
        if which(cmd).is_some() {
            backends.push(DiscoveredBackend {
                name: backend.to_string(),
                backend_type: BackendType::Cli,
                command: cmd.to_string(),
            });
        }
    }

    if std::env::var("ANTHROPIC_API_KEY").is_ok() || std::env::var("CLAUDE_API_KEY").is_ok() {
        backends.push(DiscoveredBackend {
            name: "anthropic-api".into(),
            backend_type: BackendType::Http,
            command: String::new(),
        });
    }

    backends
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredBackend {
    pub name: String,
    pub backend_type: BackendType,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackendType {
    Cli,
    Http,
}

fn which(cmd: &str) -> Option<PathBuf> {
    let path = std::env::var("PATH").ok()?;
    for dir in path.split(':') {
        let candidate = Path::new(dir).join(cmd);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
