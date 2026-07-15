use crate::auto_evolve::AutoEvolver;
use crate::autonomous::AutonomousRuntime;
use crate::driver::EvolutionDriver;
use crate::evolution::EvolutionEngine;
use crate::failure_driver::FailureDriver;
use crate::genome::ScriptedCapability;
use crate::message_bus::MessageBus;
use crate::meta_evolve::{ExecutorRegistry, MetaEvolver};
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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
    /// HTTP API 端口（0 = 不启 HTTP server）
    pub http_port: u16,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        Self {
            socket_path: PathBuf::from(format!("{}/.orch/socket", home)),
            bin_dir: PathBuf::from(format!("{}/.orch/bin", home)),
            storage_dir: PathBuf::from(format!("{}/.orch", home)),
            evolution_interval_secs: 300,
            max_rounds: 0, // 0 = 无限运行（完全自主）
            http_port: 7331,
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
pub struct SharedState {
    pub evolution: EvolutionEngine,
    pub failure_driver: Option<FailureDriver>,
    pub total_evolutions: u64,
}

/// 进化运行时 Daemon — 系统级常驻服务
pub struct Daemon {
    config: DaemonConfig,
    bus: Arc<MessageBus>,
    llm: Option<Arc<dyn EvolutionDriver>>,
    /// LlmExecutor 的热切换句柄（仅当 llm 是 LlmExecutor 时有值）— 供 HTTP /api/config
    llm_override: Option<Arc<std::sync::RwLock<Option<crate::genome::LlmConfig>>>>,
    platform: Platform,
    start_time: std::time::Instant,
    total_calls: u64,
    shared: Arc<Mutex<SharedState>>,
    /// daemon 生命周期内共享的执行器注册表；元进化产物与所有能力使用同一实例。
    executor_registry: Arc<ExecutorRegistry>,
}

struct DaemonInstanceLock {
    path: PathBuf,
}

impl Drop for DaemonInstanceLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn acquire_instance_lock(storage_dir: &std::path::Path) -> Result<DaemonInstanceLock, String> {
    let path = storage_dir.join("daemon.lock");
    let pid = std::process::id().to_string();
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(pid.as_bytes())
                .map_err(|e| format!("写入 daemon 锁失败: {}", e))?;
            Ok(DaemonInstanceLock { path })
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing_pid = std::fs::read_to_string(&path).unwrap_or_default();
            let running = existing_pid
                .trim()
                .parse::<u32>()
                .ok()
                .map(|pid| {
                    std::process::Command::new("kill")
                        .args(["-0", &pid.to_string()])
                        .status()
                        .map(|status| status.success())
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            if running {
                return Err(format!("daemon 已在运行 (PID {})", existing_pid.trim()));
            }
            std::fs::remove_file(&path).map_err(|e| format!("清理陈旧 daemon 锁失败: {}", e))?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| format!("重新获取 daemon 锁失败: {}", e))?;
            use std::io::Write;
            file.write_all(pid.as_bytes())
                .map_err(|e| format!("写入 daemon 锁失败: {}", e))?;
            Ok(DaemonInstanceLock { path })
        }
        Err(error) => Err(format!("创建 daemon 锁失败: {}", error)),
    }
}

impl Daemon {
    pub fn new(
        config: DaemonConfig,
        bus: Arc<MessageBus>,
        evolution: EvolutionEngine,
        llm: Option<Arc<dyn EvolutionDriver>>,
        platform: Platform,
    ) -> Self {
        let executor_registry = Arc::new(ExecutorRegistry::new(config.storage_dir.clone()));
        let evolution = evolution.with_executor_registry(executor_registry.clone());
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
            llm_override: None,
            platform,
            start_time: std::time::Instant::now(),
            total_calls: 0,
            shared,
            executor_registry,
        }
    }

    /// 注入 LlmExecutor 的热切换句柄（orchestrator 构造 LlmExecutor 后调用）
    pub fn with_llm_override(
        mut self,
        handle: Arc<std::sync::RwLock<Option<crate::genome::LlmConfig>>>,
    ) -> Self {
        self.llm_override = Some(handle);
        self
    }

    /// 构造 HTTP API 用的共享句柄
    pub fn http_handle(&self) -> Arc<crate::http_api::DaemonHandle> {
        // 若有 LLM driver，构造任务编排器
        let task_orchestrator = self.llm.as_ref().map(|llm| {
            Arc::new(crate::task_orchestrator::TaskOrchestrator::new(
                llm.clone(),
                self.bus.clone(),
                self.shared.clone(),
                self.config.storage_dir.to_string_lossy().to_string(),
            ))
        });
        Arc::new(crate::http_api::DaemonHandle {
            shared: self.shared.clone(),
            bus: self.bus.clone(),
            config: self.config.clone(),
            start_time: self.start_time,
            llm_override: self.llm_override.clone(),
            llm: self.llm.clone(),
            task_orchestrator,
            breaker: self.llm.as_ref().and_then(|l| l.breaker()),
        })
    }

    /// 启动 daemon
    pub async fn run(&mut self) -> Result<(), String> {
        std::fs::create_dir_all(&self.config.storage_dir)
            .map_err(|e| format!("创建存储目录失败: {}", e))?;
        std::fs::create_dir_all(&self.config.bin_dir)
            .map_err(|e| format!("创建 bin 目录失败: {}", e))?;

        let _instance_lock = acquire_instance_lock(&self.config.storage_dir)?;

        if self.config.socket_path.exists() {
            std::fs::remove_file(&self.config.socket_path).ok();
        }

        self.register_all_capabilities().await;
        self.inject_to_path().await;

        match crate::durable_run::recover_project_runs(
            self.config.storage_dir.clone(),
            self.shared.clone(),
            self.bus.clone(),
        ) {
            Ok(count) if count > 0 => {
                tracing::info!(count, "已恢复中断的持久项目任务");
            }
            Ok(_) => {}
            Err(error) => tracing::warn!("持久项目任务恢复失败: {}", error),
        }

        // Connected cloud services are bootstrapped idempotently. Only page
        // and repository identifiers are persisted; credentials remain in
        // each CLI's keychain.
        let integration_storage = self.config.storage_dir.clone();
        tokio::spawn(async move {
            let status = crate::integrations::detect_integrations().await;
            let path = integration_storage.join("integrations_status.json");
            let temporary = path.with_extension("json.tmp");
            if let Ok(bytes) = serde_json::to_vec_pretty(&status) {
                if std::fs::write(&temporary, bytes).is_ok() {
                    let _ = std::fs::rename(temporary, path);
                }
            }
            if let Err(error) = crate::cloud_sync::sync_personal_cloud(&integration_storage).await {
                tracing::warn!("个人云端初始化与同步失败: {}", error);
            }
        });

        // Sync personal capabilities and distilled knowledge independently
        // from the evolution lock. The cloud sync has its own process lock.
        let cloud_storage = self.config.storage_dir.clone();
        let cloud_interval = self.config.evolution_interval_secs.max(300);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(cloud_interval)).await;
            loop {
                if let Err(error) = crate::cloud_sync::sync_personal_cloud(&cloud_storage).await {
                    tracing::warn!("个人云端周期同步失败: {}", error);
                }
                tokio::time::sleep(std::time::Duration::from_secs(cloud_interval)).await;
            }
        });

        let listener = UnixListener::bind(&self.config.socket_path).map_err(|e| {
            format!(
                "绑定 socket 失败: {}: {}",
                self.config.socket_path.display(),
                e
            )
        })?;

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
        let observer_roots = crate::project_worker::configured_project_roots();
        let observer_storage = self.config.storage_dir.clone();
        let observer_interval = self.config.evolution_interval_secs.max(60);
        tokio::spawn(async move {
            loop {
                let _ = crate::workspace::observe(&observer_roots, &observer_storage).await;
                tokio::time::sleep(std::time::Duration::from_secs(observer_interval)).await;
            }
        });

        // 主动控制回路独立于能力进化循环：它只读取工作区快照，按 Initiative
        // 策略创建提示/隔离实验，并把每次决策写入 autonomy/state.json。
        let autonomy = crate::autonomy_controller::AutonomyController::new(
            self.config.storage_dir.clone(),
            crate::project_worker::configured_project_roots(),
            self.llm.clone(),
            self.shared.clone(),
            self.bus.clone(),
            self.config.evolution_interval_secs.max(60),
        );
        tokio::spawn(async move { autonomy.run().await });

        // HTTP API server（与 Unix socket 并存）— 供 Dashboard 查状态/注入反馈/实测能力
        if self.config.http_port > 0 {
            let http_handle = self.http_handle();
            let port = self.config.http_port;
            tokio::spawn(async move {
                if let Err(e) = crate::http_api::start_http_server(http_handle, port).await {
                    tracing::warn!("HTTP API 退出: {}", e);
                }
            });
        }

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
                cap = cap.with_llm(llm.clone());
            }
            cap = cap
                .with_bus(self.bus.clone())
                .with_executor_registry(self.executor_registry.clone());
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
                    std::fs::set_permissions(&filepath, std::fs::Permissions::from_mode(0o755))
                        .ok();
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
        let breaker = self.llm.as_ref().and_then(|l| l.breaker());
        let llm_override = self.llm_override.clone();
        let platform = self.platform.clone();
        let bin_dir = self.config.bin_dir.clone();
        let executor_registry = self.executor_registry.clone();

        tokio::spawn(async move {
            // AutoEvolver 持有跨轮进化状态（已尝试缺口、连续变异失败、统计与轮次）。
            // 若在 loop 内每轮重建，这些选择压力都会被清空，系统无法从上一轮学习。
            let mut auto_evolver = llm.as_ref().map(|llm| {
                AutoEvolver::new(llm.clone(), bus.clone(), platform.clone())
                    .with_executor_registry(executor_registry.clone())
            });
            let mut round = 0u32;
            loop {
                // A4: max_rounds = 0 表示无限运行（完全自主），>0 时按指定轮次停止
                if max_rounds > 0 && round >= max_rounds {
                    tracing::info!("进化循环: 达到最大轮次 {}, 停止", max_rounds);
                    break;
                }
                round += 1;

                // 首轮立即执行，后续轮次等待 interval 秒
                if round > 1 {
                    tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
                }

                tracing::info!("进化循环: 第 {} 轮", round);

                // === 可用性门:时段 + 熔断 ===
                let in_active_hours = match &llm_override {
                    Some(h) => {
                        let cfg = match h.read() {
                            Ok(g) => g
                                .clone()
                                .unwrap_or_else(|| crate::http_api::fallback_config_from_env()),
                            Err(_) => return, // 锁中毒,退出 task
                        };
                        cfg.active_hours
                            .as_ref()
                            .map(|tw| tw.contains_now())
                            .unwrap_or(true)
                    }
                    None => true, // 无 override 句柄(CLI driver)→ 不限时段
                };
                if !in_active_hours {
                    tracing::info!("进化循环: 第 {} 轮: API 非开放时段, 暂停", round);
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    continue;
                }
                if let Some(b) = &breaker {
                    if b.should_probe() {
                        // Open 满 60s 或 HalfOpen:转 HalfOpen,让本轮 execute 真实探测
                        b.probe_succeeded();
                        tracing::info!("进化循环: API 转入探测态(HalfOpen), 本轮将真实调用验证");
                    } else if b.is_open() {
                        tracing::info!("进化循环: 第 {} 轮: API 熔断中, 暂停", round);
                        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                        continue;
                    }
                }
                // === 正常进化流程 ===

                // 1. 失败驱动进化
                // FailureDriver owns LLM/sandbox work, so temporarily move it
                // out of SharedState. HTTP/API readers remain responsive while
                // failure analysis and synthesis run outside the global lock.
                let failure_driver = {
                    let mut state = shared.lock().await;
                    state.failure_driver.take()
                };
                if let Some(mut fd) = failure_driver {
                    let outcomes = match tokio::time::timeout(
                        std::time::Duration::from_secs(300),
                        fd.evolve_from_failures(),
                    )
                    .await
                    {
                        Ok(o) => o,
                        Err(_) => {
                            tracing::warn!("进化循环: 失败驱动超时 (300s)，跳过");
                            vec![]
                        }
                    };
                    let passed = outcomes.iter().filter(|o| o.is_passing()).count();
                    let total = outcomes.len();
                    let mut state = shared.lock().await;
                    if total > 0 {
                        tracing::info!(
                            "进化循环: 失败驱动产生 {} 个能力, {} 个通过验证",
                            total,
                            passed
                        );
                        for outcome in &outcomes {
                            if outcome.is_passing() {
                                if let Err(e) =
                                    state.evolution.register_genome(outcome.genome.clone())
                                {
                                    tracing::warn!("register_genome 保存失败: {}", e);
                                }
                                state.total_evolutions += 1;
                            }
                        }
                    } else {
                        tracing::info!("进化循环: 无失败事件或未产生新能力");
                    }
                    state.failure_driver = Some(fd);
                }

                // 2. 自省 + 变异（AutoEvolver）— 归因无锁化编排
                // 锁内：sync_fitness + introspect + 快照 → 释放锁
                // 锁外：归因（错峰 90s×i 全程不占锁，消除 HTTP 503 窗口）
                // 锁内：evolve_once_with_attribution 写回变异/测试/选择
                if let Some(auto) = auto_evolver.as_mut() {
                    tracing::info!("进化循环: 执行自省+变异...");

                    // 2.1 锁内：sync + introspect + prepare 快照（毫秒级）
                    let precomputed = {
                        let mut state = shared.lock().await;
                        if let Err(e) = auto.sync_fitness(&mut state.evolution).await {
                            tracing::warn!(
                                "进化循环: sync_fitness 失败，本轮停止选择以避免使用部分状态: {}",
                                e
                            );
                            continue;
                        }
                        let report = auto.introspect(&state.evolution);
                        // 能力库非空才归因
                        if report.total_capabilities == 0 {
                            None
                        } else {
                            let (weak_list, snapshot) =
                                auto.prepare_attribution(&state.evolution, &report);
                            if weak_list.is_empty() {
                                None
                            } else {
                                Some((weak_list, snapshot))
                            }
                        }
                    }; // 锁释放

                    // 2.2 锁外：无锁归因（错峰全程不占锁）
                    let attr_result: Option<(
                        Vec<crate::auto_evolve::WeakCapability>,
                        Vec<Option<crate::auto_evolve::AttributionResult>>,
                        crate::auto_evolve::AttributionSnapshot,
                    )> = if let Some((weak_list, snapshot)) = precomputed {
                        tracing::info!("进化循环: 无锁归因 {} 个弱能力...", weak_list.len());
                        let attrs = match tokio::time::timeout(
                            std::time::Duration::from_secs(600),
                            auto.attribute_weak_caps_snapshot(&snapshot, &weak_list),
                        )
                        .await
                        {
                            Ok(a) => a,
                            Err(_) => {
                                tracing::warn!("进化循环: 无锁归因超时 (600s)");
                                vec![None; weak_list.len()]
                            }
                        };
                        Some((weak_list, attrs, snapshot))
                    } else {
                        None
                    };

                    // 2.3 四阶段编排：锁内准备 → 锁外变异测试 → 锁内写回+收集快照 → 锁外自测试+真实验证 → 锁内最终写回
                    //
                    // 把所有 LLM 调用和能力执行移到锁外，消除长时间持锁导致的 HTTP 503。

                    // 2.3a 锁内：自省 + 变异应用 + 取快照
                    let phase1 = {
                        let mut state = shared.lock().await;
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(120),
                            auto.prepare_phase(&mut state.evolution, attr_result),
                        )
                        .await
                        {
                            Ok(Ok(p)) => p,
                            Ok(Err(e)) => {
                                tracing::warn!("进化循环: prepare_phase 失败: {}", e);
                                continue;
                            }
                            Err(_) => {
                                tracing::warn!("进化循环: prepare_phase 超时 (120s)，跳过");
                                continue;
                            }
                        }
                    }; // 锁释放

                    // 2.3b 锁外：变异测试 + 回归 + AB
                    let test_outcomes = if phase1.test_targets.is_empty() {
                        Vec::new()
                    } else {
                        tracing::info!(
                            "进化循环: 锁外变异测试 {} 个...",
                            phase1.test_targets.len()
                        );
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(600),
                            auto.test_and_select_unlocked(&phase1.test_targets),
                        )
                        .await
                        {
                            Ok(outcomes) => outcomes,
                            Err(_) => {
                                tracing::warn!("进化循环: 锁外变异测试超时 (600s)");
                                phase1
                                    .test_targets
                                    .iter()
                                    .map(|t| crate::auto_evolve::TestOutcome::TestFailed {
                                        parent_name: t.parent_name.clone(),
                                        child_name: t.child_name.clone(),
                                        test_input: None,
                                    })
                                    .collect()
                            }
                        }
                    };

                    // 2.3c 锁内：写回变异结果 + 收集自测试/真实验证快照
                    let intermediate = {
                        let mut state = shared.lock().await;
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(30),
                            auto.commit_test_results(&mut state.evolution, phase1, test_outcomes),
                        )
                        .await
                        {
                            Ok(Ok(i)) => i,
                            Ok(Err(e)) => {
                                tracing::warn!("进化循环: commit_test_results 失败: {}", e);
                                continue;
                            }
                            Err(_) => {
                                tracing::warn!("进化循环: commit_test_results 超时 (30s)，跳过");
                                continue;
                            }
                        }
                    }; // 锁释放

                    // 2.3d 锁外：自测试 + 真实验证（LLM 调用 + 执行，不持锁）
                    let (self_test_results, validation_results) = {
                        let mut st = Vec::new();
                        let mut vr = Vec::new();
                        if !intermediate.self_test_targets.is_empty() {
                            tracing::info!(
                                "进化循环: 锁外自测试 {} 个...",
                                intermediate.self_test_targets.len()
                            );
                            st = match tokio::time::timeout(
                                std::time::Duration::from_secs(300),
                                auto.run_self_tests_unlocked(&intermediate.self_test_targets),
                            )
                            .await
                            {
                                Ok(r) => r,
                                Err(_) => {
                                    tracing::warn!("进化循环: 锁外自测试超时 (300s)");
                                    Vec::new()
                                }
                            };
                        }
                        if !intermediate.validation_targets.is_empty() {
                            tracing::info!(
                                "进化循环: 锁外真实验证 {} 个...",
                                intermediate.validation_targets.len()
                            );
                            vr = match tokio::time::timeout(
                                std::time::Duration::from_secs(300),
                                auto.run_validations_unlocked(&intermediate.validation_targets),
                            )
                            .await
                            {
                                Ok(r) => r,
                                Err(_) => {
                                    tracing::warn!("进化循环: 锁外真实验证超时 (300s)");
                                    Vec::new()
                                }
                            };
                        }
                        (st, vr)
                    };

                    // 2.3e 锁内：最终写回 + 结晶化 + 淘汰 + 缺口 + 探索 + 持久化
                    let mut state = shared.lock().await;
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(120),
                        auto.commit_final(
                            &mut state.evolution,
                            intermediate,
                            self_test_results,
                            validation_results,
                        ),
                    )
                    .await
                    {
                        Ok(Ok(actions)) => {
                            if actions.is_empty() {
                                tracing::info!("进化循环: 无需进化动作");
                            } else {
                                tracing::info!("进化循环: 进化动作: {}", actions.join(", "));
                                state.total_evolutions += actions.len() as u64;
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("进化循环: commit_final 失败: {}", e);
                        }
                        Err(_) => {
                            tracing::warn!("进化循环: commit_final 超时 (120s)，跳过");
                        }
                    }
                }

                // 2.5 自主循环 — 感知环境 + 生成目标 + 主动执行
                if let Some(llm) = &llm {
                    let mut auto_runtime =
                        AutonomousRuntime::new(llm.clone(), bus.clone(), platform.clone());
                    // AutonomousRuntime performs goal generation and capability
                    // calls, both of which may await an LLM or external command.
                    // Run it against a point-in-time engine snapshot instead of
                    // holding SharedState across those awaits.
                    let mut autonomous_evolution = {
                        let state = shared.lock().await;
                        state.evolution.clone()
                    };
                    tracing::info!("自主循环: 启动...");
                    let auto_results = match tokio::time::timeout(
                        std::time::Duration::from_secs(300),
                        auto_runtime.autonomous_cycle(&mut autonomous_evolution),
                    )
                    .await
                    {
                        Ok(results) => results,
                        Err(_) => {
                            tracing::warn!("自主循环: 超时 (300s)，跳过");
                            vec![]
                        }
                    };
                    let (successes, failures) = auto_runtime.stats();
                    tracing::info!(
                        "自主循环: 完成 — {} 个目标, {} 成功, {} 失败",
                        auto_results.len(),
                        successes,
                        failures
                    );

                    // Merge only observable autonomous outcomes back into the
                    // live engine. This avoids overwriting concurrent HTTP or
                    // project-task updates made while the snapshot was running.
                    let mut state = shared.lock().await;
                    for result in &auto_results {
                        if let Some(capability) = result.goal.suggested_capabilities.first() {
                            if let Some(genome) = state.evolution.genomes_mut().get_mut(capability)
                            {
                                // The snapshot already executed the call, but
                                // live state must receive exactly one matching
                                // fitness event after the lock-free phase.
                                genome
                                    .fitness
                                    .record_real_call(result.success, result.elapsed_ms as f64);
                                if let Some(useful) = result.value_feedback {
                                    genome.fitness.record_human_signal(useful);
                                }
                            }
                        }
                        let entry = crate::evolution::AutonomousHistoryEntry {
                            goal: result.goal.description.clone(),
                            success: result.success,
                            capabilities_used: result.goal.suggested_capabilities.clone(),
                            elapsed_ms: result.elapsed_ms,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0),
                        };
                        state.evolution.record_autonomous_history(entry);
                    }
                }

                // 2.6 元进化 — 每 5 轮执行一次，进化执行器本身
                //
                // 安全开关：默认关闭。自动元进化会在宿主机上用 rustc 编译
                // LLM 生成的 Rust 源码到 WASM/native，存在编译期信息泄露和
                // 宿主权限风险。必须显式设置环境变量启用实验功能。
                //
                //   COMPOSABLE_AUTO_META_EVOLVE=1
                //
                // 长期方案：把 rustc 放入真正隔离的编译沙箱（如 Cloudflare Sandbox SDK
                // 或 Docker 隔离容器），并对 wasmtime 施加输出/内存/fuel 限制。
                let meta_evolve_enabled = std::env::var("COMPOSABLE_AUTO_META_EVOLVE")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if meta_evolve_enabled && round % 5 == 0 {
                    if let Some(llm) = &llm {
                        let meta = MetaEvolver::new(
                            llm.clone(),
                            bus.clone(),
                            platform.clone(),
                            executor_registry.clone(),
                        );

                        let mut state = shared.lock().await;
                        tracing::info!("元进化: 启动...");
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(300),
                            meta.meta_evolve_once(&mut state.evolution),
                        )
                        .await
                        {
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
                    if let Err(e) = state.evolution.save() {
                        tracing::warn!("evolution save 失败: {}", e);
                    }
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
                                std::fs::set_permissions(
                                    &filepath,
                                    std::fs::Permissions::from_mode(0o755),
                                )
                                .ok();
                            }
                        }
                    }

                    // 注册新能力到总线
                    let registered = bus.list_capabilities().await;
                    for genome in &genomes {
                        if genome.actions.is_empty() || !platform.is_compatible(genome) {
                            continue;
                        }
                        if registered.contains(&genome.name) {
                            continue;
                        }
                        let mut cap = ScriptedCapability::from_genome(genome.clone());
                        if let Some(llm) = &llm {
                            cap = cap.with_llm(llm.clone());
                        }
                        cap = cap
                            .with_bus(bus.clone())
                            .with_executor_registry(executor_registry.clone());
                        bus.register(Arc::new(cap)).await;
                        tracing::info!("进化循环: 注册新能力: {}", genome.name);
                    }
                }

                // 4. 保存跨代记忆
                {
                    let mut state = shared.lock().await;
                    state.evolution.memory_mut().global_stats.total_rounds += 1;
                    state.evolution.memory_mut().last_evolution_ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    if state.evolution.memory().global_stats.first_boot_ts == 0 {
                        state.evolution.memory_mut().global_stats.first_boot_ts =
                            state.evolution.memory().last_evolution_ts;
                    }
                    if let Err(e) = state.evolution.save_memory() {
                        tracing::error!("进化循环: 跨代记忆保存失败: {}", e);
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
                        writer
                            .write_all(format!("{}\n", resp).as_bytes())
                            .await
                            .ok();
                        writer.flush().await.ok();
                        continue;
                    }
                };

                let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");

                let response = match method {
                    "exec" => {
                        let cap = request
                            .get("capability")
                            .and_then(|c| c.as_str())
                            .unwrap_or("");
                        let action = request.get("action").and_then(|a| a.as_str()).unwrap_or("");
                        let input = request
                            .get("input")
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        let input_for_failure = input.clone();

                        let msg = crate::message::Message::builder()
                            .from("daemon")
                            .to(cap)
                            .action(action)
                            .payload(input)
                            .build();

                        match bus.send(msg).await {
                            Ok(resp) => {
                                let is_failure = resp
                                    .payload
                                    .get("success")
                                    .and_then(|s| s.as_bool())
                                    .map(|s| !s)
                                    .unwrap_or(false);

                                if is_failure {
                                    let error = resp
                                        .payload
                                        .get("error")
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
                                            timestamp: format!(
                                                "{}",
                                                std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .map(|d| d.as_secs())
                                                    .unwrap_or(0)
                                            ),
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
                                        timestamp: format!(
                                            "{}",
                                            std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map(|d| d.as_secs())
                                                .unwrap_or(0)
                                        ),
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
                    // 人类价值反馈 — 注入"有用/无用"信号到能力 fitness。
                    // 这是进化价值标准的唯一人类入口:自动 fitness 只测"能跑通",
                    // 人类实测判 useful/useless 直接主导 score(权重 0.95)。
                    // 请求: {"method":"feedback","capability":"py_sklearn_ops","useful":true,"note":"..."}
                    "feedback" => {
                        let cap = request
                            .get("capability")
                            .and_then(|c| c.as_str())
                            .unwrap_or("");
                        let useful = request
                            .get("useful")
                            .and_then(|u| u.as_bool())
                            .unwrap_or(false);
                        let note = request.get("note").and_then(|n| n.as_str()).unwrap_or("");
                        if cap.is_empty() {
                            serde_json::json!({"success": false, "error": "缺少 capability 字段"})
                        } else {
                            let mut state = shared.lock().await;
                            let previous_fitness = state
                                .evolution
                                .genomes()
                                .get(cap)
                                .map(|g| g.fitness.clone());
                            let previous_memory = state.evolution.memory().clone();
                            let mut result = match state.evolution.genomes_mut().get_mut(cap) {
                                Some(g) => {
                                    let prev_score = g.fitness.score;
                                    let prev_signals = g.fitness.human_signals_count;
                                    g.fitness.record_human_signal(useful);
                                    // 先取出结果,再释放对 g 的可变借用,避免 save_fitness 二次借用
                                    serde_json::json!({
                                        "ok": true,
                                        "capability": cap,
                                        "useful": useful,
                                        "prev_score": prev_score,
                                        "new_score": g.fitness.score,
                                        "human_signals_count": g.fitness.human_signals_count,
                                        "human_score": g.fitness.human_score,
                                        "prev_signals": prev_signals,
                                        "note": note,
                                    })
                                }
                                None => serde_json::json!({
                                    "ok": false,
                                    "error": format!("未找到能力: {}", cap)
                                }),
                            };
                            if result.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                                if !note.trim().is_empty() {
                                    state.evolution.record_lesson(
                                        crate::evolution::EvolutionLesson {
                                            lesson: format!(
                                                "人类反馈「{}」：{}（{}）",
                                                cap,
                                                note.trim(),
                                                if useful { "有用" } else { "无用" }
                                            ),
                                            capability: cap.to_string(),
                                            failure_type: if useful {
                                                "human_useful".into()
                                            } else {
                                                "human_useless".into()
                                            },
                                            learned_at: std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map(|d| d.as_secs().to_string())
                                                .unwrap_or_default(),
                                            referenced_count: 0,
                                        },
                                    );
                                }
                                let persist_result = state
                                    .evolution
                                    .save_fitness()
                                    .and_then(|_| state.evolution.save_memory());
                                if let Err(e) = persist_result {
                                    if let (Some(previous), Some(g)) = (
                                        previous_fitness,
                                        state.evolution.genomes_mut().get_mut(cap),
                                    ) {
                                        g.fitness = previous;
                                    }
                                    *state.evolution.memory_mut() = previous_memory;
                                    let _ = state.evolution.save_fitness();
                                    let _ = state.evolution.save_memory();
                                    result = serde_json::json!({
                                        "ok": false,
                                        "error": format!("反馈持久化失败，已回滚: {}", e)
                                    });
                                }
                            }
                            let mut resp = serde_json::json!({
                                "success": result
                                    .get("ok")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false)
                            });
                            resp["result"] = result;
                            resp
                        }
                    }
                    _ => {
                        serde_json::json!({"success": false, "error": format!("Unknown method: {}", method)})
                    }
                };

                writer
                    .write_all(format!("{}\n", response).as_bytes())
                    .await
                    .ok();
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

    if std::env::var("ORCH_API_KEY").is_ok()
        || std::env::var("ANTHROPIC_API_KEY").is_ok()
        || std::env::var("CLAUDE_API_KEY").is_ok()
    {
        backends.push(DiscoveredBackend {
            name: "http-api".into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_config_default() {
        let config = DaemonConfig::default();
        assert!(config
            .socket_path
            .to_string_lossy()
            .contains(".orch/socket"));
        assert!(config.bin_dir.to_string_lossy().contains(".orch/bin"));
        assert_eq!(config.evolution_interval_secs, 300);
        assert_eq!(config.max_rounds, 0); // 0 = 无限运行
    }

    #[test]
    fn test_daemon_config_custom() {
        let config = DaemonConfig {
            socket_path: PathBuf::from("/tmp/test.sock"),
            bin_dir: PathBuf::from("/tmp/bin"),
            storage_dir: PathBuf::from("/tmp/storage"),
            evolution_interval_secs: 60,
            max_rounds: 10,
            http_port: 0,
        };
        assert_eq!(config.socket_path, PathBuf::from("/tmp/test.sock"));
        assert_eq!(config.evolution_interval_secs, 60);
        assert_eq!(config.max_rounds, 10);
    }

    #[test]
    fn test_daemon_status_serialization() {
        let status = DaemonStatus {
            running: true,
            pid: 12345,
            capabilities_count: 5,
            total_calls: 100,
            total_evolutions: 3,
            uptime_secs: 7200,
            socket_path: "/tmp/sock".into(),
            bin_dir: "/tmp/bin".into(),
        };
        let json = serde_json::to_string(&status).unwrap();
        let decoded: DaemonStatus = serde_json::from_str(&json).unwrap();
        assert!(decoded.running);
        assert_eq!(decoded.pid, 12345);
        assert_eq!(decoded.capabilities_count, 5);
    }

    #[test]
    fn test_daemon_status_default_fields() {
        let status = DaemonStatus {
            running: false,
            pid: 0,
            capabilities_count: 0,
            total_calls: 0,
            total_evolutions: 0,
            uptime_secs: 0,
            socket_path: "".into(),
            bin_dir: "".into(),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"running\":false"));
    }

    #[test]
    fn test_backend_type_variants() {
        let cli = BackendType::Cli;
        let http = BackendType::Http;
        assert!(matches!(cli, BackendType::Cli));
        assert!(matches!(http, BackendType::Http));
    }

    #[tokio::test]
    async fn test_daemon_new() {
        let config = DaemonConfig::default();
        let bus = Arc::new(MessageBus::new());
        let evolution = EvolutionEngine::new("/tmp/test_evo");
        let platform = Platform::detect();
        let daemon = Daemon::new(config, bus, evolution, None, platform);
        assert_eq!(daemon.total_calls, 0);
    }
}
