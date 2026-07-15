//! HTTP API — 进化运行时的人机交互层
//!
//! 在 daemon 的 Unix socket 之外，另起一个 localhost HTTP server，暴露：
//! - 进化状态/能力列表查询（供 Dashboard 展示）
//! - 人类价值反馈注入（POST /api/feedback → FitnessGene.record_human_signal）
//! - 能力实测（POST /api/exec → 复用 message bus）
//! - API 配置查询（GET /api/config）
//!
//! 锁策略：进化循环在归因段长持 shared 锁（间歇，数分钟）。本模块的只读端点
//! 用 try_lock 配合短超时——撞上归因窗口时返回 503"进化忙"让前端优雅降级，
//! 而非挂死请求。feedback 注入是毫秒级写，通常能挤进窗口间隙。
//!
//! 端点形状（行分隔 JSON 已废弃，改 REST）：
//!   GET  /api/status        → {pid, capabilities, total_evolutions, uptime_secs}
//!   GET  /api/capabilities  → [{name, score, success_rate, call_count, human_score, ...}]
//!   POST /api/feedback      → {capability, useful, note?} → 注入人类信号
//!   POST /api/exec          → {capability, action, input?} → 实测能力
//!   GET  /api/config        → {model, base_url, has_key}（key 不回传）

use crate::daemon::DaemonConfig;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

/// HTTP server 共享状态 — 与 daemon 同源的 shared 锁 + bus
#[derive(Clone)]
pub struct HttpState {
    pub daemon: Arc<DaemonHandle>,
}

/// daemon 的可共享句柄 — 持有 shared state / bus / llm 的 Arc
pub struct DaemonHandle {
    pub shared: Arc<Mutex<crate::daemon::SharedState>>,
    pub bus: Arc<crate::message_bus::MessageBus>,
    pub config: DaemonConfig,
    pub start_time: std::time::Instant,
    /// LlmExecutor 热切换句柄（None = daemon 用 CLI driver 或无 llm，不可热切）
    pub llm_override: Option<Arc<std::sync::RwLock<Option<crate::genome::LlmConfig>>>>,
    /// LLM driver 句柄（None = CLI driver，无法测试连接）
    pub llm: Option<Arc<dyn crate::driver::EvolutionDriver>>,
    /// 任务编排器（None = 无 LLM，无法执行任务）
    pub task_orchestrator: Option<Arc<crate::task_orchestrator::TaskOrchestrator>>,
    /// LLM 熔断器句柄(None = CLI driver,无熔断器)
    pub breaker: Option<Arc<crate::llm_health::LlmCircuitBreaker>>,
}

impl DaemonHandle {
    /// 只读查询：尝试在短超时内取锁，撞上归因窗口返回 None（调用方转 503）
    pub async fn try_read<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&crate::daemon::SharedState) -> T,
    {
        match tokio::time::timeout(Duration::from_millis(1500), self.shared.lock()).await {
            Ok(guard) => Some(f(&guard)),
            Err(_) => None,
        }
    }

    /// 可变查询：同上，给 feedback/exec 用
    pub async fn try_write<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&mut crate::daemon::SharedState) -> T,
    {
        match tokio::time::timeout(Duration::from_millis(3000), self.shared.lock()).await {
            Ok(mut guard) => Some(f(&mut guard)),
            Err(_) => None,
        }
    }
}

/// 启动 HTTP server（localhost，与 Unix socket 并存）
pub async fn start_http_server(handle: Arc<DaemonHandle>, port: u16) -> Result<(), String> {
    let state = HttpState { daemon: handle };
    let app = Router::new()
        .route("/api/status", get(get_status))
        .route("/api/capabilities", get(get_capabilities))
        .route("/api/feedback", post(post_feedback))
        .route("/api/exec", post(post_exec))
        .route("/api/config", get(get_config).post(post_config))
        .route("/api/integrations/status", get(get_integrations_status))
        .route(
            "/api/integrations/bootstrap",
            post(post_integrations_bootstrap),
        )
        .route("/api/test_llm", post(post_test_llm))
        .route("/api/task", post(post_task))
        .route("/api/tasks", get(get_tasks))
        .route("/api/task/:id/feedback", post(post_task_feedback))
        .route("/api/evolution", get(get_evolution_stats))
        .route("/api/llm_health", get(get_llm_health))
        .route("/api/research", post(post_research))
        .route("/api/projects", get(get_projects))
        .route("/api/initiative", post(post_initiative))
        .route("/api/explorer", post(post_explorer))
        .route(
            "/api/experiments",
            get(get_experiments).post(post_experiments),
        )
        .route("/api/workspace/graph", get(get_workspace_graph))
        .route("/api/autonomy/status", get(get_autonomy_status))
        .route("/api/autonomy/decisions", get(get_autonomy_decisions))
        .route("/api/autonomy/prompts", get(get_autonomy_prompts))
        .route("/api/learning-agenda", get(get_learning_agenda))
        .route(
            "/api/autonomy/prompts/:id/approve",
            post(approve_autonomy_prompt),
        )
        .route(
            "/api/autonomy/prompts/:id/reject",
            post(reject_autonomy_prompt),
        )
        .route(
            "/api/autonomy/prompts/:id/dismiss",
            post(dismiss_autonomy_prompt),
        )
        .route("/api/autonomy/pause", post(pause_autonomy))
        .route("/api/autonomy/resume", post(resume_autonomy))
        .route("/api/projects/memory", post(post_project_memory))
        .route("/api/projects/execute", post(post_project_execute))
        .route(
            "/api/projects/proposals/:id/feedback",
            post(post_project_proposal_feedback),
        )
        .route("/api/projects/tasks", get(get_project_tasks))
        .route("/api/projects/runs", get(get_durable_runs))
        .route("/api/workers/status", get(get_worker_pool_status))
        .route("/api/projects/runs/:id/events", get(get_durable_run_events))
        .route("/api/projects/runs/:id/retry", post(retry_durable_run))
        .route(
            "/api/projects/tasks/:id/feedback",
            post(post_project_task_feedback),
        )
        .route(
            "/api/projects/tasks/:id/outcome",
            post(post_project_task_outcome),
        )
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    tracing::info!("HTTP API 启动: http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("HTTP 绑定 {} 失败: {}", addr, e))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("HTTP server 错误: {}", e))?;
    Ok(())
}

async fn post_research(
    State(state): State<HttpState>,
    Json(request): Json<crate::research::ResearchRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let engine = crate::research::ResearchEngine::new(&state.daemon.config.storage_dir);
    let result = engine.research(request).await;
    Ok(Json(serde_json::to_value(result).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("研究结果序列化失败: {}", e),
        )
    })?))
}

async fn get_integrations_status(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let status = crate::integrations::detect_integrations().await;
    let resources = crate::integrations::load_cloud_resources(&state.daemon.config.storage_dir);
    Ok(Json(
        json!({"success": true, "integrations": status, "resources": resources}),
    ))
}

async fn post_integrations_bootstrap(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let resources = crate::cloud_sync::sync_personal_cloud(&state.daemon.config.storage_dir)
        .await
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(json!({"success": true, "resources": resources})))
}

#[derive(Deserialize)]
struct InitiativeReq {
    confidence: f64,
    value: f64,
    risk: f64,
    attention_cost: f64,
}

async fn post_initiative(Json(req): Json<InitiativeReq>) -> Json<Value> {
    Json(
        serde_json::to_value(crate::initiative::decide(
            req.confidence,
            req.value,
            req.risk,
            req.attention_cost,
        ))
        .unwrap_or_else(|_| json!({"error":"决策序列化失败"})),
    )
}

async fn get_workspace_graph(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let roots = crate::project_worker::configured_project_roots();
    let graph = crate::workspace::observe(&roots, &state.daemon.config.storage_dir).await;
    Ok(Json(serde_json::to_value(graph).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?))
}

async fn get_autonomy_status(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let autonomy = crate::autonomy_controller::load_state(&state.daemon.config.storage_dir);
    Ok(Json(json!({"success": true, "autonomy": autonomy})))
}

async fn get_autonomy_decisions(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let autonomy = crate::autonomy_controller::load_state(&state.daemon.config.storage_dir);
    Ok(Json(
        json!({"success": true, "decisions": autonomy.decisions}),
    ))
}

async fn get_autonomy_prompts(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let autonomy = crate::autonomy_controller::load_state(&state.daemon.config.storage_dir);
    Ok(Json(json!({"success": true, "prompts": autonomy.prompts})))
}

async fn get_learning_agenda(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let agenda = crate::learning_agenda::load(&state.daemon.config.storage_dir);
    Ok(Json(json!({"success": true, "agenda": agenda})))
}

async fn approve_autonomy_prompt(
    State(state): State<HttpState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut autonomy = crate::autonomy_controller::load_state(&state.daemon.config.storage_dir);
    let prompt = autonomy
        .prompts
        .iter_mut()
        .find(|prompt| prompt.id == id)
        .ok_or((StatusCode::NOT_FOUND, "自主提示不存在".into()))?;
    if prompt.status != "pending" {
        return Err((
            StatusCode::CONFLICT,
            format!("提示当前状态为 {}，不能批准", prompt.status),
        ));
    }
    prompt.status = "approved".into();
    prompt.updated_at = crate::project_worker::now_for_memory();
    let project_path = prompt.project_path.clone();
    let task = prompt.task.clone();
    let verify = prompt.verify_command.clone();
    let proposal_id = prompt.proposal_id.clone();
    let decision_id = autonomy
        .decisions
        .iter()
        .rev()
        .find(|decision| {
            decision.proposal_id == proposal_id && decision.project_path == project_path
        })
        .map(|decision| decision.id.clone());
    if let Some(decision) = autonomy.decisions.iter_mut().rev().find(|decision| {
        decision.proposal_id == proposal_id && decision.project_path == project_path
    }) {
        decision.status = "approved".into();
        decision.detail = "用户批准，已加入隔离项目任务队列".into();
    }
    crate::autonomy_controller::save_state(&state.daemon.config.storage_dir, &autonomy)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let task_id = format!("project-{}", uuid::Uuid::new_v4());
    let run = crate::durable_run::enqueue_project_run(
        state.daemon.config.storage_dir.clone(),
        state.daemon.shared.clone(),
        state.daemon.bus.clone(),
        crate::durable_run::ProjectRunSpec {
            id: task_id.clone(),
            source: "autonomy_prompt".into(),
            project_path,
            task,
            proposal_id: Some(proposal_id),
            verify_command: verify,
            decision_id,
        },
    )
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(
        json!({"success": true, "prompt_id": id, "task_id": task_id, "status": run.status, "durable": true}),
    ))
}

async fn reject_autonomy_prompt(
    State(state): State<HttpState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut autonomy = crate::autonomy_controller::load_state(&state.daemon.config.storage_dir);
    let prompt = autonomy
        .prompts
        .iter_mut()
        .find(|prompt| prompt.id == id)
        .ok_or((StatusCode::NOT_FOUND, "自主提示不存在".into()))?;
    if prompt.status != "pending" {
        return Err((
            StatusCode::CONFLICT,
            format!("提示当前状态为 {}，不能拒绝", prompt.status),
        ));
    }
    prompt.status = "rejected".into();
    prompt.updated_at = crate::project_worker::now_for_memory();
    let proposal_id = prompt.proposal_id.clone();
    let project_path = prompt.project_path.clone();
    let title = prompt.title.clone();
    let task = prompt.task.clone();
    let category = prompt.category.clone();
    if let Some(decision) = autonomy.decisions.iter_mut().rev().find(|decision| {
        decision.proposal_id == proposal_id && decision.project_path == project_path
    }) {
        decision.status = "rejected".into();
        decision.detail = "用户拒绝，已记录为项目长期反馈".into();
    }
    crate::autonomy_controller::save_state(&state.daemon.config.storage_dir, &autonomy)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let memory_path = crate::project_worker::project_memory_path_for(
        &state.daemon.config.storage_dir,
        &project_path,
    );
    let mut memory = crate::project_worker::load_project_memory_for(
        &state.daemon.config.storage_dir,
        &project_path,
    );
    if !memory
        .rejected_goals
        .iter()
        .any(|event| event.proposal_id == proposal_id)
    {
        memory
            .rejected_goals
            .push(crate::project_worker::ProjectMemoryEvent {
                proposal_id: proposal_id.clone(),
                title,
                task,
                recorded_at: crate::project_worker::now_for_memory(),
            });
        memory.updated_at = Some(crate::project_worker::now_for_memory());
        crate::project_worker::save_project_memory_for(&memory_path, &memory)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    }
    let profile =
        crate::value_energy::record_feedback(&state.daemon.config.storage_dir, &category, false)
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let preference_weight = profile.preference_weight(&category);
    Ok(Json(json!({
        "success": true,
        "prompt_id": id,
        "status": "rejected",
        "category": category,
        "preference_weight": preference_weight
    })))
}

/// Remove a proposal from future interruptions without treating it as a
/// negative preference signal for the whole category.
async fn dismiss_autonomy_prompt(
    State(state): State<HttpState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut autonomy = crate::autonomy_controller::load_state(&state.daemon.config.storage_dir);
    let prompt = autonomy
        .prompts
        .iter_mut()
        .find(|prompt| prompt.id == id)
        .ok_or((StatusCode::NOT_FOUND, "自主提示不存在".into()))?;
    if prompt.status != "pending" {
        return Err((
            StatusCode::CONFLICT,
            format!("提示当前状态为 {}，不能忽略", prompt.status),
        ));
    }
    prompt.status = "dismissed".into();
    prompt.updated_at = crate::project_worker::now_for_memory();
    let proposal_id = prompt.proposal_id.clone();
    let project_path = prompt.project_path.clone();
    if let Some(decision) = autonomy.decisions.iter_mut().rev().find(|decision| {
        decision.proposal_id == proposal_id && decision.project_path == project_path
    }) {
        decision.status = "dismissed".into();
        decision.detail = "用户在批量确认中取消选择；不执行，也不计为类别负反馈".into();
    }
    crate::autonomy_controller::save_state(&state.daemon.config.storage_dir, &autonomy)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(
        json!({"success": true, "prompt_id": id, "status": "dismissed"}),
    ))
}

async fn pause_autonomy(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let autonomy = crate::autonomy_controller::set_paused(&state.daemon.config.storage_dir, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"success": true, "autonomy": autonomy})))
}

async fn resume_autonomy(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let autonomy = crate::autonomy_controller::set_paused(&state.daemon.config.storage_dir, false)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({"success": true, "autonomy": autonomy})))
}

#[derive(Deserialize)]
struct ExplorerReq {
    project_path: String,
    objective: String,
    #[serde(default = "default_variant_count")]
    max_variants: usize,
}

fn default_variant_count() -> usize {
    4
}

async fn post_explorer(
    State(state): State<HttpState>,
    Json(req): Json<ExplorerReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let engine = crate::experiments::ExperimentEngine::new(&state.daemon.config.storage_dir);
    let result = engine
        .explore(
            &req.project_path,
            &req.objective,
            state.daemon.llm.clone(),
            req.max_variants,
        )
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(serde_json::to_value(result).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?))
}

async fn get_experiments(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    Ok(Json(
        json!({"success":true,"batches":crate::experiments::load_batches(&state.daemon.config.storage_dir)}),
    ))
}

async fn post_experiments(
    State(state): State<HttpState>,
    Json(req): Json<crate::experiments::ExperimentRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let batch_id = format!("experiment-{}", uuid::Uuid::new_v4());
    let batch_id_for_job = batch_id.clone();
    let engine = crate::experiments::ExperimentEngine::new(&state.daemon.config.storage_dir);
    let storage = state.daemon.config.storage_dir.clone();
    let request = req.clone();
    tokio::spawn(async move {
        let _ = engine.run_batch(&batch_id_for_job, &request).await;
        let _ = storage;
    });
    Ok(Json(
        json!({"success":true,"batch_id":batch_id,"status":"queued"}),
    ))
}

async fn get_projects(State(state): State<HttpState>) -> Result<Json<Value>, (StatusCode, String)> {
    let roots = crate::project_worker::configured_project_roots();
    let worker = crate::project_worker::ProjectWorker::new(state.daemon.config.storage_dir.clone());
    // The polling endpoint must never wait on remote research or an LLM. It
    // returns static health/opportunity signals and recent cached proposals,
    // then refreshes model proposals in a single background job.
    let projects = worker.discover_projects_fast(&roots).await;
    let refreshing = if state.daemon.llm.is_some() {
        try_start_project_refresh(&state.daemon.config.storage_dir)
    } else {
        false
    };
    if refreshing {
        let storage = state.daemon.config.storage_dir.clone();
        let roots_for_job = roots.clone();
        let llm = state.daemon.llm.clone();
        tokio::spawn(async move {
            let worker = crate::project_worker::ProjectWorker::new(storage.clone());
            let _ = worker
                .discover_projects_with_driver(&roots_for_job, llm)
                .await;
            finish_project_refresh(&storage);
        });
    }
    Ok(Json(
        json!({"success": true, "roots": roots, "projects": projects, "proposals_refreshing": refreshing}),
    ))
}

fn project_refresh_marker(storage_dir: &std::path::Path) -> std::path::PathBuf {
    storage_dir.join("project_goals").join(".refreshing")
}

fn try_start_project_refresh(storage_dir: &std::path::Path) -> bool {
    let marker = project_refresh_marker(storage_dir);
    let Some(parent) = marker.parent() else {
        return false;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return false;
    }
    if let Ok(metadata) = std::fs::metadata(&marker) {
        let stale = metadata
            .modified()
            .ok()
            .and_then(|time| time.elapsed().ok())
            .map(|elapsed| elapsed > std::time::Duration::from_secs(900))
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_file(&marker);
        }
    }
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(marker)
        .is_ok()
}

fn finish_project_refresh(storage_dir: &std::path::Path) {
    let _ = std::fs::remove_file(project_refresh_marker(storage_dir));
}

#[derive(Deserialize)]
struct ProjectMemoryReq {
    project_path: String,
    #[serde(default)]
    vision: String,
    #[serde(default)]
    priorities: Vec<String>,
}

async fn post_project_memory(
    State(state): State<HttpState>,
    Json(req): Json<ProjectMemoryReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut memory = crate::project_worker::load_project_memory_for(
        &state.daemon.config.storage_dir,
        &req.project_path,
    );
    memory.project_path = req.project_path.clone();
    memory.vision = req.vision.trim().chars().take(5000).collect();
    memory.priorities = req
        .priorities
        .into_iter()
        .map(|p| p.trim().chars().take(200).collect::<String>())
        .filter(|p| !p.is_empty())
        .take(12)
        .collect();
    memory.updated_at = Some(crate::project_worker::now_for_memory());
    crate::project_worker::save_project_memory_for(
        &crate::project_worker::project_memory_path_for(
            &state.daemon.config.storage_dir,
            &req.project_path,
        ),
        &memory,
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let cache_path = state
        .daemon
        .config
        .storage_dir
        .join("project_goals")
        .join(format!(
            "{}.json",
            crate::project_worker::project_memory_key(&req.project_path)
        ));
    let _ = std::fs::remove_file(cache_path);
    Ok(Json(json!({"success":true,"memory":memory})))
}

#[derive(Deserialize)]
struct ProjectExecuteReq {
    project_path: String,
    task: String,
    #[serde(default)]
    proposal_id: Option<String>,
    #[serde(default)]
    verify_command: Option<String>,
}

#[derive(Deserialize)]
struct ProjectProposalFeedbackReq {
    project_path: String,
    title: String,
    task: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    useful: bool,
}

async fn post_project_proposal_feedback(
    State(state): State<HttpState>,
    Path(id): Path<String>,
    Json(req): Json<ProjectProposalFeedbackReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let memory_path = crate::project_worker::project_memory_path_for(
        &state.daemon.config.storage_dir,
        &req.project_path,
    );
    let mut memory = crate::project_worker::load_project_memory_for(
        &state.daemon.config.storage_dir,
        &req.project_path,
    );
    let category = if req.category.trim().is_empty() {
        crate::value_energy::infer_category(&format!("{} {}", req.title, req.task))
    } else {
        crate::value_energy::normalize_category(&req.category)
    };
    let event = crate::project_worker::ProjectMemoryEvent {
        proposal_id: id.clone(),
        title: req.title,
        task: req.task,
        recorded_at: crate::project_worker::now_for_memory(),
    };
    if req.useful {
        if !memory
            .completed_goals
            .iter()
            .any(|existing| existing.proposal_id == id)
        {
            memory.completed_goals.push(event);
        }
    } else if !memory
        .rejected_goals
        .iter()
        .any(|existing| existing.proposal_id == id)
    {
        memory.rejected_goals.push(event);
    }
    memory.updated_at = Some(crate::project_worker::now_for_memory());
    crate::project_worker::save_project_memory_for(&memory_path, &memory)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let update = crate::value_energy::record_feedback_for_project(
        &state.daemon.config.storage_dir,
        &req.project_path,
        &category,
        req.useful,
    )
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(json!({
        "success":true,
        "proposal_id":id,
        "useful":req.useful,
        "category":category,
        "preference_weight":update.effect.after_weight,
        "feedback_effect":update.effect
    })))
}

/// Approve a proposal and execute it in an isolated worktree through pi.
async fn post_project_execute(
    State(state): State<HttpState>,
    Json(req): Json<ProjectExecuteReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let task_id = format!("project-{}", uuid::Uuid::new_v4());
    let run = crate::durable_run::enqueue_project_run(
        state.daemon.config.storage_dir.clone(),
        state.daemon.shared.clone(),
        state.daemon.bus.clone(),
        crate::durable_run::ProjectRunSpec {
            id: task_id.clone(),
            source: "manual_approval".into(),
            project_path: req.project_path,
            task: req.task,
            proposal_id: req.proposal_id,
            verify_command: req.verify_command,
            decision_id: None,
        },
    )
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(
        json!({"success":true,"task_id":task_id,"status":run.status,"durable":true}),
    ))
}

async fn get_durable_runs(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let store = crate::durable_run::DurableRunStore::new(&state.daemon.config.storage_dir)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let runs = store
        .list_runs(250)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(json!({"success":true,"runs":runs})))
}

async fn get_worker_pool_status(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let store = crate::durable_run::DurableRunStore::new(&state.daemon.config.storage_dir)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let pool = store
        .worker_pool_status()
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    let research_pool = crate::research::research_worker_pool_status();
    Ok(Json(
        json!({"success": true, "pool": pool, "research_pool": research_pool}),
    ))
}

async fn get_durable_run_events(
    State(state): State<HttpState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let store = crate::durable_run::DurableRunStore::new(&state.daemon.config.storage_dir)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    if store
        .get_run(&id)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?
        .is_none()
    {
        return Err((StatusCode::NOT_FOUND, "持久任务不存在".into()));
    }
    let events = store
        .events(&id)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
    Ok(Json(json!({"success":true,"run_id":id,"events":events})))
}

async fn retry_durable_run(
    State(state): State<HttpState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let run = crate::durable_run::retry_project_run(
        state.daemon.config.storage_dir.clone(),
        state.daemon.shared.clone(),
        state.daemon.bus.clone(),
        &id,
    )
    .map_err(|error| (StatusCode::CONFLICT, error))?;
    Ok(Json(json!({
        "success": true,
        "run_id": id,
        "status": run.status,
        "attempt": run.attempt,
    })))
}

async fn get_project_tasks(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let dir = state.daemon.config.storage_dir.join("project_tasks");
    let mut tasks = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if !entry.path().extension().is_some_and(|ext| ext == "json") {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                if let Ok(mut value) = serde_json::from_str::<Value>(&content) {
                    let filename = entry.file_name().to_string_lossy().to_string();
                    if filename.ends_with(".running.json") {
                        value["status"] = Value::String("running".into());
                    } else if filename.ends_with(".queued.json") {
                        value["status"] = Value::String("queued".into());
                    } else if value.get("status").is_none() && value.get("result").is_some() {
                        value["status"] = Value::String("completed".into());
                    }
                    if let Some(result) = value.get_mut("result").and_then(Value::as_object_mut) {
                        if result.get("executor").is_none() {
                            result.insert("executor".into(), Value::String("pi".into()));
                            result.insert("used_capabilities".into(), Value::Array(Vec::new()));
                            result.insert(
                                "attribution_status".into(),
                                Value::String("legacy_inferred_pi".into()),
                            );
                            if let Some(validation) = result
                                .get_mut("real_validation")
                                .and_then(Value::as_object_mut)
                            {
                                validation.insert(
                                    "recorded_capabilities".into(),
                                    Value::Array(Vec::new()),
                                );
                            }
                        }
                    }
                    tasks.push(value);
                }
            }
        }
    }
    Ok(Json(json!({"success":true,"tasks":tasks})))
}

#[derive(Deserialize)]
struct ProjectTaskFeedbackReq {
    useful: bool,
    #[serde(default)]
    note: String,
}

fn project_feedback_attribution(
    result: &serde_json::Map<String, Value>,
) -> (Vec<String>, &'static str) {
    let post_change = result
        .get("capability_trace")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|invocation| {
            invocation.get("phase").and_then(Value::as_str) == Some("post_change")
                && invocation.get("success").and_then(Value::as_bool) == Some(true)
        })
        .filter_map(|invocation| invocation.get("capability").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    if post_change.len() == 1 {
        return (
            post_change.into_iter().collect(),
            "single_successful_post_change_capability",
        );
    }
    if post_change.len() > 1 {
        return (Vec::new(), "ambiguous_multiple_capabilities");
    }

    let declared = result
        .get("used_capabilities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    let legacy =
        result.get("attribution_status").and_then(Value::as_str) == Some("legacy_inferred_pi");
    if declared.len() == 1 && !legacy {
        return (declared.into_iter().collect(), "single_declared_capability");
    }
    if declared.len() > 1 {
        (Vec::new(), "ambiguous_multiple_capabilities")
    } else {
        (Vec::new(), "no_attributable_capability")
    }
}

/// Apply human value feedback to the project skills recorded on a completed
/// project task. Feedback is single-use and persisted with the task result.
async fn post_project_task_feedback(
    State(state): State<HttpState>,
    Path(id): Path<String>,
    Json(req): Json<ProjectTaskFeedbackReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let path = state
        .daemon
        .config
        .storage_dir
        .join("project_tasks")
        .join(format!("{}.json", id));
    let content = std::fs::read_to_string(&path)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("任务不存在: {}", e)))?;
    let mut envelope: Value = serde_json::from_str(&content).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("任务记录损坏: {}", e),
        )
    })?;
    if envelope.get("status").and_then(Value::as_str) != Some("completed") {
        return Err((StatusCode::CONFLICT, "只有已完成项目任务可以评价".into()));
    }
    let result = envelope
        .get_mut("result")
        .and_then(Value::as_object_mut)
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "任务结果缺失".into()))?;
    if result.get("feedback").is_some_and(|value| !value.is_null()) {
        return Err((StatusCode::CONFLICT, "该项目任务已经评价过".into()));
    }
    let (attributed_capabilities, feedback_attribution_status) =
        project_feedback_attribution(result);
    let rated_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default();
    let feedback = crate::project_worker::ProjectFeedback {
        useful: req.useful,
        note: req.note.clone(),
        rated_at,
    };

    let project_path = result
        .get("project_path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let task_text = result
        .get("task")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let mut state_guard = state.daemon.shared.lock().await;
    let previous_fitness = attributed_capabilities
        .iter()
        .filter_map(|candidate| {
            state_guard
                .evolution
                .genomes()
                .get(candidate)
                .map(|genome| (candidate.to_string(), genome.fitness.clone()))
        })
        .collect::<Vec<_>>();
    let mut recorded = Vec::new();
    for candidate in &attributed_capabilities {
        if let Some(genome) = state_guard.evolution.genomes_mut().get_mut(candidate) {
            genome.fitness.record_human_signal(req.useful);
            recorded.push(candidate.to_string());
        }
    }
    if !recorded.is_empty() {
        if let Err(error) = state_guard.evolution.save_fitness() {
            for (candidate, fitness) in previous_fitness {
                if let Some(genome) = state_guard.evolution.genomes_mut().get_mut(&candidate) {
                    genome.fitness = fitness;
                }
            }
            let _ = state_guard.evolution.save_fitness();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("反馈持久化失败，已回滚: {}", error),
            ));
        }
    }
    result.insert("feedback".into(), serde_json::to_value(feedback).unwrap());
    result.insert(
        "feedback_attribution_status".into(),
        Value::String(feedback_attribution_status.into()),
    );
    result.insert(
        "feedback_recorded_capabilities".into(),
        Value::Array(
            recorded
                .iter()
                .map(|name| Value::String(name.clone()))
                .collect(),
        ),
    );
    let category = crate::value_energy::infer_category(&task_text);
    let value_profile_update = crate::value_energy::record_feedback_for_project(
        &state.daemon.config.storage_dir,
        &project_path,
        &category,
        req.useful,
    )
    .ok();
    if let Some(update) = &value_profile_update {
        result.insert(
            "feedback_effect".into(),
            serde_json::to_value(&update.effect).unwrap_or(Value::Null),
        );
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&envelope).unwrap()).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("任务反馈写入失败: {}", e),
        )
    })?;
    if let Ok(store) = crate::durable_run::DurableRunStore::new(&state.daemon.config.storage_dir) {
        if store.get_run(&id).ok().flatten().is_some() {
            store
                .update_completed_result(
                    &id,
                    &envelope,
                    "feedback.recorded",
                    &json!({
                        "useful": req.useful,
                        "attribution_status": feedback_attribution_status,
                        "recorded_capabilities": recorded,
                    }),
                )
                .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
        }
    }
    drop(state_guard);
    let _ = crate::project_worker::record_project_memory_feedback(
        &state.daemon.config.storage_dir,
        &project_path,
        &id,
        &task_text,
        req.useful,
        &req.note,
    );
    Ok(Json(json!({
        "success":true,
        "task_id":id,
        "recorded_capabilities":recorded,
        "feedback_attribution_status":feedback_attribution_status,
        "value_category":category,
        "value_profile_updated":value_profile_update.is_some(),
        "feedback_effect":value_profile_update.map(|update| update.effect)
    })))
}

#[derive(Deserialize)]
struct ProjectTaskOutcomeReq {
    horizon_days: u32,
    status: String,
    #[serde(default)]
    note: String,
}

fn validate_long_term_outcome(
    horizon_days: u32,
    status: &str,
    rated_at: u64,
    now: u64,
) -> Result<bool, String> {
    if !matches!(horizon_days, 7 | 30) {
        return Err("长期结果仅支持 7 天或 30 天复核".into());
    }
    let useful = match status {
        "adopted" | "still_using" => true,
        "rolled_back" => false,
        _ => return Err("结果状态必须是 adopted、still_using 或 rolled_back".into()),
    };
    let due_at = rated_at.saturating_add(horizon_days as u64 * 24 * 60 * 60);
    if now < due_at {
        return Err(format!("{} 天复核尚未到期", horizon_days));
    }
    Ok(useful)
}

async fn post_project_task_outcome(
    State(state): State<HttpState>,
    Path(id): Path<String>,
    Json(req): Json<ProjectTaskOutcomeReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let path = state
        .daemon
        .config
        .storage_dir
        .join("project_tasks")
        .join(format!("{}.json", id));
    let content = std::fs::read_to_string(&path)
        .map_err(|error| (StatusCode::NOT_FOUND, format!("任务不存在: {}", error)))?;
    let mut envelope: Value = serde_json::from_str(&content).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("任务记录损坏: {}", error),
        )
    })?;
    if envelope.get("status").and_then(Value::as_str) != Some("completed") {
        return Err((StatusCode::CONFLICT, "只有已完成项目任务可以复核".into()));
    }
    let result = envelope
        .get_mut("result")
        .and_then(Value::as_object_mut)
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "任务结果缺失".into()))?;
    let rated_at = result
        .get("feedback")
        .and_then(|feedback| feedback.get("rated_at"))
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or((StatusCode::CONFLICT, "请先提交即时有用性反馈".into()))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    let useful = validate_long_term_outcome(req.horizon_days, &req.status, rated_at, now)
        .map_err(|error| (StatusCode::CONFLICT, error))?;
    let outcomes = result
        .entry("long_term_outcomes")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "长期结果记录格式损坏".into(),
        ))?;
    if outcomes.iter().any(|outcome| {
        outcome.get("horizon_days").and_then(Value::as_u64) == Some(req.horizon_days as u64)
    }) {
        return Err((StatusCode::CONFLICT, "该周期已经完成复核".into()));
    }
    let outcome = json!({
        "horizon_days": req.horizon_days,
        "status": req.status,
        "note": req.note,
        "recorded_at": now,
    });
    outcomes.push(outcome.clone());

    let project_path = result
        .get("project_path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let task_text = result
        .get("task")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let category = crate::value_energy::infer_category(&task_text);
    let update = crate::value_energy::record_feedback_for_project(
        &state.daemon.config.storage_dir,
        &project_path,
        &category,
        useful,
    )
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;

    std::fs::write(&path, serde_json::to_vec_pretty(&envelope).unwrap()).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("长期结果写入失败: {}", error),
        )
    })?;
    if let Ok(store) = crate::durable_run::DurableRunStore::new(&state.daemon.config.storage_dir) {
        if store.get_run(&id).ok().flatten().is_some() {
            store
                .update_completed_result(&id, &envelope, "outcome.recorded", &outcome)
                .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error))?;
        }
    }
    Ok(Json(json!({
        "success": true,
        "task_id": id,
        "outcome": outcome,
        "feedback_effect": update.effect,
    })))
}

async fn get_status(State(state): State<HttpState>) -> Result<Json<Value>, (StatusCode, String)> {
    match state
        .daemon
        .try_read(|s| {
            json!({
                "pid": std::process::id(),
                "capabilities": s.evolution.genomes().len(),
                "total_evolutions": s.total_evolutions,
                "uptime_secs": state.daemon.start_time.elapsed().as_secs(),
            })
        })
        .await
    {
        Some(v) => Ok(Json(json!({"success": true, "status": v}))),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "进化忙（归因中），稍后重试".into(),
        )),
    }
}

async fn get_capabilities(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    match state
        .daemon
        .try_read(|s| {
            let caps: Vec<Value> = s
                .evolution
                .genomes()
                .values()
                .map(|g| {
                    json!({
                        "name": g.name,
                        "description": g.description,
                        "score": g.fitness.score,
                        "success_rate": g.fitness.success_rate,
                        "call_count": g.fitness.call_count,
                        "rounds_dormant": g.fitness.rounds_dormant,
                        "human_score": g.fitness.human_score,
                        "human_signals_count": g.fitness.human_signals_count,
                        "real_validation_passes": g.fitness.real_validation_passes,
                        "real_validation_failures": g.fitness.real_validation_failures,
                        "utility_score": g.fitness.utility_score,
                        "innovation_score": g.fitness.innovation_score,
                        "strongest_signal": format!("{:?}", g.fitness.strongest_signal),
                        "total_token_cost": g.fitness.total_token_cost,
                        "last_token_cost": g.fitness.last_token_cost,
                        "profit_ratio": g.fitness.profit_ratio,
                        "actions": g.actions.iter().map(|a| a.name.clone()).collect::<Vec<_>>(),
                    })
                })
                .collect();
            json!({"success": true, "capabilities": caps})
        })
        .await
    {
        Some(v) => Ok(Json(v)),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "进化忙（归因中），稍后重试".into(),
        )),
    }
}

#[derive(Deserialize)]
struct FeedbackReq {
    capability: String,
    useful: bool,
    #[serde(default)]
    note: String,
}

async fn post_feedback(
    State(state): State<HttpState>,
    Json(req): Json<FeedbackReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    match state
        .daemon
        .try_write(|s| -> Result<Value, (StatusCode, String)> {
            let previous_fitness = s
                .evolution
                .genomes()
                .get(&req.capability)
                .map(|g| g.fitness.clone())
                .ok_or_else(|| {
                    (
                        StatusCode::NOT_FOUND,
                        format!("未找到能力: {}", req.capability),
                    )
                })?;
            let previous_memory = s.evolution.memory().clone();

            let result = {
                let g = s.evolution.genomes_mut().get_mut(&req.capability).unwrap();
                let prev_score = g.fitness.score;
                let prev_signals = g.fitness.human_signals_count;
                g.fitness.record_human_signal(req.useful);
                json!({
                    "ok": true,
                    "capability": req.capability,
                    "useful": req.useful,
                    "prev_score": prev_score,
                    "new_score": g.fitness.score,
                    "human_signals_count": g.fitness.human_signals_count,
                    "human_score": g.fitness.human_score,
                    "prev_signals": prev_signals,
                })
            };

            // 评分决定选择方向；文字反馈解释“为什么”，供后续归因和变异读取。
            let note = req.note.trim();
            if !note.is_empty() {
                s.evolution
                    .record_lesson(crate::evolution::EvolutionLesson {
                        lesson: format!(
                            "人类反馈「{}」：{}（{}）",
                            req.capability,
                            note,
                            if req.useful { "有用" } else { "无用" }
                        ),
                        capability: req.capability.clone(),
                        failure_type: if req.useful {
                            "human_useful".into()
                        } else {
                            "human_useless".into()
                        },
                        learned_at: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs().to_string())
                            .unwrap_or_default(),
                        referenced_count: 0,
                    });
            }

            let persist_result = s.evolution.save_fitness().and_then(|_| {
                if note.is_empty() {
                    Ok(())
                } else {
                    s.evolution.save_memory()
                }
            });
            if let Err(e) = persist_result {
                // 对内存和已成功写入的一侧做最佳努力回滚，且绝不向客户端谎报成功。
                if let Some(g) = s.evolution.genomes_mut().get_mut(&req.capability) {
                    g.fitness = previous_fitness;
                }
                *s.evolution.memory_mut() = previous_memory;
                let _ = s.evolution.save_fitness();
                let _ = s.evolution.save_memory();
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("反馈持久化失败，已回滚: {}", e),
                ));
            }

            Ok(result)
        })
        .await
    {
        Some(Ok(v)) => Ok(Json(json!({"success": true, "result": v}))),
        Some(Err(e)) => Err(e),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "进化忙（归因中），稍后重试".into(),
        )),
    }
}

#[derive(Deserialize)]
struct ExecReq {
    capability: String,
    action: String,
    #[serde(default)]
    input: Value,
}

async fn post_exec(
    State(state): State<HttpState>,
    Json(req): Json<ExecReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    // exec 不需要 shared 锁（bus 是独立 Arc），可直接执行——避免被归因窗口阻塞
    let msg = crate::message::Message::builder()
        .from("http_api")
        .to(&req.capability)
        .action(&req.action)
        .payload(req.input)
        .build();
    match state.daemon.bus.send(msg).await {
        Ok(resp) => Ok(Json(json!({"success": true, "payload": resp.payload}))),
        Err(e) => Ok(Json(json!({"success": false, "error": e.to_string()}))),
    }
}

async fn get_config(State(state): State<HttpState>) -> Json<Value> {
    // 优先从 LlmExecutor 的覆盖层读真实生效配置；回退环境变量
    if let Some(override_h) = &state.daemon.llm_override {
        if let Ok(guard) = override_h.read() {
            let cfg = guard.clone().unwrap_or_else(|| fallback_config_from_env());
            return Json(json!({
                "success": true,
                "config": config_json(&cfg),
                "has_override": guard.is_some(),
            }));
        }
    }
    let cfg = fallback_config_from_env();
    Json(json!({
        "success": true,
        "config": config_json(&cfg),
        "has_override": false,
        "note": "daemon 未使用 LlmExecutor（可能为 CLI driver），仅返回环境变量配置"
    }))
}

/// GET /api/llm_health — 返回熔断器状态 + 当前是否在 API 开放时段内
///
/// 供 Dashboard 展示 LLM 可用性徽章。返回结构：
/// - state: closed / open / half_open / unknown（无熔断器时）
/// - in_active_hours: 当前是否在 active_hours 窗口内（无窗口 = true）
/// - active_hours: "HH:MM-HH:MM" 或 null
/// - consecutive_failures: 连续失败数
/// - opened_at: 熔断打开的 RFC3339 时间戳或 null
async fn get_llm_health(State(state): State<HttpState>) -> Json<Value> {
    let breaker = state.daemon.breaker.as_ref();
    let snapshot = breaker.map(|b| b.snapshot());
    // 读 active_hours + 判断当前是否在窗口内
    let (in_active, active_hours_str) = match &state.daemon.llm_override {
        Some(h) => {
            let cfg = h
                .read()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_else(|| fallback_config_from_env());
            let in_aw = cfg
                .active_hours
                .as_ref()
                .map(|tw| tw.contains_now())
                .unwrap_or(true);
            let aw_str = cfg
                .active_hours
                .as_ref()
                .map(|tw| format!("{}-{}", tw.start, tw.end));
            (in_aw, aw_str)
        }
        None => (true, None),
    };
    // 拆分借用：先算 state_str（借用），再取 failures（借用），最后 move snapshot 取 opened_at
    let state_str = snapshot
        .as_ref()
        .map(|s| match &s.state {
            crate::llm_health::BreakerState::Closed => "closed",
            crate::llm_health::BreakerState::Open => "open",
            crate::llm_health::BreakerState::HalfOpen => "half_open",
        })
        .unwrap_or("unknown");
    let failures = snapshot
        .as_ref()
        .map(|s| s.consecutive_failures)
        .unwrap_or(0);
    let opened = snapshot.and_then(|s| s.opened_at);
    Json(json!({
        "success": true,
        "state": state_str,
        "in_active_hours": in_active,
        "active_hours": active_hours_str,
        "consecutive_failures": failures,
        "opened_at": opened,
    }))
}

#[derive(Deserialize)]
struct ConfigReq {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    /// 通用模型名 — 若 fast_model/smart_model/coder_model 未指定则都用此值
    #[serde(default)]
    model: Option<String>,
    /// 分角色模型 — 优先于 model
    #[serde(default)]
    fast_model: Option<String>,
    #[serde(default)]
    smart_model: Option<String>,
    #[serde(default)]
    coder_model: Option<String>,
    /// 按角色指定供应商连接；传入任一角色的 api_key/base_url 即启用多供应商模式
    #[serde(default)]
    fast_api_key: Option<String>,
    #[serde(default)]
    fast_base_url: Option<String>,
    #[serde(default)]
    smart_api_key: Option<String>,
    #[serde(default)]
    smart_base_url: Option<String>,
    #[serde(default)]
    coder_api_key: Option<String>,
    #[serde(default)]
    coder_base_url: Option<String>,
    /// 来自设置页全部预设的有序备用连接。
    #[serde(default)]
    fallback_configs: Option<Vec<crate::genome::LlmRoleConfig>>,
    /// API 开放时段 {start, end}(本地 HH:MM);设置时传完整对象
    #[serde(default)]
    active_hours: Option<crate::llm_health::TimeWindow>,
    /// true = 清除 active_hours(显式清除,因为 Option 反序列化无法区分"未传"和"传null")
    #[serde(default)]
    active_hours_clear: bool,
    /// true = 清除覆盖，回退启动配置
    #[serde(default)]
    reset: bool,
}

/// 热切换 API 配置 — 写入 LlmExecutor 覆盖层，下一次 LLM 调用即生效，不重启 daemon
async fn post_config(
    State(state): State<HttpState>,
    Json(req): Json<ConfigReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let override_h = state
        .daemon
        .llm_override
        .as_ref()
        .ok_or((
            StatusCode::CONFLICT,
            "daemon 未使用 LlmExecutor，无法热切换（CLI driver 不支持）".into(),
        ))?
        .clone();

    // 先取当前生效配置作为基础，再应用请求里的覆盖字段
    let current = match override_h.read() {
        Ok(g) => g.clone().unwrap_or_else(|| fallback_config_from_env()),
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("读锁失败: {}", e),
            ))
        }
    };
    let has_role_overrides = req.fast_api_key.is_some()
        || req.fast_base_url.is_some()
        || req.smart_api_key.is_some()
        || req.smart_base_url.is_some()
        || req.coder_api_key.is_some()
        || req.coder_base_url.is_some();
    let fallback_configs = req
        .fallback_configs
        .clone()
        .unwrap_or_else(|| current.fallback_configs.clone())
        .into_iter()
        .filter(|item| {
            !item.api_key.trim().is_empty()
                && !item.base_url.trim().is_empty()
                && !item.model.trim().is_empty()
        })
        .take(24)
        .collect::<Vec<_>>();
    let new_cfg = if req.reset {
        // 清除覆盖 → 回退启动配置
        if let Ok(mut g) = override_h.write() {
            *g = None;
        }
        // 删除持久化文件
        let path = persisted_config_path(&state.daemon.config.storage_dir);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        return Ok(Json(
            json!({"success": true, "reset": true, "message": "已回退到启动配置"}),
        ));
    } else if has_role_overrides {
        let fast = role_config_from_request(
            &current,
            "fast",
            req.fast_api_key,
            req.fast_base_url,
            req.fast_model,
        );
        let smart = role_config_from_request(
            &current,
            "smart",
            req.smart_api_key,
            req.smart_base_url,
            req.smart_model,
        );
        let coder = role_config_from_request(
            &current,
            "coder",
            req.coder_api_key,
            req.coder_base_url,
            req.coder_model,
        );
        crate::genome::LlmConfig {
            // 顶层字段保留为 fast 角色的兼容镜像；新代码读取 fast_config 等角色字段。
            api_key: fast.api_key.clone(),
            base_url: fast.base_url.clone(),
            fast_model: fast.model.clone(),
            smart_model: smart.model.clone(),
            coder_model: coder.model.clone(),
            fast_config: Some(fast),
            smart_config: Some(smart),
            coder_config: Some(coder),
            fallback_configs: fallback_configs.clone(),
            active_hours: if req.active_hours_clear {
                None
            } else {
                req.active_hours.clone().or(current.active_hours.clone())
            },
        }
    } else {
        // 优先级：分角色字段 > 通用 model > 当前值
        let m = req.model.as_deref();
        crate::genome::LlmConfig {
            api_key: req.api_key.unwrap_or(current.api_key),
            base_url: req.base_url.unwrap_or(current.base_url),
            fast_model: req
                .fast_model
                .or_else(|| m.map(|s| s.to_string()))
                .unwrap_or(current.fast_model),
            smart_model: req
                .smart_model
                .or_else(|| m.map(|s| s.to_string()))
                .unwrap_or(current.smart_model),
            coder_model: req
                .coder_model
                .or_else(|| m.map(|s| s.to_string()))
                .unwrap_or(current.coder_model),
            // 使用旧版共享字段更新时，清除按角色配置，保持旧 API 语义。
            fast_config: None,
            smart_config: None,
            coder_config: None,
            fallback_configs,
            active_hours: if req.active_hours_clear {
                None
            } else {
                req.active_hours.clone().or(current.active_hours.clone())
            },
        }
    };
    if new_cfg.api_key.is_empty()
        || new_cfg.role_config("smart").api_key.is_empty()
        || new_cfg.role_config("coder").api_key.is_empty()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "每个角色的 api_key 不能为空".into(),
        ));
    }
    match override_h.write() {
        Ok(mut g) => *g = Some(new_cfg.clone()),
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("写锁失败: {}", e),
            ))
        }
    }
    // 持久化到磁盘，daemon 重启后自动加载
    if let Err(e) = persist_config(&state.daemon.config.storage_dir, &new_cfg) {
        tracing::warn!("配置热切换成功但持久化失败: {}", e);
    }
    Ok(Json(json!({
        "success": true,
        "message": "配置已热切换，下一次 LLM 调用生效",
        "config": config_json(&new_cfg),
    })))
}

fn role_config_from_request(
    current: &crate::genome::LlmConfig,
    role: &str,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
) -> crate::genome::LlmRoleConfig {
    let previous = current.role_config(role);
    crate::genome::LlmRoleConfig {
        api_key: api_key.unwrap_or(previous.api_key),
        base_url: base_url.unwrap_or(previous.base_url),
        model: model.unwrap_or(previous.model),
    }
}

#[derive(Deserialize)]
struct TestLlmReq {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

/// 测试 LLM 连接 — 用（可选覆盖后的）配置发一次 ping 调用，返回耗时与回复
async fn post_test_llm(
    State(state): State<HttpState>,
    Json(req): Json<TestLlmReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let driver = state
        .daemon
        .llm
        .as_ref()
        .ok_or((
            StatusCode::CONFLICT,
            "daemon 未使用 LlmExecutor，无法测试（CLI driver 不支持）".into(),
        ))?
        .clone();

    // 若请求带了字段，临时写入覆盖层测一次，测完恢复原值；否则用当前生效配置测
    let override_h = state.daemon.llm_override.clone();
    let snapshot_before = override_h
        .as_ref()
        .and_then(|h| h.read().ok())
        .and_then(|g| g.clone());

    let has_overrides = req.api_key.is_some() || req.base_url.is_some() || req.model.is_some();
    if has_overrides {
        if let Some(h) = &override_h {
            let current = h
                .read()
                .map(|g| g.clone().unwrap_or_else(fallback_config_from_env))
                .unwrap_or_else(|_| fallback_config_from_env());
            let model = req
                .model
                .clone()
                .unwrap_or_else(|| current.fast_model.clone());
            let test_cfg = crate::genome::LlmConfig {
                api_key: req.api_key.clone().unwrap_or(current.api_key),
                base_url: req.base_url.clone().unwrap_or(current.base_url),
                fast_model: model.clone(),
                smart_model: model.clone(),
                coder_model: model,
                fast_config: None,
                smart_config: None,
                coder_config: None,
                fallback_configs: current.fallback_configs.clone(),
                active_hours: None,
            };
            if test_cfg.api_key.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "api_key 不能为空".into()));
            }
            if let Ok(mut g) = h.write() {
                *g = Some(test_cfg);
            }
        }
    }

    let resolved_model = override_h
        .as_ref()
        .and_then(|h| h.read().ok())
        .and_then(|g| g.as_ref().map(|c| c.fast_model.clone()))
        .unwrap_or_default();

    let start = std::time::Instant::now();
    let call = driver.execute("ping: 请回复 ok", &resolved_model, None);
    let result = tokio::time::timeout(Duration::from_secs(30), call).await;
    let elapsed_secs = start.elapsed().as_secs_f64();

    // 恢复覆盖层原值（若上面做了临时覆盖）
    if has_overrides {
        if let Some(h) = &override_h {
            if let Ok(mut g) = h.write() {
                *g = snapshot_before;
            }
        }
    }

    match result {
        Ok(Ok(reply)) => Ok(Json(json!({
            "success": true,
            "status": "ok",
            "result": reply,
            "model": resolved_model,
            "elapsed_secs": (elapsed_secs * 1000.0).round() / 1000.0,
        }))),
        Ok(Err(e)) => Ok(Json(json!({
            "success": false,
            "status": "error",
            "error": e,
            "model": resolved_model,
            "elapsed_secs": (elapsed_secs * 1000.0).round() / 1000.0,
        }))),
        Err(_) => Ok(Json(json!({
            "success": false,
            "status": "error",
            "error": "请求超时（30s），请检查 API URL 是否可达或网络连接",
            "model": resolved_model,
            "elapsed_secs": 30.0,
        }))),
    }
}

pub(crate) fn fallback_config_from_env() -> crate::genome::LlmConfig {
    let model = std::env::var("ORCH_MODEL").unwrap_or_default();
    crate::genome::LlmConfig {
        api_key: std::env::var("ORCH_API_KEY").unwrap_or_default(),
        base_url: std::env::var("ORCH_BASE_URL")
            .or_else(|_| std::env::var("ORCH_API_BASE_URL"))
            .unwrap_or_default(),
        fast_model: std::env::var("ORCH_MODEL_FAST").unwrap_or_else(|_| model.clone()),
        smart_model: std::env::var("ORCH_MODEL_SMART").unwrap_or_else(|_| model.clone()),
        coder_model: std::env::var("ORCH_MODEL_CODER").unwrap_or_else(|_| model),
        fast_config: None,
        smart_config: None,
        coder_config: None,
        fallback_configs: Vec::new(),
        active_hours: None,
    }
}

fn config_json(cfg: &crate::genome::LlmConfig) -> Value {
    let fast = cfg.role_config("fast");
    let smart = cfg.role_config("smart");
    let coder = cfg.role_config("coder");
    json!({
        // 顶层字段保留兼容性，默认代表 fast 角色。
        "model": fast.model,
        "base_url": fast.base_url,
        "has_key": !fast.api_key.is_empty(),
        "fast_model": fast.model,
        "smart_model": smart.model,
        "coder_model": coder.model,
        "roles": {
            "fast": {
                "model": fast.model,
                "base_url": fast.base_url,
                "has_key": !fast.api_key.is_empty(),
            },
            "smart": {
                "model": smart.model,
                "base_url": smart.base_url,
                "has_key": !smart.api_key.is_empty(),
            },
            "coder": {
                "model": coder.model,
                "base_url": coder.base_url,
                "has_key": !coder.api_key.is_empty(),
            },
        },
        "fallbacks": cfg.fallback_configs.iter().map(|item| json!({
            "model": item.model,
            "base_url": item.base_url,
            "has_key": !item.api_key.is_empty(),
        })).collect::<Vec<_>>(),
        "active_hours": cfg.active_hours.as_ref().map(|tw| json!({
            "start": tw.start,
            "end": tw.end,
        })),
    })
}

// ── 配置持久化 ──────────────────────────────────────────────

/// 持久化配置文件路径
fn persisted_config_path(storage_dir: &std::path::Path) -> std::path::PathBuf {
    storage_dir.join("llm_config.json")
}

/// 将配置写入磁盘（daemon 重启后自动加载）
fn persist_config(
    storage_dir: &std::path::Path,
    cfg: &crate::genome::LlmConfig,
) -> Result<(), String> {
    let path = persisted_config_path(storage_dir);
    let json = serde_json::to_string_pretty(cfg).map_err(|e| format!("序列化失败: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("写入失败: {}: {}", path.display(), e))
}

/// 从磁盘加载已持久化的配置（daemon 启动时调用）
pub fn load_persisted_config(storage_dir: &std::path::Path) -> Option<crate::genome::LlmConfig> {
    let path = persisted_config_path(storage_dir);
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data)
        .map_err(|e| {
            tracing::warn!("持久化配置解析失败: {}: {}", path.display(), e);
            e
        })
        .ok()
}

// ===== 自主进化全景 API =====

/// GET /api/evolution — 返回完整进化统计，供 UI 自主进化页面展示
///
/// 返回结构：
/// - global_stats: 全局进化指标（变异总数、成功数、创造数、淘汰数、自主目标数）
/// - autonomous_history: 最近自主目标执行记录
/// - tried_mutations: 最近变异尝试记录（父代→子代、成功/失败、变异方案）
/// - lessons: 最近进化教训
/// - thought_chains: 最近思维链（LLM 推理过程）
/// - top_capabilities: 按得分排序的能力快照（含 human_score，反映自主判定结果）
async fn get_evolution_stats(
    State(state): State<HttpState>,
) -> Result<Json<Value>, (StatusCode, String)> {
    match state
        .daemon
        .try_read(|s| {
            let mem = s.evolution.memory();

            // 全局统计
            let stats = &mem.global_stats;
            let mutation_success_rate = if stats.total_mutations > 0 {
                stats.total_mutation_successes as f64 / stats.total_mutations as f64
            } else {
                0.0
            };
            let autonomous_success_rate = if stats.total_autonomous_goals > 0 {
                stats.total_autonomous_successes as f64 / stats.total_autonomous_goals as f64
            } else {
                0.0
            };

            let global_stats = json!({
                "total_rounds": stats.total_rounds,
                "total_created": stats.total_created,
                "total_eliminated": stats.total_eliminated,
                "total_mutations": stats.total_mutations,
                "total_mutation_successes": stats.total_mutation_successes,
                "mutation_success_rate": (mutation_success_rate * 10000.0).round() / 10000.0,
                "total_autonomous_goals": stats.total_autonomous_goals,
                "total_autonomous_successes": stats.total_autonomous_successes,
                "autonomous_success_rate": (autonomous_success_rate * 10000.0).round() / 10000.0,
                "first_boot_ts": stats.first_boot_ts,
                "rounds_since_last_creation": stats.rounds_since_last_creation,
            });

            // 自主目标历史（最近 50 条，倒序）
            let autonomous_history: Vec<Value> = mem
                .autonomous_history
                .iter()
                .rev()
                .take(50)
                .map(|e| {
                    json!({
                        "goal": e.goal,
                        "success": e.success,
                        "capabilities_used": e.capabilities_used,
                        "elapsed_ms": e.elapsed_ms,
                        "timestamp": e.timestamp,
                    })
                })
                .collect();

            // 变异时间线（最近 50 条，倒序）
            let tried_mutations: Vec<Value> = mem
                .tried_mutations
                .iter()
                .rev()
                .take(50)
                .map(|m| {
                    json!({
                        "capability": m.capability,
                        "mutation_type": m.mutation_type,
                        "description": m.description,
                        "success": m.success,
                        "tried_at": m.tried_at,
                    })
                })
                .collect();

            // 教训库（最近 50 条，倒序）
            let lessons: Vec<Value> = mem
                .lessons
                .iter()
                .rev()
                .take(50)
                .map(|l| {
                    json!({
                        "lesson": l.lesson,
                        "capability": l.capability,
                        "failure_type": l.failure_type,
                        "learned_at": l.learned_at,
                        "referenced_count": l.referenced_count,
                    })
                })
                .collect();

            // 思维链（最近 20 条，倒序）
            let thought_chains: Vec<Value> = mem
                .thought_chains
                .iter()
                .rev()
                .take(20)
                .map(|c| {
                    json!({
                        "chain_type": c.chain_type,
                        "reasoning": c.reasoning,
                        "conclusion": c.conclusion,
                        "related_capabilities": c.related_capabilities,
                        "related_goal": c.related_goal,
                        "success": c.success,
                        "timestamp": c.timestamp,
                    })
                })
                .collect();

            // 能力排行（按 score 降序，含 human_score 反映 LLM 自主判定结果）
            let mut top_caps: Vec<Value> = s
                .evolution
                .genomes()
                .values()
                .map(|g| {
                    json!({
                        "name": g.name,
                        "description": g.description,
                        "score": g.fitness.score,
                        "success_rate": g.fitness.success_rate,
                        "call_count": g.fitness.call_count,
                        "human_score": g.fitness.human_score,
                        "human_signals_count": g.fitness.human_signals_count,
                        "strongest_signal": format!("{:?}", g.fitness.strongest_signal),
                        "rounds_dormant": g.fitness.rounds_dormant,
                    })
                })
                .collect();
            top_caps.sort_by(|a, b| {
                b.get("score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
                    .partial_cmp(&a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            json!({
                "global_stats": global_stats,
                "autonomous_history": autonomous_history,
                "tried_mutations": tried_mutations,
                "lessons": lessons,
                "thought_chains": thought_chains,
                "top_capabilities": top_caps,
                "total_evolutions": s.total_evolutions,
                "uptime_secs": state.daemon.start_time.elapsed().as_secs(),
            })
        })
        .await
    {
        Some(v) => Ok(Json(json!({"success": true, "evolution": v}))),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "进化忙（归因中），稍后重试".into(),
        )),
    }
}

// ===== 任务驱动进化 API =====

#[derive(Deserialize)]
struct TaskReq {
    description: String,
}

/// POST /api/task — 提交自然语言任务，执行 Plan-Execute-Observe 循环
async fn post_task(
    State(state): State<HttpState>,
    Json(req): Json<TaskReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let orchestrator = state
        .daemon
        .task_orchestrator
        .as_ref()
        .ok_or((
            StatusCode::CONFLICT,
            "daemon 未配置 LLM，无法执行任务（请先配置 API）".into(),
        ))?
        .clone();

    // 任务执行可能较久，在后台执行并立即返回 task_id
    let result = orchestrator
        .run_task(&req.description)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(json!({"success": true, "task": result})))
}

/// GET /api/tasks — 获取最近的任务列表
async fn get_tasks(State(state): State<HttpState>) -> Result<Json<Value>, (StatusCode, String)> {
    let orchestrator = state
        .daemon
        .task_orchestrator
        .as_ref()
        .ok_or((StatusCode::CONFLICT, "daemon 未配置 LLM，无任务历史".into()))?
        .clone();

    let tasks = orchestrator.list_tasks().await;
    Ok(Json(json!({"success": true, "tasks": tasks})))
}

#[derive(Deserialize)]
struct TaskFeedbackReq {
    success: bool,
    #[serde(default)]
    note: String,
}

/// POST /api/task/:id/feedback — 对任务结果评价，驱动能力进化
async fn post_task_feedback(
    State(state): State<HttpState>,
    Path(id): Path<String>,
    Json(req): Json<TaskFeedbackReq>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let orchestrator = state
        .daemon
        .task_orchestrator
        .as_ref()
        .ok_or((
            StatusCode::CONFLICT,
            "daemon 未配置 LLM，无法处理反馈".into(),
        ))?
        .clone();

    let feedback = crate::task_orchestrator::TaskFeedback {
        success: req.success,
        note: req.note,
        rated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_default(),
    };

    orchestrator
        .apply_feedback(&id, feedback)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(
        json!({"success": true, "task_id": id, "message": "反馈已应用，能力进化已驱动"}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_feedback_only_attributes_a_unique_successful_capability() {
        let result = serde_json::json!({
            "used_capabilities": ["cap-a", "cap-b"],
            "capability_trace": [
                {"capability":"cap-a","phase":"post_change","success":true},
                {"capability":"cap-b","phase":"baseline","success":true}
            ]
        });
        let (capabilities, status) = project_feedback_attribution(result.as_object().unwrap());
        assert_eq!(capabilities, vec!["cap-a"]);
        assert_eq!(status, "single_successful_post_change_capability");
    }

    #[test]
    fn project_feedback_keeps_multi_capability_value_at_task_level() {
        let result = serde_json::json!({
            "used_capabilities": ["cap-a", "cap-b"],
            "capability_trace": [
                {"capability":"cap-a","phase":"post_change","success":true},
                {"capability":"cap-b","phase":"post_change","success":true}
            ]
        });
        let (capabilities, status) = project_feedback_attribution(result.as_object().unwrap());
        assert!(capabilities.is_empty());
        assert_eq!(status, "ambiguous_multiple_capabilities");
    }

    #[test]
    fn long_term_outcomes_require_supported_horizon_and_elapsed_time() {
        let day = 24 * 60 * 60;
        assert!(validate_long_term_outcome(7, "still_using", 100, 100 + 7 * day).unwrap());
        assert!(!validate_long_term_outcome(30, "rolled_back", 100, 100 + 30 * day).unwrap());
        assert!(validate_long_term_outcome(7, "adopted", 100, 100 + 6 * day).is_err());
        assert!(validate_long_term_outcome(14, "adopted", 100, 100 + 30 * day).is_err());
        assert!(validate_long_term_outcome(7, "unknown", 100, 100 + 30 * day).is_err());
    }

    #[tokio::test]
    async fn human_feedback_persists_signal_and_explanation() {
        let storage = tempfile::tempdir().expect("创建反馈测试目录");
        let mut evolution = crate::evolution::EvolutionEngine::new(storage.path());
        evolution
            .register_genome(crate::genome::CapabilityGenome::new(
                "feedback-cap",
                "feedback test capability",
            ))
            .unwrap();

        let mut config = DaemonConfig::default();
        config.storage_dir = storage.path().to_path_buf();
        let handle = Arc::new(DaemonHandle {
            shared: Arc::new(Mutex::new(crate::daemon::SharedState {
                evolution,
                failure_driver: None,
                total_evolutions: 0,
            })),
            bus: Arc::new(crate::message_bus::MessageBus::new()),
            config,
            start_time: std::time::Instant::now(),
            llm_override: None,
            llm: None,
            task_orchestrator: None,
            breaker: None,
        });

        let response = post_feedback(
            State(HttpState { daemon: handle }),
            Json(FeedbackReq {
                capability: "feedback-cap".into(),
                useful: false,
                note: "结果可运行，但没有解决真实需求".into(),
            }),
        )
        .await
        .expect("反馈请求应成功");
        assert_eq!(response.0["success"], true);

        // 从磁盘新建引擎，证明评分与解释都不是只留在当前进程内存中。
        let reloaded = crate::evolution::EvolutionEngine::new(storage.path());
        let fitness = &reloaded.genomes()["feedback-cap"].fitness;
        assert_eq!(fitness.human_signals_count, 1);
        assert_eq!(fitness.human_score, 0.0);
        assert!(reloaded.memory().lessons.iter().any(|lesson| {
            lesson.capability == "feedback-cap"
                && lesson.failure_type == "human_useless"
                && lesson.lesson.contains("没有解决真实需求")
        }));
    }
}
