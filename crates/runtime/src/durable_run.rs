//! Durable project-task journal.
//!
//! SQLite is the source of truth for task state and ordered events. Legacy
//! `project_tasks/*.json` files are still emitted so existing UI and feedback
//! endpoints continue to work while they migrate to the journal.

use crate::message_bus::MessageBus;
use crate::project_worker::{ProjectTaskResult, ProjectWorker};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Semaphore};

static PROJECT_WORKER_POOL: OnceLock<Arc<Semaphore>> = OnceLock::new();
static PROJECT_WRITE_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Semaphore>>>> = OnceLock::new();

fn worker_pool() -> &'static Arc<Semaphore> {
    PROJECT_WORKER_POOL.get_or_init(|| {
        let concurrency = std::env::var("ORCH_PROJECT_WORKER_CONCURRENCY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);
        Arc::new(Semaphore::new(concurrency))
    })
}

fn worker_concurrency() -> usize {
    std::env::var("ORCH_PROJECT_WORKER_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 8)
}

async fn project_write_lock(project_path: &str) -> Arc<Semaphore> {
    let locks = PROJECT_WRITE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().await;
    locks
        .entry(project_path.to_string())
        .or_insert_with(|| Arc::new(Semaphore::new(1)))
        .clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectWorkerPoolStatus {
    pub max_workers: usize,
    pub available_workers: usize,
    pub running: usize,
    pub queued: usize,
    pub waiting_user: usize,
    pub active_projects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRunSpec {
    pub id: String,
    pub source: String,
    pub project_path: String,
    pub task: String,
    #[serde(default)]
    pub proposal_id: Option<String>,
    #[serde(default)]
    pub verify_command: Option<String>,
    #[serde(default)]
    pub decision_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableRun {
    pub id: String,
    pub kind: String,
    pub source: String,
    pub project_path: String,
    pub task: String,
    pub proposal_id: Option<String>,
    pub verify_command: Option<String>,
    pub decision_id: Option<String>,
    pub status: String,
    pub phase: String,
    pub attempt: u32,
    pub created_at: u64,
    pub updated_at: u64,
    pub last_error: Option<String>,
    pub result: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableRunEvent {
    pub seq: i64,
    pub run_id: String,
    pub event_type: String,
    pub phase: String,
    pub payload: Value,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct DurableRunStore {
    path: PathBuf,
}

impl DurableRunStore {
    pub fn new(storage_dir: impl AsRef<Path>) -> Result<Self, String> {
        let storage_dir = storage_dir.as_ref();
        std::fs::create_dir_all(storage_dir).map_err(|error| error.to_string())?;
        let store = Self {
            path: storage_dir.join("durable_runs.sqlite3"),
        };
        store.initialize()?;
        Ok(store)
    }

    fn connect(&self) -> Result<Connection, String> {
        let connection = Connection::open(&self.path).map_err(|error| error.to_string())?;
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|error| error.to_string())?;
        Ok(connection)
    }

    fn initialize(&self) -> Result<(), String> {
        let connection = self.connect()?;
        connection
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA foreign_keys=ON;
                 CREATE TABLE IF NOT EXISTS durable_runs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    project_path TEXT NOT NULL,
                    task TEXT NOT NULL,
                    proposal_id TEXT,
                    verify_command TEXT,
                    decision_id TEXT,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    last_error TEXT,
                    result_json TEXT
                 );
                 CREATE TABLE IF NOT EXISTS durable_run_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES durable_runs(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS idx_durable_runs_status
                    ON durable_runs(status, updated_at);
                 CREATE INDEX IF NOT EXISTS idx_durable_run_events_run
                    ON durable_run_events(run_id, seq);",
            )
            .map_err(|error| error.to_string())
    }

    pub fn create_project_run(&self, spec: &ProjectRunSpec) -> Result<DurableRun, String> {
        let now = unix_now();
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "INSERT INTO durable_runs
                 (id, kind, source, project_path, task, proposal_id, verify_command,
                  decision_id, status, phase, attempt, created_at, updated_at)
                 VALUES (?1, 'project_task', ?2, ?3, ?4, ?5, ?6, ?7,
                         'queued', 'queued', 0, ?8, ?8)",
                params![
                    spec.id,
                    spec.source,
                    spec.project_path,
                    spec.task,
                    spec.proposal_id,
                    spec.verify_command,
                    spec.decision_id,
                    now as i64,
                ],
            )
            .map_err(|error| format!("创建持久任务失败: {}", error))?;
        insert_event(
            &transaction,
            &spec.id,
            "run.queued",
            "queued",
            &json!({"source": spec.source, "project_path": spec.project_path}),
            now,
        )?;
        transaction.commit().map_err(|error| error.to_string())?;
        self.get_run(&spec.id)?
            .ok_or_else(|| "持久任务创建后无法读取".to_string())
    }

    pub fn get_run(&self, id: &str) -> Result<Option<DurableRun>, String> {
        let connection = self.connect()?;
        connection
            .query_row(
                "SELECT id, kind, source, project_path, task, proposal_id,
                        verify_command, decision_id, status, phase, attempt,
                        created_at, updated_at, last_error, result_json
                 FROM durable_runs WHERE id = ?1",
                params![id],
                row_to_run,
            )
            .optional()
            .map_err(|error| error.to_string())
    }

    pub fn list_runs(&self, limit: usize) -> Result<Vec<DurableRun>, String> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare(
                "SELECT id, kind, source, project_path, task, proposal_id,
                        verify_command, decision_id, status, phase, attempt,
                        created_at, updated_at, last_error, result_json
                 FROM durable_runs ORDER BY updated_at DESC LIMIT ?1",
            )
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![limit.clamp(1, 1000) as i64], row_to_run)
            .map_err(|error| error.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())
    }

    pub fn worker_pool_status(&self) -> Result<ProjectWorkerPoolStatus, String> {
        let runs = self.list_runs(1000)?;
        let mut active_projects = runs
            .iter()
            .filter(|run| run.status == "running")
            .map(|run| run.project_path.clone())
            .collect::<Vec<_>>();
        active_projects.sort();
        active_projects.dedup();
        Ok(ProjectWorkerPoolStatus {
            max_workers: worker_concurrency(),
            available_workers: worker_pool().available_permits(),
            running: runs.iter().filter(|run| run.status == "running").count(),
            queued: runs
                .iter()
                .filter(|run| matches!(run.status.as_str(), "queued" | "recovering"))
                .count(),
            waiting_user: runs
                .iter()
                .filter(|run| run.status == "waiting_user")
                .count(),
            active_projects,
        })
    }

    pub fn events(&self, run_id: &str) -> Result<Vec<DurableRunEvent>, String> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare(
                "SELECT seq, run_id, event_type, phase, payload_json, created_at
                 FROM durable_run_events WHERE run_id = ?1 ORDER BY seq ASC",
            )
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![run_id], |row| {
                let payload: String = row.get(4)?;
                Ok(DurableRunEvent {
                    seq: row.get(0)?,
                    run_id: row.get(1)?,
                    event_type: row.get(2)?,
                    phase: row.get(3)?,
                    payload: serde_json::from_str(&payload).unwrap_or(Value::Null),
                    created_at: row.get::<_, i64>(5)? as u64,
                })
            })
            .map_err(|error| error.to_string())?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())
    }

    pub fn append_event(
        &self,
        run_id: &str,
        event_type: &str,
        phase: &str,
        payload: &Value,
    ) -> Result<(), String> {
        let connection = self.connect()?;
        insert_event(&connection, run_id, event_type, phase, payload, unix_now())
    }

    pub fn update_completed_result(
        &self,
        run_id: &str,
        result: &Value,
        event_type: &str,
        payload: &Value,
    ) -> Result<(), String> {
        let now = unix_now();
        let result_json = serde_json::to_string(result).map_err(|error| error.to_string())?;
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        let changed = transaction
            .execute(
                "UPDATE durable_runs SET result_json = ?2, updated_at = ?3
                 WHERE id = ?1 AND status = 'completed'",
                params![run_id, result_json, now as i64],
            )
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err(format!("已完成持久任务不存在: {}", run_id));
        }
        insert_event(&transaction, run_id, event_type, "feedback", payload, now)?;
        transaction.commit().map_err(|error| error.to_string())
    }

    pub fn heartbeat(&self, run_id: &str) -> Result<(), String> {
        let now = unix_now();
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "UPDATE durable_runs SET updated_at = ?2 WHERE id = ?1 AND status = 'running'",
                params![run_id, now as i64],
            )
            .map_err(|error| error.to_string())?;
        insert_event(
            &transaction,
            run_id,
            "run.heartbeat",
            "executing",
            &json!({}),
            now,
        )?;
        transaction.commit().map_err(|error| error.to_string())
    }

    pub fn pause_for_user(&self, run_id: &str, result: &Value, error: &str) -> Result<(), String> {
        let now = unix_now();
        let result_json = serde_json::to_string(result).map_err(|error| error.to_string())?;
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "UPDATE durable_runs
                 SET status = 'waiting_user', phase = 'paused', updated_at = ?2,
                     last_error = ?3, result_json = ?4
                 WHERE id = ?1",
                params![run_id, now as i64, error, result_json],
            )
            .map_err(|error| error.to_string())?;
        insert_event(&transaction, run_id, "run.paused", "paused", result, now)?;
        transaction.commit().map_err(|error| error.to_string())
    }

    pub fn retry(&self, run_id: &str) -> Result<bool, String> {
        let now = unix_now();
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        let changed = transaction
            .execute(
                "UPDATE durable_runs
                 SET status = 'queued', phase = 'queued', updated_at = ?2,
                     last_error = NULL, result_json = NULL
                 WHERE id = ?1 AND status IN ('failed', 'timeout', 'waiting_user')",
                params![run_id, now as i64],
            )
            .map_err(|error| error.to_string())?;
        if changed > 0 {
            insert_event(
                &transaction,
                run_id,
                "run.retry_requested",
                "queued",
                &json!({"requested_by": "user"}),
                now,
            )?;
        }
        transaction.commit().map_err(|error| error.to_string())?;
        Ok(changed > 0)
    }

    pub fn claim(&self, run_id: &str) -> Result<bool, String> {
        let now = unix_now();
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        let changed = transaction
            .execute(
                "UPDATE durable_runs
                 SET status = 'running', phase = 'executing', attempt = attempt + 1,
                     updated_at = ?2, last_error = NULL
                 WHERE id = ?1 AND status IN ('queued', 'recovering')",
                params![run_id, now as i64],
            )
            .map_err(|error| error.to_string())?;
        if changed > 0 {
            insert_event(
                &transaction,
                run_id,
                "run.started",
                "executing",
                &json!({}),
                now,
            )?;
        }
        transaction.commit().map_err(|error| error.to_string())?;
        Ok(changed > 0)
    }

    pub fn finish(
        &self,
        run_id: &str,
        status: &str,
        event_type: &str,
        result: Option<&Value>,
        error: Option<&str>,
    ) -> Result<(), String> {
        let now = unix_now();
        let result_json = result
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| error.to_string())?;
        let mut connection = self.connect()?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "UPDATE durable_runs
                 SET status = ?2, phase = 'finished', updated_at = ?3,
                     last_error = ?4, result_json = ?5
                 WHERE id = ?1",
                params![run_id, status, now as i64, error, result_json],
            )
            .map_err(|error| error.to_string())?;
        insert_event(
            &transaction,
            run_id,
            event_type,
            "finished",
            result.unwrap_or(&json!({"error": error})),
            now,
        )?;
        transaction.commit().map_err(|error| error.to_string())
    }

    /// Requeue work that was interrupted by a daemon shutdown. `claim` keeps
    /// recovery idempotent when HTTP startup and other callers race.
    pub fn recover_incomplete(&self) -> Result<Vec<String>, String> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare(
                "SELECT id FROM durable_runs
                 WHERE status IN ('queued', 'running', 'recovering')
                 ORDER BY created_at ASC",
            )
            .map_err(|error| error.to_string())?;
        let ids = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        drop(statement);
        let now = unix_now();
        for id in &ids {
            connection
                .execute(
                    "UPDATE durable_runs SET status = 'recovering', phase = 'queued',
                     updated_at = ?2 WHERE id = ?1",
                    params![id, now as i64],
                )
                .map_err(|error| error.to_string())?;
            insert_event(
                &connection,
                id,
                "run.recovered",
                "queued",
                &json!({"reason": "daemon_restart"}),
                now,
            )?;
        }
        Ok(ids)
    }
}

pub fn enqueue_project_run(
    storage_dir: PathBuf,
    shared: Arc<Mutex<crate::daemon::SharedState>>,
    bus: Arc<MessageBus>,
    spec: ProjectRunSpec,
) -> Result<DurableRun, String> {
    let store = DurableRunStore::new(&storage_dir)?;
    let run = store.create_project_run(&spec)?;
    persist_legacy_state(&storage_dir, &run, "queued", None);
    spawn_project_run(storage_dir, shared, bus, run.id.clone());
    Ok(run)
}

pub fn spawn_project_run(
    storage_dir: PathBuf,
    shared: Arc<Mutex<crate::daemon::SharedState>>,
    bus: Arc<MessageBus>,
    run_id: String,
) {
    tokio::spawn(async move {
        if let Err(error) = execute_project_run(&storage_dir, shared, bus, &run_id).await {
            tracing::warn!(run_id = %run_id, "持久项目任务执行失败: {}", error);
        }
    });
}

pub fn recover_project_runs(
    storage_dir: PathBuf,
    shared: Arc<Mutex<crate::daemon::SharedState>>,
    bus: Arc<MessageBus>,
) -> Result<usize, String> {
    let ids = DurableRunStore::new(&storage_dir)?.recover_incomplete()?;
    let count = ids.len();
    for id in ids {
        spawn_project_run(storage_dir.clone(), shared.clone(), bus.clone(), id);
    }
    Ok(count)
}

pub fn retry_project_run(
    storage_dir: PathBuf,
    shared: Arc<Mutex<crate::daemon::SharedState>>,
    bus: Arc<MessageBus>,
    run_id: &str,
) -> Result<DurableRun, String> {
    let store = DurableRunStore::new(&storage_dir)?;
    if !store.retry(run_id)? {
        return Err("只有失败、超时或等待处理的任务可以继续".into());
    }
    let run = store
        .get_run(run_id)?
        .ok_or_else(|| format!("持久任务不存在: {}", run_id))?;
    persist_legacy_state(&storage_dir, &run, "queued", None);
    spawn_project_run(storage_dir, shared, bus, run.id.clone());
    Ok(run)
}

async fn execute_project_run(
    storage_dir: &Path,
    shared: Arc<Mutex<crate::daemon::SharedState>>,
    bus: Arc<MessageBus>,
    run_id: &str,
) -> Result<(), String> {
    let store = DurableRunStore::new(storage_dir)?;
    let queued_run = store
        .get_run(run_id)?
        .ok_or_else(|| format!("持久任务不存在: {}", run_id))?;
    let _ = store.append_event(
        run_id,
        "worker.waiting",
        "queued",
        &json!({
            "pool_size": worker_concurrency(),
            "project_path": queued_run.project_path,
        }),
    );
    let _worker_permit = worker_pool()
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| "项目 Worker Pool 已关闭".to_string())?;
    let project_lock = project_write_lock(&queued_run.project_path).await;
    let _project_permit = project_lock
        .acquire_owned()
        .await
        .map_err(|_| "项目写锁已关闭".to_string())?;
    if !store.claim(run_id)? {
        return Ok(());
    }
    let run = store
        .get_run(run_id)?
        .ok_or_else(|| format!("持久任务不存在: {}", run_id))?;
    persist_legacy_state(storage_dir, &run, "running", None);
    let sandbox_backend = crate::project_worker::project_pi_sandbox_backend();
    let _ = store.append_event(
        run_id,
        "executor.selected",
        "executing",
        &json!({"executor": "pi", "sandbox": sandbox_backend, "worker_pool_size": worker_concurrency(), "project_write_lock": true}),
    );

    let worker = ProjectWorker::new(storage_dir.to_path_buf()).with_bus(bus);
    let heartbeat_store = store.clone();
    let heartbeat_run_id = run_id.to_string();
    let heartbeat_secs = std::env::var("ORCH_RUN_HEARTBEAT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30)
        .max(5);
    let heartbeat = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(heartbeat_secs));
        interval.tick().await;
        loop {
            interval.tick().await;
            if heartbeat_store.heartbeat(&heartbeat_run_id).is_err() {
                break;
            }
        }
    });
    let durable_timeout_secs = std::env::var("ORCH_DURABLE_RUN_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1200)
        .max(60);
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(durable_timeout_secs),
        worker.run_with_task_id(
            &run.project_path,
            &run.task,
            run.verify_command.as_deref(),
            true,
            Some(run_id),
        ),
    )
    .await;
    heartbeat.abort();

    let envelope = match outcome {
        Ok(Ok(mut result)) => {
            result.task_id = run_id.to_string();
            result.proposal_id = run.proposal_id.clone();
            for invocation in &result.capability_trace {
                let _ = store.append_event(
                    run_id,
                    "capability.completed",
                    &invocation.phase,
                    &serde_json::to_value(invocation).unwrap_or(Value::Null),
                );
            }
            if let Some(verification) = result.verification.as_ref() {
                let _ = store.append_event(
                    run_id,
                    "verification.completed",
                    "verification",
                    &serde_json::to_value(verification).unwrap_or(Value::Null),
                );
            }
            crate::project_worker::record_project_validation(&shared, &mut result).await;
            let _ = crate::project_worker::record_project_task_outcome(storage_dir, &result);
            if let Some(decision_id) = run.decision_id.as_deref() {
                let detail = if result.applied {
                    "持久项目任务完成，验证后的改动已应用"
                } else {
                    "持久项目任务完成，改动保留在隔离 worktree"
                };
                let _ = crate::autonomy_controller::update_decision(
                    storage_dir,
                    decision_id,
                    "completed",
                    detail,
                );
            }
            completed_envelope(&run, result)
        }
        Ok(Err(error)) if error.contains("超时") => json!({
            "status":"waiting_user",
            "task_id":run_id,
            "source":run.source,
            "error":format!("{}；worktree 已保留，可继续执行", error),
            "resumable":true
        }),
        Ok(Err(error)) => {
            json!({"status":"failed", "task_id":run_id, "source":run.source, "error":error})
        }
        Err(_) => json!({
            "status":"waiting_user",
            "task_id":run_id,
            "source":run.source,
            "error":format!("项目 Agent 执行超过 {} 分钟，worktree 已保留，可继续执行", durable_timeout_secs / 60),
            "resumable":true
        }),
    };

    let status = envelope
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("failed");
    let error = envelope.get("error").and_then(Value::as_str);
    if status != "completed" {
        if let Some(decision_id) = run.decision_id.as_deref() {
            let decision_status = if status == "waiting_user" {
                "waiting_user"
            } else {
                "failed"
            };
            let _ = crate::autonomy_controller::update_decision(
                storage_dir,
                decision_id,
                decision_status,
                &format!(
                    "持久项目任务{}: {}",
                    if status == "waiting_user" {
                        "暂停等待用户处理"
                    } else {
                        "失败"
                    },
                    error.unwrap_or("未知错误")
                ),
            );
        }
    }
    if status == "waiting_user" {
        store.pause_for_user(run_id, &envelope, error.unwrap_or("任务等待用户处理"))?;
    } else {
        let event_type = if status == "completed" {
            "run.completed"
        } else {
            "run.failed"
        };
        store.finish(run_id, status, event_type, Some(&envelope), error)?;
    }
    persist_legacy_state(storage_dir, &run, status, Some(&envelope));
    Ok(())
}

fn completed_envelope(run: &DurableRun, result: ProjectTaskResult) -> Value {
    json!({
        "status": "completed",
        "task_id": run.id,
        "project_path": result.project_path,
        "task": result.task,
        "source": run.source,
        "result": result,
    })
}

fn persist_legacy_state(
    storage_dir: &Path,
    run: &DurableRun,
    status: &str,
    envelope: Option<&Value>,
) {
    let directory = storage_dir.join("project_tasks");
    if std::fs::create_dir_all(&directory).is_err() {
        return;
    }
    for suffix in ["queued.json", "running.json"] {
        let _ = std::fs::remove_file(directory.join(format!("{}.{}", run.id, suffix)));
    }
    let (filename, value) = match status {
        "queued" => (
            format!("{}.queued.json", run.id),
            json!({
                "task_id": run.id,
                "project_path": run.project_path,
                "task": run.task,
                "proposal_id": run.proposal_id,
                "source": run.source,
                "status": "queued",
            }),
        ),
        "running" => (
            format!("{}.running.json", run.id),
            json!({
                "task_id": run.id,
                "project_path": run.project_path,
                "task": run.task,
                "proposal_id": run.proposal_id,
                "source": run.source,
                "status": "running",
                "attempt": run.attempt,
            }),
        ),
        _ => (
            format!("{}.json", run.id),
            envelope.cloned().unwrap_or_else(|| {
                json!({
                    "task_id": run.id,
                    "status": status,
                })
            }),
        ),
    };
    if let Ok(bytes) = serde_json::to_vec_pretty(&value) {
        let temporary = directory.join(format!("{}.tmp", filename));
        if std::fs::write(&temporary, bytes).is_ok() {
            let _ = std::fs::rename(temporary, directory.join(filename));
        }
    }
}

fn insert_event(
    connection: &Connection,
    run_id: &str,
    event_type: &str,
    phase: &str,
    payload: &Value,
    created_at: u64,
) -> Result<(), String> {
    let payload = serde_json::to_string(payload).map_err(|error| error.to_string())?;
    connection
        .execute(
            "INSERT INTO durable_run_events
             (run_id, event_type, phase, payload_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![run_id, event_type, phase, payload, created_at as i64],
        )
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn row_to_run(row: &Row<'_>) -> rusqlite::Result<DurableRun> {
    let result_json: Option<String> = row.get(14)?;
    Ok(DurableRun {
        id: row.get(0)?,
        kind: row.get(1)?,
        source: row.get(2)?,
        project_path: row.get(3)?,
        task: row.get(4)?,
        proposal_id: row.get(5)?,
        verify_command: row.get(6)?,
        decision_id: row.get(7)?,
        status: row.get(8)?,
        phase: row.get(9)?,
        attempt: row.get::<_, i64>(10)? as u32,
        created_at: row.get::<_, i64>(11)? as u64,
        updated_at: row.get::<_, i64>(12)? as u64,
        last_error: row.get(13)?,
        result: result_json.and_then(|value| serde_json::from_str(&value).ok()),
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str) -> ProjectRunSpec {
        ProjectRunSpec {
            id: id.into(),
            source: "test".into(),
            project_path: "/tmp/project".into(),
            task: "run tests".into(),
            proposal_id: Some("proposal-1".into()),
            verify_command: Some("cargo test".into()),
            decision_id: None,
        }
    }

    #[test]
    fn durable_run_records_ordered_transitions() {
        let directory = tempfile::tempdir().unwrap();
        let store = DurableRunStore::new(directory.path()).unwrap();
        store.create_project_run(&spec("run-1")).unwrap();
        assert!(store.claim("run-1").unwrap());
        assert!(!store.claim("run-1").unwrap());
        store
            .append_event(
                "run-1",
                "verification.completed",
                "verification",
                &json!({"success":true}),
            )
            .unwrap();
        store
            .finish(
                "run-1",
                "completed",
                "run.completed",
                Some(&json!({"ok":true})),
                None,
            )
            .unwrap();
        let run = store.get_run("run-1").unwrap().unwrap();
        assert_eq!(run.status, "completed");
        assert_eq!(run.attempt, 1);
        let events = store.events("run-1").unwrap();
        assert_eq!(events[0].event_type, "run.queued");
        assert_eq!(events.last().unwrap().event_type, "run.completed");
    }

    #[test]
    fn interrupted_run_is_requeued_once() {
        let directory = tempfile::tempdir().unwrap();
        let store = DurableRunStore::new(directory.path()).unwrap();
        store.create_project_run(&spec("run-2")).unwrap();
        assert!(store.claim("run-2").unwrap());
        let recovered = store.recover_incomplete().unwrap();
        assert_eq!(recovered, vec!["run-2"]);
        assert_eq!(
            store.get_run("run-2").unwrap().unwrap().status,
            "recovering"
        );
        assert!(store.claim("run-2").unwrap());
        assert_eq!(store.get_run("run-2").unwrap().unwrap().attempt, 2);
    }

    #[test]
    fn paused_run_can_be_retried_without_losing_history() {
        let directory = tempfile::tempdir().unwrap();
        let store = DurableRunStore::new(directory.path()).unwrap();
        store.create_project_run(&spec("run-3")).unwrap();
        assert!(store.claim("run-3").unwrap());
        store.heartbeat("run-3").unwrap();
        store
            .pause_for_user(
                "run-3",
                &json!({"status":"waiting_user","resumable":true}),
                "timeout",
            )
            .unwrap();
        assert_eq!(
            store.get_run("run-3").unwrap().unwrap().status,
            "waiting_user"
        );
        assert!(store.retry("run-3").unwrap());
        assert_eq!(store.get_run("run-3").unwrap().unwrap().status, "queued");
        let events = store.events("run-3").unwrap();
        assert!(events
            .iter()
            .any(|event| event.event_type == "run.heartbeat"));
        assert!(events.iter().any(|event| event.event_type == "run.paused"));
        assert_eq!(events.last().unwrap().event_type, "run.retry_requested");
    }

    #[test]
    fn worker_pool_status_reports_running_and_queued_runs() {
        let directory = tempfile::tempdir().unwrap();
        let store = DurableRunStore::new(directory.path()).unwrap();
        store.create_project_run(&spec("pool-running")).unwrap();
        store.create_project_run(&spec("pool-queued")).unwrap();
        assert!(store.claim("pool-running").unwrap());
        let status = store.worker_pool_status().unwrap();
        assert_eq!(status.running, 1);
        assert_eq!(status.queued, 1);
        assert_eq!(status.active_projects, vec!["/tmp/project"]);
        assert!(status.max_workers >= 1);
    }

    #[tokio::test]
    async fn project_write_lock_serializes_same_project_only() {
        let project = format!("/tmp/project-lock-{}", uuid::Uuid::new_v4());
        let same_a = project_write_lock(&project).await;
        let same_b = project_write_lock(&project).await;
        let other = project_write_lock(&format!("{}-other", project)).await;
        let _first = same_a.clone().try_acquire_owned().unwrap();
        assert!(same_b.clone().try_acquire_owned().is_err());
        assert!(other.try_acquire_owned().is_ok());
    }
}
