import { useCallback, useEffect, useRef, useState } from "react"
import { CheckCircle2, ChevronDown, ChevronRight, Clock3, History, Loader2, RefreshCw, RotateCcw, ThumbsDown, ThumbsUp, XCircle } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

type Task = {
  task_id?: string
  status?: string
  project_path?: string
  task?: string
  error?: string
  source?: string
  phase?: string
  attempt?: number
  updated_at?: number
  durable?: boolean
  result?: { task_id?: string; worktree?: string; branch?: string; diff_stat?: string; executor?: string; attribution_status?: string; used_capabilities?: string[]; capability_trace?: { capability?: string; action?: string; phase?: string; output_summary?: string; success?: boolean; elapsed_ms?: number }[]; applied?: boolean; apply_error?: string; skill_candidates?: string[]; real_validation?: { success?: boolean; command?: string; evidence?: string; recorded_capabilities?: string[] }; feedback?: { useful?: boolean; note?: string; rated_at?: string }; feedback_effect?: { category?: string; before_weight?: number; after_weight?: number; confidence?: number; sample_count?: number }; long_term_outcomes?: Array<{ horizon_days: 7 | 30; status: "adopted" | "still_using" | "rolled_back"; note?: string; recorded_at: number }>; verification?: { success?: boolean; command?: string; stdout?: string; stderr?: string } }
}

type DurableRun = { id: string; source: string; project_path: string; task: string; status: string; phase: string; attempt: number; updated_at: number; last_error?: string | null; result?: Task | null }
type RunEvent = { seq: number; event_type: string; phase: string; payload?: unknown; created_at: number }
type WorkerPoolStatus = { max_workers: number; available_workers: number; running: number; queued: number; waiting_user: number; active_projects: string[] }
type ResearchWorkerPoolStatus = { max_workers: number; available_workers: number; running: number; queued: number; completed: number; failed: number }

function statusView(status: string) {
  if (status === "completed") return { label: "已完成", variant: "pos" as const, Icon: CheckCircle2 }
  if (status === "failed" || status === "timeout") return { label: status === "timeout" ? "超时" : "失败", variant: "neg" as const, Icon: XCircle }
  if (status === "waiting_user") return { label: "需要处理", variant: "warn" as const, Icon: Clock3 }
  if (status === "running" || status === "recovering") return { label: status === "recovering" ? "恢复中" : "执行中", variant: "info" as const, Icon: Loader2 }
  return { label: "排队中", variant: "warn" as const, Icon: Clock3 }
}

function eventLabel(event: RunEvent) {
  const labels: Record<string, string> = {
    "run.queued": "任务已进入持久队列",
    "worker.waiting": "等待 Worker 槽位和项目写锁",
    "run.started": "执行器开始工作",
    "run.recovered": "daemon 重启后恢复任务",
    "run.heartbeat": "执行器仍在工作",
    "run.paused": "任务暂停，等待用户处理",
    "run.retry_requested": "用户要求继续执行",
    "executor.selected": "已选择执行器和隔离环境",
    "capability.completed": "进化能力调用完成",
    "verification.completed": "项目验证完成",
    "run.completed": "任务完成",
    "run.failed": "任务失败",
  }
  return labels[event.event_type] ?? event.event_type
}

export function TasksPage() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)
  const [feedbackBusy, setFeedbackBusy] = useState<string | null>(null)
  const [outcomeBusy, setOutcomeBusy] = useState<string | null>(null)
  const [retryBusy, setRetryBusy] = useState<string | null>(null)
  const [expandedTask, setExpandedTask] = useState<string | null>(null)
  const [eventsByTask, setEventsByTask] = useState<Record<string, RunEvent[]>>({})
  const [workerPool, setWorkerPool] = useState<WorkerPoolStatus | null>(null)
  const [researchPool, setResearchPool] = useState<ResearchWorkerPoolStatus | null>(null)
  const refreshInFlight = useRef(false)

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return
    refreshInFlight.current = true
    try {
      const [taskResponse, runResponse, poolResponse] = await Promise.all([
        window.daemon.projectTasks(),
        window.daemon.projectRuns(),
        window.daemon.workerPoolStatus(),
      ])
      if (poolResponse.ok && poolResponse.status === 200) {
        const pools = poolResponse.data as { pool?: WorkerPoolStatus; research_pool?: ResearchWorkerPoolStatus }
        setWorkerPool(pools.pool ?? null)
        setResearchPool(pools.research_pool ?? null)
      }
      if (taskResponse.ok && taskResponse.status === 200) {
        const legacy = ((taskResponse.data as { tasks?: Task[] }).tasks ?? [])
        const byId = new Map(legacy.filter((task) => task.task_id).map((task) => [task.task_id as string, task]))
        const runs = runResponse.ok && runResponse.status === 200 ? ((runResponse.data as { runs?: DurableRun[] }).runs ?? []) : []
        const durable = runs.map((run): Task => {
          const envelope = run.result ?? undefined
          const previous = byId.get(run.id)
          byId.delete(run.id)
          return {
            ...previous,
            task_id: run.id,
            status: run.status,
            phase: run.phase,
            attempt: run.attempt,
            updated_at: run.updated_at,
            durable: true,
            source: run.source,
            project_path: run.project_path,
            task: run.task,
            error: run.last_error ?? envelope?.error ?? previous?.error,
            result: envelope?.result ?? previous?.result,
          }
        })
        const combined = [...durable, ...byId.values()]
        combined.sort((left, right) => (right.updated_at ?? 0) - (left.updated_at ?? 0))
        setTasks(combined)
      }
    } finally {
      refreshInFlight.current = false
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const id = window.setInterval(() => void refresh(), 3000)
    return () => window.clearInterval(id)
  }, [refresh])

  const sendFeedback = async (task: Task, useful: boolean) => {
    if (!task.task_id || task.result?.feedback) return
    setFeedbackBusy(task.task_id)
    try {
      await window.daemon.projectTaskFeedback(task.task_id, { useful })
      await refresh()
    } finally {
      setFeedbackBusy(null)
    }
  }

  const sendOutcome = async (task: Task, horizonDays: 7 | 30, status: "adopted" | "still_using" | "rolled_back") => {
    if (!task.task_id) return
    setOutcomeBusy(task.task_id)
    try {
      await window.daemon.projectTaskOutcome(task.task_id, { horizon_days: horizonDays, status })
      await refresh()
    } finally {
      setOutcomeBusy(null)
    }
  }

  const toggleEvents = async (task: Task) => {
    if (!task.task_id || !task.durable) return
    if (expandedTask === task.task_id) {
      setExpandedTask(null)
      return
    }
    setExpandedTask(task.task_id)
    const response = await window.daemon.projectRunEvents(task.task_id)
    if (response.ok && response.status === 200) {
      const data = response.data as { events?: RunEvent[] }
      setEventsByTask((current) => ({ ...current, [task.task_id as string]: data.events ?? [] }))
    }
  }

  const retryRun = async (task: Task) => {
    if (!task.task_id || !task.durable) return
    setRetryBusy(task.task_id)
    try {
      await window.daemon.retryProjectRun(task.task_id)
      await refresh()
      if (expandedTask === task.task_id) {
        const response = await window.daemon.projectRunEvents(task.task_id)
        if (response.ok && response.status === 200) {
          const data = response.data as { events?: RunEvent[] }
          setEventsByTask((current) => ({ ...current, [task.task_id as string]: data.events ?? [] }))
        }
      }
    } finally {
      setRetryBusy(null)
    }
  }

  return <div className="space-y-5">
    <div className="flex items-center justify-between">
      <div><h2 className="text-lg font-semibold">自动任务</h2><p className="mt-1 text-xs text-muted-foreground">查看自主发现、用户批准和隔离执行的项目任务</p></div>
      <Button size="sm" variant="ghost" onClick={() => void refresh()} className="gap-1.5"><RefreshCw className={loading ? "size-3.5 animate-spin" : "size-3.5"} />刷新</Button>
    </div>
    {workerPool && <Card className="p-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><div className="font-medium">项目 Worker Pool</div><div className="mt-1 text-xs text-muted-foreground">固定容量执行池；同一项目同时只允许一个写任务</div></div><Badge variant={workerPool.running > 0 ? "info" : "outline"}>{workerPool.running > 0 ? "运行中" : "空闲"}</Badge></div><div className="mt-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-5"><div><div className="text-xs text-muted-foreground">池容量</div><div className="mt-1 font-semibold">{workerPool.max_workers}</div></div><div><div className="text-xs text-muted-foreground">可用槽位</div><div className="mt-1 font-semibold text-pos">{workerPool.available_workers}</div></div><div><div className="text-xs text-muted-foreground">执行中</div><div className="mt-1 font-semibold">{workerPool.running}</div></div><div><div className="text-xs text-muted-foreground">排队</div><div className="mt-1 font-semibold text-warn">{workerPool.queued}</div></div><div><div className="text-xs text-muted-foreground">等待处理</div><div className="mt-1 font-semibold">{workerPool.waiting_user}</div></div></div>{workerPool.active_projects.length > 0 && <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground"><span>活跃项目</span>{workerPool.active_projects.map((project) => <Badge key={project} variant="outline">{project.split("/").pop()}</Badge>)}</div>}</Card>}
    {researchPool && <Card className="p-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><div className="font-medium">Research Worker Pool</div><div className="mt-1 text-xs text-muted-foreground">网络与开源信息并行检索；全局限制请求数量</div></div><Badge variant={researchPool.running > 0 ? "info" : "outline"}>{researchPool.running > 0 ? "检索中" : "空闲"}</Badge></div><div className="mt-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-6"><div><div className="text-xs text-muted-foreground">池容量</div><div className="mt-1 font-semibold">{researchPool.max_workers}</div></div><div><div className="text-xs text-muted-foreground">可用槽位</div><div className="mt-1 font-semibold text-pos">{researchPool.available_workers}</div></div><div><div className="text-xs text-muted-foreground">检索中</div><div className="mt-1 font-semibold">{researchPool.running}</div></div><div><div className="text-xs text-muted-foreground">排队</div><div className="mt-1 font-semibold text-warn">{researchPool.queued}</div></div><div><div className="text-xs text-muted-foreground">已完成</div><div className="mt-1 font-semibold">{researchPool.completed}</div></div><div><div className="text-xs text-muted-foreground">失败</div><div className="mt-1 font-semibold text-neg">{researchPool.failed}</div></div></div></Card>}
    {tasks.length === 0 ? <Card className="p-8 text-center text-sm text-muted-foreground">还没有已批准的 AI 任务</Card> : <div className="space-y-3">{tasks.map((task, index) => {
      const view = statusView(task.status ?? "queued")
      const Icon = view.Icon
      const ratedAt = Number(task.result?.feedback?.rated_at ?? 0)
      const elapsedDays = ratedAt > 0 ? (Date.now() / 1000 - ratedAt) / (24 * 60 * 60) : 0
      const completedHorizons = new Set(task.result?.long_term_outcomes?.map((outcome) => outcome.horizon_days) ?? [])
      const dueHorizon = ([7, 30] as const).find((horizon) => elapsedDays >= horizon && !completedHorizons.has(horizon))
      return <Card key={`${task.task_id ?? "task"}-${index}`} className="p-4">
        <div className="flex items-start justify-between gap-4"><div className="min-w-0"><div className="flex items-center gap-2"><Icon className={`size-4 ${task.status === "running" || task.status === "recovering" ? "animate-spin" : ""}`} /><span className="font-medium">{task.task || "项目任务"}</span></div><div className="mt-1 truncate text-xs text-muted-foreground">{task.project_path}</div></div><div className="flex shrink-0 items-center gap-2">{task.durable && <Badge variant="outline">可恢复 · 第 {task.attempt || 0} 次</Badge>}<Badge variant={view.variant}>{view.label}</Badge></div></div>
        {task.error && <div className="mt-3 rounded-md bg-neg/10 px-3 py-2 text-xs text-neg">{task.error}</div>}
        {task.durable && <div className="mt-3 text-xs text-muted-foreground">当前阶段：{task.phase || "queued"}</div>}
        {task.result && <div className="mt-3 space-y-1 text-xs text-muted-foreground"><div>来源：{task.source === "autonomy_auto_execute" ? "自主决策" : task.source === "manual_approval" ? "用户批准" : "项目自动任务"}</div><div>执行者：{task.result.executor || "pi"}</div><div>归因状态：{task.result.attribution_status === "legacy_inferred_pi" ? "历史记录：按 pi 处理，原始能力调用不可验证" : task.result.used_capabilities?.length ? `已调用能力：${task.result.used_capabilities.join(", ")}` : "未直接调用进化能力"}</div>{task.result.capability_trace && task.result.capability_trace.length > 0 && <div className="mt-2 rounded-md border border-border/60 px-3 py-2"><div className="mb-1 text-foreground">能力执行轨迹</div>{task.result.capability_trace.map((call, callIndex) => <div key={`${call.capability}-${call.action}-${call.phase}-${callIndex}`} className="flex items-center justify-between gap-3"><span>{call.phase === "baseline" ? "基线" : "变更后"} · {call.capability}.{call.action}</span><span className={call.success ? "text-emerald-600" : "text-red-600"}>{call.success ? "通过" : "失败"} · {call.elapsed_ms ?? 0}ms</span></div>)}</div>}{task.result.applied ? <div>项目应用：已合并到项目根目录</div> : <div>项目应用：未应用{task.result.apply_error ? ` · ${task.result.apply_error}` : ""}</div>}<div>分支：{task.result.branch || "未生成"}</div><div>Worktree：{task.result.worktree || "未生成"}</div>{task.result.diff_stat && <div>变更：{task.result.diff_stat}</div>}{task.result.verification && <div>验证：{task.result.verification.success ? "通过" : "失败"} · {task.result.verification.command}</div>}{task.result.real_validation && <div>真实任务信号：{task.result.real_validation.success ? "已记录" : "已记录失败"} · {task.result.real_validation.recorded_capabilities?.length ? `能力：${task.result.real_validation.recorded_capabilities.join(", ")}` : "技术验证已记录，未归因给候选能力"}</div>}{task.result.skill_candidates && task.result.skill_candidates.length > 0 && <div>候选技能（未必使用）：{task.result.skill_candidates.join(", ")}</div>}</div>}
        {task.status === "completed" && task.result && !task.result.feedback && <div className="mt-4 flex items-center justify-between gap-3 border-t border-border/60 pt-3"><span className="text-xs text-muted-foreground">这次修改对项目有帮助吗？</span><div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void sendFeedback(task, false)} disabled={feedbackBusy === task.task_id} title="标记为无用"><ThumbsDown className="mr-1.5 size-3.5" />无用</Button><Button size="sm" onClick={() => void sendFeedback(task, true)} disabled={feedbackBusy === task.task_id} title="标记为有用"><ThumbsUp className="mr-1.5 size-3.5" />有用</Button></div></div>}
        {task.result?.feedback && <div className="mt-3 text-xs text-muted-foreground">已反馈：{task.result.feedback.useful ? "有用" : "无用"}{task.result.feedback.note ? ` · ${task.result.feedback.note}` : ""}</div>}
        {task.result?.feedback_effect && <div className="mt-1 rounded-md border border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">已调整 {task.result.feedback_effect.category} 偏好：{(task.result.feedback_effect.before_weight ?? 1).toFixed(3)} → {(task.result.feedback_effect.after_weight ?? 1).toFixed(3)} · 置信度 {Math.round((task.result.feedback_effect.confidence ?? 0) * 100)}% · {task.result.feedback_effect.sample_count ?? 0} 个样本</div>}
        {(task.result?.long_term_outcomes?.length ?? 0) > 0 && <div className="mt-2 flex flex-wrap gap-2">{task.result?.long_term_outcomes?.map((outcome) => <Badge key={outcome.horizon_days} variant={outcome.status === "rolled_back" ? "neg" : "pos"}>{outcome.horizon_days} 天 · {outcome.status === "rolled_back" ? "已回滚" : outcome.status === "still_using" ? "仍在使用" : "已采纳"}</Badge>)}</div>}
        {dueHorizon && <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-md border border-info/30 bg-info/5 px-3 py-2"><span className="text-xs text-muted-foreground">{dueHorizon} 天复核：这项改进现在怎么样？</span><div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void sendOutcome(task, dueHorizon, "rolled_back")} disabled={outcomeBusy === task.task_id}>已回滚</Button><Button size="sm" onClick={() => void sendOutcome(task, dueHorizon, dueHorizon === 7 ? "adopted" : "still_using")} disabled={outcomeBusy === task.task_id}>{dueHorizon === 7 ? "已采纳" : "仍在使用"}</Button></div></div>}
        {task.durable && ["failed", "timeout", "waiting_user"].includes(task.status ?? "") && <div className="mt-3 flex justify-end"><Button size="sm" variant="outline" className="gap-1.5" onClick={() => void retryRun(task)} disabled={retryBusy === task.task_id}>{retryBusy === task.task_id ? <Loader2 className="size-3.5 animate-spin" /> : <RotateCcw className="size-3.5" />}继续执行</Button></div>}
        {task.durable && <div className="mt-3 border-t border-border/60 pt-2"><Button size="sm" variant="ghost" className="gap-1.5 px-1 text-xs" onClick={() => void toggleEvents(task)}>{expandedTask === task.task_id ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}<History className="size-3.5" />运行记录</Button>{expandedTask === task.task_id && <div className="mt-2 space-y-2 border-l border-border pl-3">{(eventsByTask[task.task_id ?? ""] ?? []).map((event) => <div key={event.seq} className="text-xs"><span className="text-foreground">{eventLabel(event)}</span><span className="ml-2 text-muted-foreground">{event.phase} · {new Date(event.created_at * 1000).toLocaleString()}</span></div>)}</div>}</div>}
        <div className="mt-3 font-mono text-[10px] text-muted-foreground/70">{task.task_id}</div>
      </Card>
    })}</div>}
  </div>
}
