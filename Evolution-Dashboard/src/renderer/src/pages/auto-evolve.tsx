import { useCallback, useEffect, useRef, useState } from "react"
import {
  Brain,
  CheckCircle2,
  XCircle,
  Clock,
  Flame,
  GitBranch,
  Lightbulb,
  Loader2,
  RefreshCw,
  Sparkles,
  TrendingUp,
  X,
  Zap,
} from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import { unwrap } from "@/lib/types"
import type { EvolutionData } from "@/lib/types"

function fmtTime(ts: number | string): string {
  if (typeof ts === "string") return ts
  if (!ts || ts === 0) return "—"
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

function fmtElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  percent,
}: {
  icon: typeof Brain
  label: string
  value: string
  detail?: string
  percent?: number
}) {
  return (
    <Card className="p-5 hover-elevate">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div>
      {detail && <div className="mt-1 text-xs text-muted-foreground">{detail}</div>}
      {percent !== undefined && (
        <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "h-full rounded-full transition-all",
              percent >= 50 ? "bg-pos" : percent >= 20 ? "bg-warn" : "bg-neg",
            )}
            style={{ width: `${Math.min(100, percent)}%` }}
          />
        </div>
      )}
    </Card>
  )
}

function successIcon(success: boolean) {
  return success ? (
    <CheckCircle2 className="h-3.5 w-3.5 text-pos shrink-0" />
  ) : (
    <XCircle className="h-3.5 w-3.5 text-neg shrink-0" />
  )
}

export function AutoEvolvePage() {
  const [data, setData] = useState<EvolutionData | null>(null)
  const [projects, setProjects] = useState<Array<{ name: string; path: string; branch: string; head: string; dirty: boolean; changed_files: number; kind: string[]; verify_command?: string | null; health_status?: string; evidence?: string[]; last_checked_at?: number | null; memory?: { vision?: string; priorities?: string[]; completed_count?: number; rejected_count?: number; feedback_count?: number; updated_at?: number | null }; proposals?: Array<{ id: string; title: string; reason: string; task: string; evidence?: string[]; priority: string; status: string; verify_command?: string | null; expected_value?: string; risk?: string; category?: string; impact_scope?: string; leverage_score?: number; confidence?: number; initiative?: { recommended_action?: string; effective_action?: string; rationale?: string } }> }>>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [expandedChain, setExpandedChain] = useState<number | null>(null)
  type ActiveProposal = { project: string; path: string; id: string; autonomy_prompt_id?: string; title: string; reason: string; task: string; evidence?: string[]; verify_command?: string | null; expected_value?: string; risk?: string; category?: string; impact_scope?: string; leverage_score?: number; confidence?: number; value_energy?: number; energy_units?: number; initiative?: { recommended_action?: string; effective_action?: string; rationale?: string } }
  const [activeProposalGroup, setActiveProposalGroup] = useState<{ project: string; path: string; proposals: ActiveProposal[] } | null>(null)
  const [selectedProposalIds, setSelectedProposalIds] = useState<string[]>([])
  const activeGroupPath = useRef("")
  const [proposalDismissed, setProposalDismissed] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem("evolution-dashboard:dismissed-proposals") || "[]") } catch { return [] }
  })
  const [executingProposal, setExecutingProposal] = useState(false)
  const [editingMemory, setEditingMemory] = useState<string | null>(null)
  const [memoryDraft, setMemoryDraft] = useState({ vision: "", priorities: "" })
  const [memorySaving, setMemorySaving] = useState(false)
  const [workspace, setWorkspace] = useState<{ totals?: { projects?: number; changed_files?: number; todos?: number; branches?: number; remotes?: number }; generated_at?: number } | null>(null)
  const [exploreProjectPath, setExploreProjectPath] = useState("")
  const [exploreObjective, setExploreObjective] = useState("")
  const [exploreLoading, setExploreLoading] = useState(false)
  const [explorerResult, setExplorerResult] = useState<{ project_path: string; objective: string; source: string; proposals: Array<{ id: string; title: string; approach: string; task: string; expected_value: string; risk: string; confidence: number }> } | null>(null)
  const [selectedVariants, setSelectedVariants] = useState<string[]>([])
  const [experimentBusy, setExperimentBusy] = useState(false)
  const [experimentBatches, setExperimentBatches] = useState<Array<{ batch_id: string; objective: string; status: string; runs: Array<{ variant_id: string; title: string; status: string; verification?: { success: boolean }; benchmark?: { success: boolean }; cleaned_up: boolean }> }>>([])
  const [autonomy, setAutonomy] = useState<{ enabled: boolean; paused: boolean; last_tick_at?: number; last_error?: string | null; energy_budget?: number; energy_spent?: number; exploitation_budget?: number; exploration_budget?: number; exploitation_spent?: number; exploration_spent?: number; value_allocations?: Array<{ project_name: string; proposal_id: string; title: string; category: string; impact_scope?: string; leverage_score?: number; net_value: number; energy_units: number; selected: boolean; selection_reason?: string; action: string; rationale: string }>; decisions: Array<{ id: string; project_name: string; objective: string; action: string; status: string; confidence: number; value_energy?: number; energy_units?: number; rationale: string; detail?: string; created_at: number }>; prompts: Array<{ id: string; project_path: string; project_name: string; proposal_id: string; title: string; reason: string; task: string; expected_value?: string; risk?: string; verify_command?: string | null; evidence: string[]; confidence: number; category?: string; impact_scope?: string; leverage_score?: number; value_energy?: number; energy_units?: number; rationale: string; status: string; created_at: number }> } | null>(null)
  const [learningAgenda, setLearningAgenda] = useState<{ updated_at: number; projects: Array<{ project_path: string; project_name: string; north_star: string; milestones: string[]; active_goals: Array<{ id: string; title: string; category: string; expected_value: string; status: string }>; knowledge_gaps: Array<{ id: string; question: string; related_goal: string; evidence: string[]; confidence: number; goal_alignment: number; leverage_score: number; information_gain: number; reuse_probability: number; research_cost: number; learning_value: number; status: string }>; learned_patterns: Array<{ id: string; summary: string; source: string; confidence: number; learned_at: number }>; review_at: number }> } | null>(null)
  const [autonomyBusy, setAutonomyBusy] = useState(false)
  const refreshInFlight = useRef(false)

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return
    refreshInFlight.current = true
    try {
      const resp = await window.daemon.evolution()
      if (resp.ok && resp.status === 200) {
        const payload = unwrap<{ evolution: EvolutionData }>(resp).evolution
        setData(payload)
        setError("")
      } else if (resp.status === 503) {
        setError("进化忙（归因中），稍后刷新")
      } else {
        const d = resp.data as Record<string, unknown>
        setError(String(d?.error ?? `HTTP ${resp.status}`))
      }
      const projectResp = await window.daemon.projects()
      if (projectResp.ok && projectResp.status === 200) {
        const projectData = projectResp.data as { projects?: typeof projects }
        setProjects(projectData.projects ?? [])
      }
      const graphResp = await window.daemon.workspaceGraph()
      if (graphResp.ok && graphResp.status === 200) setWorkspace(graphResp.data as typeof workspace)
      const batchesResp = await window.daemon.experimentBatches()
      if (batchesResp.ok && batchesResp.status === 200) setExperimentBatches((batchesResp.data as { batches?: typeof experimentBatches }).batches ?? [])
      const autonomyResp = await window.daemon.autonomyStatus()
      if (autonomyResp.ok && autonomyResp.status === 200) {
        const nextAutonomy = (autonomyResp.data as { autonomy?: typeof autonomy }).autonomy ?? null
        setAutonomy(nextAutonomy)
        const nextPrompt = nextAutonomy?.prompts
          .filter((prompt) => prompt.status === "pending" && !proposalDismissed.includes(`${prompt.project_name}:${prompt.proposal_id}`))
          .sort((left, right) => (right.value_energy ?? 0) - (left.value_energy ?? 0))[0]
        if (nextPrompt) {
          const projectPrompts = nextAutonomy?.prompts
            .filter((prompt) => prompt.status === "pending" && prompt.project_path === nextPrompt.project_path && !proposalDismissed.includes(`${prompt.project_name}:${prompt.proposal_id}`))
            .sort((left, right) => (right.value_energy ?? 0) - (left.value_energy ?? 0)) ?? []
          const proposals = projectPrompts.map((prompt) => ({
            project: prompt.project_name, path: prompt.project_path, id: prompt.proposal_id,
            autonomy_prompt_id: prompt.id, title: prompt.title, reason: prompt.reason, task: prompt.task,
            expected_value: prompt.expected_value, risk: prompt.risk, verify_command: prompt.verify_command,
            evidence: prompt.evidence, confidence: prompt.confidence, category: prompt.category,
            impact_scope: prompt.impact_scope, leverage_score: prompt.leverage_score,
            value_energy: prompt.value_energy, energy_units: prompt.energy_units,
            initiative: { effective_action: "ask_user", rationale: prompt.rationale },
          }))
          if (activeGroupPath.current !== nextPrompt.project_path) {
            activeGroupPath.current = nextPrompt.project_path
            setSelectedProposalIds(proposals.map((proposal) => proposal.autonomy_prompt_id ?? proposal.id))
          }
          setActiveProposalGroup({ project: nextPrompt.project_name, path: nextPrompt.project_path, proposals })
        } else {
          activeGroupPath.current = ""
          setActiveProposalGroup(null)
        }
      }
      const agendaResp = await window.daemon.learningAgenda()
      if (agendaResp.ok && agendaResp.status === 200) {
        setLearningAgenda((agendaResp.data as { agenda?: typeof learningAgenda }).agenda ?? null)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
      refreshInFlight.current = false
    }
  }, [proposalDismissed])

  const hideProposalGroup = (group: NonNullable<typeof activeProposalGroup>) => {
    const next = Array.from(new Set([...proposalDismissed, ...group.proposals.map((proposal) => `${proposal.project}:${proposal.id}`)]))
    setProposalDismissed(next)
    localStorage.setItem("evolution-dashboard:dismissed-proposals", JSON.stringify(next))
    activeGroupPath.current = ""
    setActiveProposalGroup(null)
  }

  const rejectProposalGroup = async (group: NonNullable<typeof activeProposalGroup>) => {
    setExecutingProposal(true)
    try {
      await Promise.all(group.proposals.map((proposal) => proposal.autonomy_prompt_id
        ? window.daemon.rejectAutonomyPrompt(proposal.autonomy_prompt_id)
        : window.daemon.projectProposalFeedback(proposal.id, { project_path: proposal.path, title: proposal.title, task: proposal.task, category: proposal.category, useful: false })))
      hideProposalGroup(group)
    } finally { setExecutingProposal(false) }
  }

  const approveProposalGroup = async (group: NonNullable<typeof activeProposalGroup>) => {
    setExecutingProposal(true)
    try {
      await Promise.all(group.proposals.map((proposal) => {
        const key = proposal.autonomy_prompt_id ?? proposal.id
        if (!selectedProposalIds.includes(key)) {
          return proposal.autonomy_prompt_id ? window.daemon.dismissAutonomyPrompt(proposal.autonomy_prompt_id) : Promise.resolve({ ok: true, status: 200, data: null })
        }
        return proposal.autonomy_prompt_id
          ? window.daemon.approveAutonomyPrompt(proposal.autonomy_prompt_id)
          : window.daemon.executeProject({ project_path: proposal.path, task: proposal.task, proposal_id: proposal.id, verify_command: proposal.verify_command ?? undefined })
      }))
      hideProposalGroup(group)
      await refresh()
    } finally { setExecutingProposal(false) }
  }

  const startMemoryEdit = (project: (typeof projects)[number]) => {
    setEditingMemory(project.path)
    setMemoryDraft({ vision: project.memory?.vision ?? "", priorities: project.memory?.priorities?.join(", ") ?? "" })
  }

  const saveMemory = async (projectPath: string) => {
    setMemorySaving(true)
    try {
      await window.daemon.updateProjectMemory({ project_path: projectPath, vision: memoryDraft.vision, priorities: memoryDraft.priorities.split(",").map((item) => item.trim()).filter(Boolean) })
      setEditingMemory(null)
      await refresh()
    } finally { setMemorySaving(false) }
  }

  const explore = async () => {
    if (!exploreProjectPath || !exploreObjective.trim()) return
    setExploreLoading(true)
    try {
      const response = await window.daemon.exploreProject({ project_path: exploreProjectPath, objective: exploreObjective.trim(), max_variants: 4 })
      if (response.ok && response.status === 200) {
        const result = response.data as typeof explorerResult
        setExplorerResult(result)
        setSelectedVariants(result?.proposals.map((proposal) => proposal.id) ?? [])
      }
    } finally { setExploreLoading(false) }
  }

  const runSelectedExperiments = async () => {
    if (!explorerResult || selectedVariants.length < 2) return
    setExperimentBusy(true)
    try {
      await window.daemon.runExperiments({ project_path: explorerResult.project_path, objective: explorerResult.objective, variants: explorerResult.proposals.filter((proposal) => selectedVariants.includes(proposal.id)).map(({ id, title, task }) => ({ id, title, task })) })
      await refresh()
    } finally { setExperimentBusy(false) }
  }

  const setAutonomyPaused = async (paused: boolean) => {
    setAutonomyBusy(true)
    try {
      const response = paused ? await window.daemon.pauseAutonomy() : await window.daemon.resumeAutonomy()
      if (response.ok && response.status === 200) setAutonomy((response.data as { autonomy?: typeof autonomy }).autonomy ?? null)
    } finally { setAutonomyBusy(false) }
  }

  const resolveAutonomyPrompt = async (id: string, approve: boolean) => {
    setAutonomyBusy(true)
    try {
      if (approve) await window.daemon.approveAutonomyPrompt(id)
      else await window.daemon.rejectAutonomyPrompt(id)
      await refresh()
    } finally { setAutonomyBusy(false) }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 4000)
    return () => clearInterval(id)
  }, [refresh])

  if (loading && !data) {
    return (
      <div className="flex flex-1 items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 py-20">
        <X className="h-8 w-8 text-neg" />
        <p className="text-sm text-muted-foreground">{error}</p>
        <Button size="sm" variant="ghost" onClick={refresh} className="gap-1.5">
          <RefreshCw className="h-3.5 w-3.5" />
          重试
        </Button>
      </div>
    )
  }

  if (!data) return null

  const stats = data.global_stats
  const projectsNeedingDirection = (learningAgenda?.projects ?? []).filter((project) => project.north_star === "待用户确认")
  const topLearningGaps = (learningAgenda?.projects ?? [])
    .flatMap((project) => project.knowledge_gaps.map((gap) => ({ ...gap, project_name: project.project_name, north_star: project.north_star })))
    .filter((gap) => gap.status === "open")
    .sort((left, right) => right.learning_value - left.learning_value)
    .slice(0, 6)

  return (
    <>
      {activeProposalGroup && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6 backdrop-blur-sm">
          <Card className="flex max-h-[88vh] w-full max-w-3xl flex-col border-primary/30 p-6 shadow-2xl">
            <div className="flex items-start justify-between gap-4">
              <div><Badge variant="warn">项目改进清单</Badge><h3 className="mt-3 text-lg font-semibold">{activeProposalGroup.project}</h3><p className="mt-1 text-sm text-muted-foreground">AI 本轮希望完成 {activeProposalGroup.proposals.length} 项改进，默认全部选择</p></div>
              <Button variant="ghost" size="icon" title="稍后处理" onClick={() => hideProposalGroup(activeProposalGroup)}><X className="size-4" /></Button>
            </div>
            <div className="mt-5 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
              {activeProposalGroup.proposals.map((proposal) => {
                const key = proposal.autonomy_prompt_id ?? proposal.id
                const checked = selectedProposalIds.includes(key)
                return <label key={key} className={cn("block cursor-pointer rounded-xl border p-4 transition-colors", checked ? "border-primary/30 bg-primary/5" : "border-border/60 bg-muted/20 opacity-65")}>
                  <div className="flex items-start gap-3">
                    <input type="checkbox" checked={checked} onChange={(event) => setSelectedProposalIds((current) => event.target.checked ? Array.from(new Set([...current, key])) : current.filter((id) => id !== key))} className="mt-1 size-4 accent-primary" />
                    <div className="min-w-0 flex-1 text-sm">
                      <div className="flex flex-wrap items-center gap-2"><span className="font-semibold">{proposal.title}</span>{proposal.category && <Badge variant="outline">{proposal.category}</Badge>}{(proposal.leverage_score ?? 0) > 0 && <Badge variant="info">核心杠杆 {Math.round((proposal.leverage_score ?? 0) * 100)}%</Badge>}<Badge variant="warn">价值 {Math.round((proposal.value_energy ?? 0) * 100)}% · {proposal.energy_units ?? 0} 能量</Badge></div>
                      <p className="mt-2 text-muted-foreground">{proposal.reason}</p>
                      <div className="mt-3 rounded-lg bg-muted/50 p-3"><div className="text-xs text-muted-foreground">执行内容</div><div className="mt-1">{proposal.task}</div></div>
                      {proposal.expected_value && <p className="mt-2 text-xs text-pos">预期收益：{proposal.expected_value}</p>}
                      {proposal.risk && <p className="mt-1 text-xs text-muted-foreground">风险：{proposal.risk}</p>}
                      {proposal.evidence && proposal.evidence.length > 0 && <details className="mt-2 text-xs"><summary className="cursor-pointer text-warn">查看 {proposal.evidence.length} 条检测证据</summary><ul className="mt-1 list-disc space-y-1 pl-5 text-muted-foreground">{proposal.evidence.map((item) => <li key={item}>{item}</li>)}</ul></details>}
                    </div>
                  </div>
                </label>
              })}
            </div>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-4">
              <p className="text-xs text-muted-foreground">取消勾选的项目不会执行，也不会计为负反馈或再次提示。</p>
              <div className="flex gap-2"><Button variant="ghost" onClick={() => void rejectProposalGroup(activeProposalGroup)} disabled={executingProposal}>全部不感兴趣</Button><Button variant="outline" onClick={() => hideProposalGroup(activeProposalGroup)} disabled={executingProposal}>稍后处理</Button><Button onClick={() => void approveProposalGroup(activeProposalGroup)} disabled={executingProposal || selectedProposalIds.length === 0}>{executingProposal ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Sparkles className="mr-2 size-4" />}批准 {selectedProposalIds.length} 项进入执行队列</Button></div>
            </div>
          </Card>
        </div>
      )}
      {/* 标题栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="grid size-9 place-items-center rounded-xl bg-primary/10 text-primary">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold tracking-tight">自主进化</h2>
            <p className="text-xs text-muted-foreground">
              发现可验证的问题 → 提交提案 → 经批准后由 pi 执行 → 验证并学习反馈
            </p>
          </div>
        </div>
        <Button size="sm" variant="ghost" onClick={refresh} className="gap-1.5">
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          刷新
        </Button>
      </div>

      {error && (
        <Badge variant="warn" className="gap-1.5 self-start">
          {error}
        </Badge>
      )}

      {autonomy && <Card className="border-primary/20 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-medium">自主控制回路</h3>
            <p className="mt-1 text-xs text-muted-foreground">持续观察工作区，按证据决定等待、提示、实验或隔离执行</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={autonomy.paused ? "warn" : "pos"}>{autonomy.paused ? "已暂停" : "运行中"}</Badge>
            <Button size="sm" variant="outline" onClick={() => void setAutonomyPaused(!autonomy.paused)} disabled={autonomyBusy}>
              {autonomy.paused ? "恢复自主" : "暂停自主"}
            </Button>
          </div>
        </div>
        <div className="mt-4 grid gap-3 text-sm md:grid-cols-5">
          <div><div className="text-xs text-muted-foreground">待确认提示</div><div className="mt-1 text-xl font-semibold">{autonomy.prompts.filter((prompt) => prompt.status === "pending").length}</div></div>
          <div><div className="text-xs text-muted-foreground">已记录决策</div><div className="mt-1 text-xl font-semibold">{autonomy.decisions.length}</div></div>
          <div><div className="text-xs text-muted-foreground">价值能量</div><div className="mt-1 text-xl font-semibold text-primary">{autonomy.energy_spent ?? 0}<span className="text-sm text-muted-foreground"> / {autonomy.energy_budget ?? 0}</span></div><div className="mt-1 text-[10px] text-muted-foreground">利用 {autonomy.exploitation_spent ?? 0}/{autonomy.exploitation_budget ?? 0} · 探索 {autonomy.exploration_spent ?? 0}/{autonomy.exploration_budget ?? 0}</div></div>
          <div><div className="text-xs text-muted-foreground">最近观察</div><div className="mt-1 text-sm">{autonomy.last_tick_at ? fmtTime(autonomy.last_tick_at) : "尚未运行"}</div></div>
          <div><div className="text-xs text-muted-foreground">状态说明</div><div className="mt-1 truncate text-sm">{autonomy.last_error || "控制器正常工作"}</div></div>
        </div>
        {(autonomy.value_allocations?.length ?? 0) > 0 && <div className="mt-4 rounded-lg border border-border/60 bg-muted/20 p-3">
          <div className="text-xs font-medium">本轮价值分配</div>
          <div className="mt-2 space-y-1.5">{autonomy.value_allocations?.slice(0, 5).map((allocation) => <div key={`${allocation.project_name}:${allocation.proposal_id}`} className="flex items-center gap-2 text-xs">
            <span className={cn("size-1.5 rounded-full", allocation.selected ? "bg-pos" : "bg-muted-foreground/40")} />
            <span className="min-w-0 flex-1 truncate">{allocation.project_name} · {allocation.title}</span>
            <span className="text-muted-foreground">{allocation.category}</span>
            {allocation.selected && <Badge variant={allocation.selection_reason === "explore" ? "info" : "pos"}>{allocation.selection_reason === "explore" ? "探索" : "利用"}</Badge>}
            <span className="text-muted-foreground">杠杆 {Math.round((allocation.leverage_score ?? 0) * 100)}%</span>
            <span className="font-medium text-primary">{Math.round(allocation.net_value * 100)}%</span>
            <span className="text-muted-foreground">{allocation.energy_units} 能量</span>
          </div>)}</div>
        </div>}
        {autonomy.prompts.filter((prompt) => prompt.status === "pending").slice().reverse().slice(0, 3).map((prompt) => <div key={prompt.id} className="mt-3 rounded-lg border border-warn/30 bg-warn/5 p-3 text-xs">
          <div className="flex items-start justify-between gap-3"><div className="min-w-0"><div className="font-medium">{prompt.project_name} · {prompt.title}</div><div className="mt-1 text-muted-foreground">{prompt.reason}</div></div><Badge variant="warn">价值 {Math.round((prompt.value_energy ?? 0) * 100)}% · {prompt.energy_units ?? 0} 能量</Badge></div>
          <div className="mt-2 text-foreground/80">{prompt.task}</div>
          {prompt.evidence.length > 0 && <div className="mt-2 text-muted-foreground">证据：{prompt.evidence.join("；")}</div>}
          <div className="mt-3 flex justify-end gap-2"><Button size="sm" variant="ghost" onClick={() => void resolveAutonomyPrompt(prompt.id, false)} disabled={autonomyBusy}>拒绝</Button><Button size="sm" onClick={() => void resolveAutonomyPrompt(prompt.id, true)} disabled={autonomyBusy}>批准隔离执行</Button></div>
        </div>)}
        {autonomy.decisions.length > 0 && <div className="mt-4 space-y-1 border-t border-border/50 pt-3">{autonomy.decisions.slice().reverse().slice(0, 4).map((decision) => <div key={decision.id} className="flex items-center justify-between gap-3 text-xs"><span className="truncate"><span className="font-medium">{decision.project_name}</span> · {decision.objective}</span><span className="shrink-0 text-muted-foreground">{decision.action} · {decision.status}</span></div>)}</div>}
      </Card>}

      {/* 进化指标卡 */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          icon={GitBranch}
          label="变异总数"
          value={String(stats.total_mutations)}
          detail={`成功 ${stats.total_mutation_successes} 次`}
          percent={stats.total_mutations > 0 ? (stats.mutation_success_rate * 100) : 0}
        />
        <MetricCard
          icon={Sparkles}
          label="自主目标"
          value={String(stats.total_autonomous_goals)}
          detail={`成功 ${stats.total_autonomous_successes} 次`}
          percent={stats.total_autonomous_goals > 0 ? (stats.autonomous_success_rate * 100) : 0}
        />
        <MetricCard
          icon={TrendingUp}
          label="创造 / 淘汰"
          value={`${stats.total_created} / ${stats.total_eliminated}`}
          detail={`累计进化 ${stats.total_rounds} 轮`}
        />
        <MetricCard
          icon={Brain}
          label="LLM 自主判定"
          value={String(
            data.top_capabilities.reduce((s, c) => s + c.human_signals_count, 0),
          )}
          detail="替代人类反馈的价值信号"
        />
      </div>

      {learningAgenda && <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h3 className="font-medium">自主学习议程</h3><p className="mt-1 text-xs text-muted-foreground">围绕项目目标维护关键未知；学习价值决定先研究什么，不直接触发代码修改</p></div>
          <div className="flex items-center gap-2"><Badge variant="outline">{learningAgenda.projects.length} 个项目</Badge>{projectsNeedingDirection.length > 0 && <Badge variant="warn">{projectsNeedingDirection.length} 个待确认方向</Badge>}<Badge variant="info">{topLearningGaps.length} 个优先知识缺口</Badge></div>
        </div>
        {projectsNeedingDirection.length > 0 && <div className="mt-4 flex flex-wrap items-center gap-2 rounded-lg border border-warn/30 bg-warn/5 p-3 text-xs"><span className="font-medium text-warn">需要确认长期目标</span>{projectsNeedingDirection.map((project) => <Badge key={project.project_path} variant="outline">{project.project_name}</Badge>)}</div>}
        {topLearningGaps.length === 0 ? <p className="mt-4 text-sm text-muted-foreground">暂未发现会改变决策方向的关键未知。</p> : <div className="mt-4 grid gap-2 md:grid-cols-2">
          {topLearningGaps.map((gap) => <div key={`${gap.project_name}:${gap.id}`} className="rounded-lg border border-border/60 bg-muted/20 p-3 text-xs">
            <div className="flex items-start justify-between gap-3"><div className="min-w-0"><div className="font-medium">{gap.project_name}</div><div className="mt-1 text-foreground/90">{gap.question}</div></div><Badge variant={gap.status === "needs_user" ? "warn" : "info"}>学习价值 {Math.round(gap.learning_value * 100)}%</Badge></div>
            <div className="mt-2 text-muted-foreground">关联目标：{gap.related_goal}</div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground"><span>目标 {Math.round(gap.goal_alignment * 100)}%</span><span>核心杠杆 {Math.round((gap.leverage_score ?? 0) * 100)}%</span><span>信息增益 {Math.round(gap.information_gain * 100)}%</span><span>复用 {Math.round(gap.reuse_probability * 100)}%</span><span>成本 {Math.round(gap.research_cost * 100)}%</span></div>
          </div>)}
        </div>}
      </Card>}

      <Card className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">主动发现的项目</h3>
            <p className="mt-1 text-xs text-muted-foreground">只读扫描项目状态；只有发现具体证据才会主动提示</p>
          </div>
          <Badge variant="outline">{projects.length} 个项目</Badge>
        </div>
        <div className="mt-4 space-y-2">
          {projects.length === 0 ? <p className="text-sm text-muted-foreground">暂未发现 Git 项目。设置 ORCH_PROJECT_ROOTS 可指定扫描目录。</p> : projects.map((project) => (
            <div key={project.path}>
              <div className="flex items-center justify-between rounded-lg border border-border/60 px-3 py-2 text-sm">
                <div className="min-w-0"><div className="font-medium">{project.name}</div><div className="truncate text-xs text-muted-foreground">{project.path} · {project.branch} · {project.head}</div></div>
                <div className="flex shrink-0 items-center gap-2"><Badge variant={project.dirty ? "warn" : "outline"}>{project.dirty ? `${project.changed_files} 处变更` : "干净"}</Badge><Badge variant={project.health_status === "failing" ? "neg" : project.health_status === "passing" ? "pos" : project.health_status === "checking" ? "info" : project.health_status === "observed" ? "info" : "outline"}>{project.health_status === "failing" ? "验证失败" : project.health_status === "passing" ? "验证通过" : project.health_status === "checking" ? "检查中" : project.health_status === "observed" ? "已完成只读分析" : "待检查"}</Badge></div>
              </div>
              <div className="ml-3 mt-2 rounded-lg border border-border/50 bg-muted/20 px-3 py-2 text-xs"><div className="flex items-center justify-between gap-3"><div className="min-w-0"><div className="font-medium">项目记忆</div><div className="mt-1 text-muted-foreground">{project.memory?.vision || "尚未设置项目愿景"}</div><div className="mt-1 text-muted-foreground">优先级：{project.memory?.priorities?.join("、") || "未设置"} · 已完成 {project.memory?.completed_count ?? 0} · 已拒绝 {project.memory?.rejected_count ?? 0} · 反馈 {project.memory?.feedback_count ?? 0}</div></div><Button size="sm" variant="outline" onClick={() => startMemoryEdit(project)}>编辑</Button></div>{editingMemory === project.path && <div className="mt-3 space-y-2"><Textarea value={memoryDraft.vision} onChange={(event) => setMemoryDraft((draft) => ({ ...draft, vision: event.target.value }))} placeholder="这个项目长期要为用户带来什么？" className="min-h-16 text-xs" /><input value={memoryDraft.priorities} onChange={(event) => setMemoryDraft((draft) => ({ ...draft, priorities: event.target.value }))} placeholder="优先级，用逗号分隔" className="flex h-8 w-full rounded-md border border-input bg-transparent px-2 text-xs" /><div className="flex justify-end gap-2"><Button size="sm" variant="ghost" onClick={() => setEditingMemory(null)}>取消</Button><Button size="sm" onClick={() => void saveMemory(project.path)} disabled={memorySaving}>{memorySaving ? "保存中" : "保存项目方向"}</Button></div></div>}</div>
              {project.proposals?.filter((proposal) => proposal.status === "proposed").map((proposal) => <div key={proposal.id} className="ml-3 rounded-lg bg-muted/40 px-3 py-2 text-xs"><div className="font-medium">主动提案：{proposal.title}{proposal.category && <span className="ml-2 text-muted-foreground">{proposal.category}</span>}{(proposal.leverage_score ?? 0) > 0 && <span className="ml-2 text-primary">核心杠杆 {Math.round((proposal.leverage_score ?? 0) * 100)}%</span>}</div><div className="mt-1 text-muted-foreground">{proposal.reason}</div>{proposal.initiative && <div className="mt-1 text-primary">策略：{proposal.initiative.effective_action || "ask_user"} · 置信度 {Math.round((proposal.confidence ?? 0) * 100)}%</div>}{proposal.expected_value && <div className="mt-1 text-pos">收益：{proposal.expected_value}</div>}{proposal.risk && <div className="mt-1 text-muted-foreground">风险：{proposal.risk}</div>}{proposal.evidence && proposal.evidence.length > 0 && <div className="mt-1 text-warn">证据：{proposal.evidence.join("；")}</div>}<div className="mt-1 text-foreground/80">{proposal.task}</div></div>)}
            </div>
          ))}
        </div>
      </Card>

      {workspace && <Card className="p-5"><div className="flex items-center justify-between"><div><h3 className="font-medium">Workspace Graph</h3><p className="mt-1 text-xs text-muted-foreground">持续观察项目、分支、变更和待办信号</p></div><Badge variant="outline">{workspace.totals?.projects ?? 0} 个项目</Badge></div><div className="mt-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-5"><div><div className="text-xs text-muted-foreground">变更文件</div><div className="mt-1 font-semibold">{workspace.totals?.changed_files ?? 0}</div></div><div><div className="text-xs text-muted-foreground">TODO/风险标记</div><div className="mt-1 font-semibold">{workspace.totals?.todos ?? 0}</div></div><div><div className="text-xs text-muted-foreground">分支</div><div className="mt-1 font-semibold">{workspace.totals?.branches ?? 0}</div></div><div><div className="text-xs text-muted-foreground">远程仓库</div><div className="mt-1 font-semibold">{workspace.totals?.remotes ?? 0}</div></div><div><div className="text-xs text-muted-foreground">状态</div><div className="mt-1 font-semibold text-pos">观察中</div></div></div></Card>}

      <Card className="p-5"><div className="flex items-center justify-between"><div><h3 className="font-medium">多方案实验台</h3><p className="mt-1 text-xs text-muted-foreground">先生成独立方案，再在隔离 worktree 中比较；不会自动合并</p></div><Badge variant="outline">{experimentBatches.length} 个实验批次</Badge></div><div className="mt-4 grid gap-2 md:grid-cols-[1fr_2fr_auto]"><select value={exploreProjectPath} onChange={(event) => setExploreProjectPath(event.target.value)} className="h-9 rounded-md border border-input bg-transparent px-2 text-xs"><option value="">选择项目</option>{projects.map((project) => <option key={project.path} value={project.path}>{project.name}</option>)}</select><input value={exploreObjective} onChange={(event) => setExploreObjective(event.target.value)} placeholder="例如：降低 API 延迟并保持行为兼容" className="h-9 rounded-md border border-input bg-transparent px-3 text-xs" /><Button size="sm" onClick={() => void explore()} disabled={exploreLoading || !exploreProjectPath || !exploreObjective.trim()}>{exploreLoading ? "探索中" : "生成方案"}</Button></div>{explorerResult && <div className="mt-4 space-y-2">{explorerResult.proposals.map((proposal) => <label key={proposal.id} className="flex cursor-pointer items-start gap-3 rounded-lg border border-border/60 bg-muted/20 p-3 text-xs"><input type="checkbox" checked={selectedVariants.includes(proposal.id)} onChange={(event) => setSelectedVariants((current) => event.target.checked ? [...current, proposal.id] : current.filter((id) => id !== proposal.id))} className="mt-0.5" /><span className="min-w-0"><span className="font-medium">{proposal.title}</span><span className="ml-2 text-primary">置信度 {Math.round(proposal.confidence * 100)}%</span><span className="mt-1 block text-muted-foreground">{proposal.approach}</span><span className="mt-1 block">{proposal.task}</span></span></label>)}<div className="flex justify-end"><Button size="sm" onClick={() => void runSelectedExperiments()} disabled={experimentBusy || selectedVariants.length < 2}>{experimentBusy ? "已排队" : `启动 ${selectedVariants.length} 个隔离实验`}</Button></div></div>}{experimentBatches.length > 0 && <div className="mt-4 space-y-2">{experimentBatches.slice().reverse().slice(0, 3).map((batch) => <div key={batch.batch_id} className="rounded-lg border border-border/60 p-3 text-xs"><div className="flex items-center justify-between"><span className="font-medium">{batch.objective}</span><Badge variant={batch.status === "completed" ? "pos" : "warn"}>{batch.status}</Badge></div><div className="mt-2 grid gap-1 md:grid-cols-2">{batch.runs.map((run) => <div key={run.variant_id} className="text-muted-foreground">{run.title}：{run.status} · 验证 {run.verification ? (run.verification.success ? "通过" : "失败") : "未执行"} · 清理 {run.cleaned_up ? "完成" : "待处理"}</div>)}</div></div>)}</div>}</Card>

      {/* 能力概览指标卡 — 合并自原进化概览页 */}
      {(() => {
        const caps = data.top_capabilities
        const avgScore = caps.length > 0 ? caps.reduce((s, c) => s + c.score, 0) / caps.length : 0
        const totalCalls = caps.reduce((s, c) => s + c.call_count, 0)
        const dormant = caps.filter((c) => c.rounds_dormant > 0).length
        return (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricCard
              icon={TrendingUp}
              label="平均适应度"
              value={`${(avgScore * 100).toFixed(0)}%`}
              detail={`${caps.length} 个能力均值`}
              percent={avgScore * 100}
            />
            <MetricCard
              icon={Zap}
              label="总调用次数"
              value={fmtTokens(totalCalls)}
              detail={`${dormant} 个休眠中`}
            />
            <MetricCard
              icon={Sparkles}
              label="进化轮次"
              value={String(stats.total_rounds)}
              detail={stats.rounds_since_last_creation > 0 ? `${stats.rounds_since_last_creation} 轮未创造` : "近期有创造"}
            />
            <MetricCard
              icon={Clock}
              label="运行时间"
              value={data.uptime_secs < 60 ? `${data.uptime_secs}s` : data.uptime_secs < 3600 ? `${Math.floor(data.uptime_secs / 60)}m` : `${Math.floor(data.uptime_secs / 3600)}h ${Math.floor((data.uptime_secs % 3600) / 60)}m`}
              detail={`累计进化 ${data.total_evolutions} 次`}
            />
          </div>
        )
      })()}

      {/* 两栏：自主目标流 + 变异时间线 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* 自主目标流 */}
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
            <span className="text-sm font-medium">自主目标流</span>
            <span className="text-xs text-muted-foreground">
              {data.autonomous_history.length} 条最近记录
            </span>
          </div>
          <div className="max-h-[420px] overflow-y-auto">
            {data.autonomous_history.length === 0 ? (
              <div className="px-5 py-8 text-center text-sm text-muted-foreground">
                尚无自主目标记录
              </div>
            ) : (
              data.autonomous_history.map((entry, i) => (
                <div
                  key={i}
                  className="border-t border-border/40 px-5 py-3 hover:bg-muted/30"
                >
                  <div className="flex items-start gap-2">
                    {successIcon(entry.success)}
                    <div className="min-w-0 flex-1">
                      <p className="line-clamp-2 text-sm font-medium">{entry.goal}</p>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        {entry.capabilities_used.length > 0 && (
                          <Badge variant="secondary" className="text-[10px]">
                            {entry.capabilities_used.join(", ")}
                          </Badge>
                        )}
                        <span className="flex items-center gap-0.5">
                          <Clock className="h-3 w-3" />
                          {fmtElapsed(entry.elapsed_ms)}
                        </span>
                        <span>{fmtTime(entry.timestamp)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* 变异时间线 */}
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
            <span className="text-sm font-medium">变异时间线</span>
            <span className="text-xs text-muted-foreground">
              成功率 {((stats.mutation_success_rate || 0) * 100).toFixed(0)}%
            </span>
          </div>
          <div className="max-h-[420px] overflow-y-auto">
            {data.tried_mutations.length === 0 ? (
              <div className="px-5 py-8 text-center text-sm text-muted-foreground">
                尚无变异记录
              </div>
            ) : (
              data.tried_mutations.map((m, i) => (
                <div
                  key={i}
                  className="border-t border-border/40 px-5 py-3 hover:bg-muted/30"
                >
                  <div className="flex items-start gap-2">
                    {successIcon(m.success)}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-medium text-info">
                          {m.capability}
                        </span>
                        <Badge variant="outline" className="text-[10px]">
                          {m.mutation_type}
                        </Badge>
                      </div>
                      <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
                        {m.description}
                      </p>
                      <span className="mt-0.5 block text-[10px] text-muted-foreground/70">
                        {m.tried_at}
                      </span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* 教训库 + 思维链 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* 教训库 */}
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
            <span className="flex items-center gap-1.5 text-sm font-medium">
              <Lightbulb className="h-4 w-4 text-warn" />
              进化教训
            </span>
            <span className="text-xs text-muted-foreground">
              {data.lessons.length} 条
            </span>
          </div>
          <div className="max-h-[360px] overflow-y-auto">
            {data.lessons.length === 0 ? (
              <div className="px-5 py-8 text-center text-sm text-muted-foreground">
                尚无教训记录
              </div>
            ) : (
              data.lessons.map((l, i) => (
                <div
                  key={i}
                  className="border-t border-border/40 px-5 py-3 hover:bg-muted/30"
                >
                  <div className="flex items-start gap-2">
                    <Badge
                      variant={l.failure_type.includes("success") ? "pos" : "neg"}
                      className="shrink-0 text-[10px]"
                    >
                      {l.failure_type}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <p className="line-clamp-3 text-xs">{l.lesson}</p>
                      <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground/70">
                        <span className="font-mono text-info">{l.capability}</span>
                        <span>引用 {l.referenced_count} 次</span>
                        <span>{l.learned_at}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* LLM 思维链 */}
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
            <span className="flex items-center gap-1.5 text-sm font-medium">
              <Brain className="h-4 w-4 text-primary" />
              LLM 思维链
            </span>
            <span className="text-xs text-muted-foreground">
              {data.thought_chains.length} 条
            </span>
          </div>
          <div className="max-h-[360px] overflow-y-auto">
            {data.thought_chains.length === 0 ? (
              <div className="px-5 py-8 text-center text-sm text-muted-foreground">
                尚无思维链记录
              </div>
            ) : (
              data.thought_chains.map((c, i) => (
                <div
                  key={i}
                  className="border-t border-border/40 px-5 py-3 hover:bg-muted/30"
                >
                  <div className="flex items-start gap-2">
                    {successIcon(c.success)}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-[10px]">
                          {c.chain_type}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground/70">
                          {fmtTime(c.timestamp)}
                        </span>
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs font-medium">
                        {c.conclusion || "(无结论)"}
                      </p>
                      {expandedChain === i && (
                        <p className="mt-1 whitespace-pre-wrap text-[11px] text-muted-foreground">
                          {c.reasoning}
                        </p>
                      )}
                      <button
                        type="button"
                        className="mt-1 text-[10px] text-primary hover:underline"
                        onClick={() => setExpandedChain(expandedChain === i ? null : i)}
                      >
                        {expandedChain === i ? "收起推理" : "展开推理"}
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* 能力排行 */}
      <Card className="overflow-hidden">
        <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
          <span className="text-sm font-medium">能力适应度排行</span>
          <span className="text-xs text-muted-foreground">
            human_signals_count = LLM 自主判定次数
          </span>
        </div>
        <div className="max-h-[300px] overflow-y-auto">
          <table className="w-full text-sm">
            <tbody>
              {data.top_capabilities.slice(0, 20).map((c) => (
                <tr key={c.name} className="border-t border-border hover:bg-muted/30">
                  <td className="px-5 py-2.5">
                    <div className="font-mono text-xs font-medium text-info">{c.name}</div>
                    <div className="line-clamp-1 text-xs text-muted-foreground">
                      {c.description}
                    </div>
                  </td>
                  <td className="px-5 py-2.5 w-20 text-right tabular-nums">
                    {(c.score * 100).toFixed(0)}%
                  </td>
                  <td className="px-5 py-2.5 w-20 text-right tabular-nums text-muted-foreground">
                    {(c.success_rate * 100).toFixed(0)}%
                  </td>
                  <td className="px-5 py-2.5 w-24 text-right">
                    {c.human_signals_count > 0 ? (
                      <Badge variant="secondary" className="gap-1 text-[10px]">
                        <Brain className="h-3 w-3" />
                        {c.human_signals_count} 次
                      </Badge>
                    ) : (
                      <span className="text-xs text-muted-foreground">未判定</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  )
}
