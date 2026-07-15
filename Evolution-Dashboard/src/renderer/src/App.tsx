import { useEffect, useState, useCallback, useRef } from "react"
import {
  Activity,
  BookOpen,
  Dna,
  GitFork,
  PanelLeft,
  RefreshCw,
  Settings,
  Sparkles,
  Zap,
  ListTodo,
} from "lucide-react"
import { cn, createDebouncedStorageWriter } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { Capability, DaemonConfig, DaemonStatus, LlmHealth } from "@/lib/types"
import { unwrap } from "@/lib/types"
import { CapabilitiesPage } from "@/pages/capabilities"
import { SettingsPage } from "@/pages/settings"
import { AutoEvolvePage } from "@/pages/auto-evolve"
import { TasksPage } from "@/pages/tasks"

declare global {
  interface Window {
    daemon: {
      status: () => Promise<{ ok: boolean; status: number; data: unknown }>
      capabilities: () => Promise<{ ok: boolean; status: number; data: unknown }>
      config: () => Promise<{ ok: boolean; status: number; data: unknown }>
      integrationsStatus: () => Promise<{ ok: boolean; status: number; data: unknown }>
      bootstrapIntegrations: () => Promise<{ ok: boolean; status: number; data: unknown }>
      updateConfig: (b: Record<string, unknown>) => Promise<{ ok: boolean; status: number; data: unknown }>
      resetConfig: () => Promise<{ ok: boolean; status: number; data: unknown }>
      testLlm: (b: Record<string, unknown>) => Promise<{ ok: boolean; status: number; data: unknown }>
      exec: (b: { capability: string; action: string; input?: unknown }) => Promise<{ ok: boolean; status: number; data: unknown }>
      evolution: () => Promise<{ ok: boolean; status: number; data: unknown }>
      llmHealth: () => Promise<{ ok: boolean; status: number; data: unknown }>
      research: (b: { urls?: string[]; query?: string; max_sources?: number; force_refresh?: boolean }) => Promise<{ ok: boolean; status: number; data: unknown }>
      workspaceGraph: () => Promise<{ ok: boolean; status: number; data: unknown }>
      autonomyStatus: () => Promise<{ ok: boolean; status: number; data: unknown }>
      autonomyDecisions: () => Promise<{ ok: boolean; status: number; data: unknown }>
      autonomyPrompts: () => Promise<{ ok: boolean; status: number; data: unknown }>
      learningAgenda: () => Promise<{ ok: boolean; status: number; data: unknown }>
      approveAutonomyPrompt: (id: string) => Promise<{ ok: boolean; status: number; data: unknown }>
      rejectAutonomyPrompt: (id: string) => Promise<{ ok: boolean; status: number; data: unknown }>
      dismissAutonomyPrompt: (id: string) => Promise<{ ok: boolean; status: number; data: unknown }>
      pauseAutonomy: () => Promise<{ ok: boolean; status: number; data: unknown }>
      resumeAutonomy: () => Promise<{ ok: boolean; status: number; data: unknown }>
      exploreProject: (b: { project_path: string; objective: string; max_variants?: number }) => Promise<{ ok: boolean; status: number; data: unknown }>
      experimentBatches: () => Promise<{ ok: boolean; status: number; data: unknown }>
      runExperiments: (b: { project_path: string; objective: string; variants: Array<{ id: string; title: string; task: string }>; verify_command?: string; benchmark_command?: string }) => Promise<{ ok: boolean; status: number; data: unknown }>
      port: () => Promise<number>
      projects: () => Promise<{ ok: boolean; status: number; data: unknown }>
      updateProjectMemory: (b: { project_path: string; vision: string; priorities: string[] }) => Promise<{ ok: boolean; status: number; data: unknown }>
      executeProject: (b: { project_path: string; task: string; proposal_id?: string; verify_command?: string }) => Promise<{ ok: boolean; status: number; data: unknown }>
      projectProposalFeedback: (id: string, b: { project_path: string; title: string; task: string; category?: string; useful: boolean }) => Promise<{ ok: boolean; status: number; data: unknown }>
      projectTasks: () => Promise<{ ok: boolean; status: number; data: unknown }>
      projectRuns: () => Promise<{ ok: boolean; status: number; data: unknown }>
      workerPoolStatus: () => Promise<{ ok: boolean; status: number; data: unknown }>
      projectRunEvents: (id: string) => Promise<{ ok: boolean; status: number; data: unknown }>
      retryProjectRun: (id: string) => Promise<{ ok: boolean; status: number; data: unknown }>
      projectTaskFeedback: (id: string, b: { useful: boolean; note?: string }) => Promise<{ ok: boolean; status: number; data: unknown }>
      projectTaskOutcome: (id: string, b: { horizon_days: 7 | 30; status: "adopted" | "still_using" | "rolled_back"; note?: string }) => Promise<{ ok: boolean; status: number; data: unknown }>
      syncPiConfig: (b: { apiKey: string; baseUrl: string; model: string; models: string[] }) => Promise<{ ok: boolean; message: string }>
      loadPresets: () => Promise<unknown[]>
      savePresets: (presets: unknown[]) => Promise<boolean>
    }
  }
}

type NavKey = "capabilities" | "auto-evolve" | "tasks" | "settings"

type CliConnection = {
  installed: boolean
  authenticated: boolean
  healthy: boolean
  version?: string | null
  account?: string | null
  workspace?: string | null
}

type IntegrationSummary = {
  notion: CliConnection
  github: CliConnection
}

const LAST_ACTIVE_NAV_KEY = "evo-dashboard:last-active-nav"
const SIDEBAR_COLLAPSED_KEY = "evo-dashboard:sidebar-collapsed"
const lastActiveNavWriter = createDebouncedStorageWriter(LAST_ACTIVE_NAV_KEY)
const sidebarCollapsedWriter = createDebouncedStorageWriter(SIDEBAR_COLLAPSED_KEY)

const navItems: Array<{ key: NavKey; icon: typeof Dna; label: string }> = [
  { key: "auto-evolve", icon: Sparkles, label: "自主进化" },
  { key: "tasks", icon: ListTodo, label: "自动任务" },
  { key: "capabilities", icon: Dna, label: "能力列表" },
]

function isNavKey(value: string): value is NavKey {
  return value === "settings" || navItems.some((item) => item.key === value)
}

function readLastActiveNav(): NavKey {
  try {
    const stored = window.localStorage.getItem(LAST_ACTIVE_NAV_KEY)
    return stored && isNavKey(stored) ? stored : "auto-evolve"
  } catch {
    return "auto-evolve"
  }
}

function readSidebarCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true"
  } catch {
    return false
  }
}

function sidebarItemClass(active: boolean) {
  return cn(
    "group flex w-full items-center overflow-hidden rounded-3xl text-left transition-all duration-300 ease-out h-11 gap-3 px-3.5",
    active
      ? "bg-background font-medium text-foreground shadow-sm ring-1 ring-border/60 dark:bg-white/[0.08] dark:shadow-none dark:ring-white/10"
      : "text-muted-foreground hover:bg-muted/60 hover:text-foreground dark:hover:bg-white/[0.05]",
  )
}

function fmtUptime(secs: number): string {
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  if (m < 60) return `${m}m ${secs % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

export default function App() {
  const [activeNav, setActiveNav] = useState<NavKey>(() => readLastActiveNav())
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => readSidebarCollapsed())
  // 进入设置前的导航项快照；用于"返回"时恢复到点击设置时所在的页面。
  const prevNavRef = useRef<NavKey>("auto-evolve")

  const [status, setStatus] = useState<DaemonStatus | null>(null)
  const [caps, setCaps] = useState<Capability[]>([])
  const [config, setConfig] = useState<DaemonConfig | null>(null)
  const [error, setError] = useState<string>("")
  const [llmHealth, setLlmHealth] = useState<LlmHealth | null>(null)
  const [integrations, setIntegrations] = useState<IntegrationSummary | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const refreshInFlight = useRef(false)

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return
    refreshInFlight.current = true
    setError("")
    try {
      const [s, c, cfg, h] = await Promise.all([
        window.daemon.status(),
        window.daemon.capabilities(),
        window.daemon.config(),
        window.daemon.llmHealth(),
      ])
      if (s.status === 503 || c.status === 503) {
        setError("进化忙（归因中），读取受限")
      }
      if (s.ok && s.status === 200) setStatus(unwrap<{ status: DaemonStatus }>(s).status)
      if (c.ok && c.status === 200) setCaps(unwrap<{ capabilities: Capability[] }>(c).capabilities)
      if (cfg.ok && cfg.status === 200) setConfig(unwrap<{ config: DaemonConfig }>(cfg).config)
      if (h.ok && h.status === 200) {
        const hd = h.data as Record<string, unknown> | undefined
        if (hd && hd.success !== false) {
          setLlmHealth({
            state: (hd.state as LlmHealth["state"]) ?? "unknown",
            in_active_hours: hd.in_active_hours as boolean,
            active_hours: (hd.active_hours as string | null) ?? null,
            consecutive_failures: (hd.consecutive_failures as number) ?? 0,
            opened_at: (hd.opened_at as string | null) ?? null,
          })
        }
      }
      setLastUpdated(new Date())
    } catch (e) {
      setError(String(e))
    } finally {
      refreshInFlight.current = false
    }
  }, [])

  const refreshIntegrations = useCallback(async () => {
    try {
      const response = await window.daemon.integrationsStatus()
      if (response.ok && response.status === 200) {
        const data = response.data as { integrations?: IntegrationSummary }
        setIntegrations(data.integrations ?? null)
      }
    } catch {
      setIntegrations(null)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 4000)
    return () => clearInterval(id)
  }, [refresh])

  useEffect(() => {
    void refreshIntegrations()
    const id = setInterval(refreshIntegrations, 60_000)
    return () => clearInterval(id)
  }, [refreshIntegrations])

  useEffect(() => {
    lastActiveNavWriter.schedule(activeNav)
  }, [activeNav])

  useEffect(() => {
    sidebarCollapsedWriter.schedule(sidebarCollapsed ? "true" : "false")
  }, [sidebarCollapsed])

  const connected = !!status
  const isSettings = activeNav === "settings"

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        {!isSettings && (
          <div
            className={cn(
              "pointer-events-none absolute top-3 z-30 flex items-center gap-3",
              sidebarCollapsed ? "left-3" : "left-3 lg:left-[118px]",
            )}
          >
            <Button
              variant="ghost"
              size="icon"
              className="pointer-events-auto size-11 rounded-2xl text-muted-foreground transition-all duration-300 hover:bg-background/45 hover:text-foreground"
              type="button"
              title={sidebarCollapsed ? "展开" : "折叠"}
              onClick={() => setSidebarCollapsed((c) => !c)}
            >
              <PanelLeft className={cn("size-[22px] transition-transform duration-300", sidebarCollapsed && "rotate-180")} />
            </Button>
          </div>
        )}

        {!isSettings && (
          <aside
            className={cn(
              "hidden h-full min-h-0 shrink-0 flex-col border-r border-border/60 bg-sidebar/95 pb-2 pt-16 transition-[width,padding,opacity] duration-300 ease-out dark:border-white/10 lg:flex",
              sidebarCollapsed ? "w-0 overflow-hidden px-0 opacity-0" : "w-[288px] px-4 opacity-100",
            )}
          >
            <div className="mt-5 flex shrink-0 items-center gap-3 px-3.5">
              <div className="grid size-10 place-items-center rounded-2xl bg-primary/10 text-primary">
                <Dna className="size-5" />
              </div>
              <div className="min-w-0">
                <p className="truncate text-lg font-semibold tracking-tight">Evolution</p>
                <p className="truncate text-xs text-muted-foreground">人机协同进化</p>
              </div>
            </div>

            <nav className="mt-8 flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto overflow-x-hidden px-1 [scrollbar-gutter:stable]">
              <div className="px-3.5 pb-2 text-[12px] font-medium text-muted-foreground/65">导航</div>
              {navItems.map((item) => {
                const active = item.key === activeNav
                const Icon = item.icon
                return (
                  <button key={item.key} type="button" onClick={() => setActiveNav(item.key)} className={sidebarItemClass(active)}>
                    <Icon className={cn("size-[18px] shrink-0", active ? "text-foreground/80" : "opacity-70")} />
                    <span className="whitespace-nowrap text-[15px]">{item.label}</span>
                  </button>
                )
              })}
            </nav>

            <div className="shrink-0 pb-2">
              <Button
                type="button"
                variant="ghost"
                className={cn(
                  "flex h-10 w-full items-center gap-3 rounded-2xl px-3.5 text-left shadow-none transition-all duration-200",
                  isSettings
                    ? "bg-background text-foreground ring-1 ring-border/60 dark:bg-white/[0.08] dark:ring-white/10"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground dark:hover:bg-white/[0.05]",
                )}
                onClick={() => {
                  if (!isSettings) prevNavRef.current = activeNav
                  setActiveNav("settings")
                }}
              >
                <Settings className="size-[18px] shrink-0 opacity-85" />
                <span className="whitespace-nowrap text-sm font-medium">设置</span>
              </Button>
            </div>
          </aside>
        )}

        <main className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-tl-[28px] border-l border-t border-border/60 bg-background shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] dark:border-white/10 dark:shadow-none">
          {!isSettings && (
            <header
              className={cn(
                "flex h-14 shrink-0 items-center gap-4 overflow-x-auto bg-background pr-5 lg:pr-7",
                sidebarCollapsed ? "pl-16" : "pl-16 lg:pl-7",
              )}
            >
              <div className="ml-auto flex shrink-0 items-center gap-2">
                {error ? (
                  <Badge variant="warn" className="gap-1.5">
                    <Activity className="size-3" />
                    {error}
                  </Badge>
                ) : !connected ? (
                  <Badge variant="neg" className="gap-1.5">
                    <Activity className="size-3" />
                    daemon 未连接
                  </Badge>
                ) : null}
                {llmHealth && <LlmHealthBadge health={llmHealth} />}
                <CliStatusBadge label="Notion CLI" connection={integrations?.notion ?? null} icon={BookOpen} />
                <CliStatusBadge label="GitHub CLI" connection={integrations?.github ?? null} icon={GitFork} />
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => void Promise.all([refresh(), refreshIntegrations()])}
                  className="size-7 rounded-lg"
                  title={lastUpdated ? `刷新状态 · ${lastUpdated.toLocaleTimeString()}` : "刷新状态"}
                >
                  <RefreshCw className="size-3.5" />
                  <span className="sr-only">刷新状态</span>
                </Button>
                {status ? (
                  <div
                    className="flex shrink-0 items-center rounded-xl border border-border/60 bg-muted/20 px-2 py-1 dark:border-white/10 dark:bg-white/[0.03]"
                    title={`daemon PID ${status.pid}`}
                  >
                    <TopMetric label="运行" value={fmtUptime(status.uptime_secs)} />
                    <TopDivider />
                    <TopMetric label="能力" value={status.capabilities} valueClassName="text-info" />
                    <TopDivider />
                    <TopMetric label="进化" value={status.total_evolutions} valueClassName="text-pos" />
                  </div>
                ) : (
                  <div className="w-[172px] shrink-0" aria-hidden="true" />
                )}
              </div>
            </header>
          )}

          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <div
            className={cn(
              "flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto overflow-x-hidden",
              isSettings ? "px-0 py-0" : "px-5 py-6 lg:px-8",
            )}
          >
              <div className={cn(
                "mx-auto flex w-full min-h-0 flex-1 flex-col",
                isSettings ? "" : "max-w-6xl gap-5",
              )}>
                {activeNav === "auto-evolve" && <AutoEvolvePage />}
                {activeNav === "tasks" && <TasksPage />}
                {activeNav === "capabilities" && <CapabilitiesPage caps={caps} />}
                {isSettings && (
                  <SettingsPage config={config} onUpdated={refresh} onExit={() => setActiveNav(prevNavRef.current)} />
                )}
              </div>
            </div>
          </div>
        </main>
      </div>

    </div>
  )
}

function TopMetric({
  label,
  value,
  valueClassName,
}: {
  label: string
  value: string | number
  valueClassName?: string
}) {
  return (
    <div className="flex items-center gap-1.5 whitespace-nowrap px-2 text-[11px]">
      <span className="font-medium text-muted-foreground">{label}</span>
      <span className={cn("font-semibold tabular-nums", valueClassName)}>{value}</span>
    </div>
  )
}

function TopDivider() {
  return <span className="h-3.5 w-px bg-border/70 dark:bg-white/10" />
}

function CliStatusBadge({
  label,
  connection,
  icon: Icon,
}: {
  label: string
  connection: CliConnection | null
  icon: typeof BookOpen
}) {
  const connected = connection?.healthy === true
  const state = !connection
    ? "检测中"
    : connected
      ? "已连接"
      : connection.installed
        ? "未登录"
        : "未安装"
  const title = connection
    ? [connection.version, connection.workspace, connection.account].filter(Boolean).join(" · ") || state
    : "正在检测 CLI 状态"

  return (
    <Badge variant={!connection ? "info" : connected ? "pos" : "warn"} className="gap-1.5" title={title}>
      <Icon className="size-3" />
      {label}{connected ? "" : ` · ${state}`}
    </Badge>
  )
}

function LlmHealthBadge({ health }: { health: LlmHealth }) {
  // 决定徽章文案 + 样式
  // - breaker open 且不在时段内 → "暂停" (neg) — 主动 + 被动双重停
  // - breaker open 但在时段内 → "熔断" (neg) — API 故障
  // - 不在时段内（breaker closed/half_open）→ "休眠" (warn) — 主动暂停，等待时段
  // - half_open（在时段内）→ "探测中" (warn) — 正在试探恢复
  // - closed 且在时段内 → "可用" (pos)
  // - state unknown → "未知" (warn)
  const { state, in_active_hours, active_hours, consecutive_failures } = health
  let label: string
  let variant: "pos" | "neg" | "warn" | "info"
  let title: string

  if (!in_active_hours && state !== "open") {
    label = "休眠"
    variant = "warn"
    title = active_hours
      ? `API 时段外（${active_hours}），进化暂停`
      : "API 时段外，进化暂停"
  } else if (state === "open") {
    label = in_active_hours ? "熔断" : "暂停"
    variant = "neg"
    title = in_active_hours
      ? `熔断中（连续 ${consecutive_failures} 次失败），等待恢复探测`
      : `熔断中 且 时段外（${active_hours ?? "?"}）`
  } else if (state === "half_open") {
    label = "探测中"
    variant = "warn"
    title = "熔断恢复探测中"
  } else if (state === "closed") {
    label = "可用"
    variant = "pos"
    title = "LLM API 正常"
  } else {
    label = "未知"
    variant = "warn"
    title = "熔断器状态未知（daemon 可能未使用 LlmExecutor）"
  }

  return (
    <Badge variant={variant} className="gap-1.5" title={title}>
      <Zap className="size-3" />
      {label}
    </Badge>
  )
}
