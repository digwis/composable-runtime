import { useEffect, useMemo, useState } from "react"
import {
  ArrowLeft,
  BookOpen,
  Check,
  CircleAlert,
  Cloud,
  Cpu,
  GitBranch,
  GitFork,
  Info,
  Monitor,
  Moon,
  Palette,
  Plus,
  RefreshCw,
  Sun,
  Trash2,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { applyTheme, persistTheme, readTheme, type ThemeMode } from "@/lib/theme"
import { Button } from "@/components/ui/button"
import type { DaemonConfig } from "@/lib/types"
import {
  findMatchingPreset,
  getActivePresetId,
  loadPresets,
  removePreset,
  setActivePresetId,
  type ApiPreset,
} from "@/lib/presets"

function fallbackConfigsFromPresets(presets: ApiPreset[]) {
  const configs: Array<{ api_key: string; base_url: string; model: string }> = []
  const seen = new Set<string>()
  for (const preset of presets) {
    const mapped = preset.role_mapping
      ? [preset.role_mapping.fast, preset.role_mapping.smart, preset.role_mapping.coder].filter(
          (model): model is string => !!model,
        )
      : preset.models
    for (const model of mapped) {
      const key = preset.base_url + "|" + model + "|" + preset.api_key
      if (!model || seen.has(key)) continue
      seen.add(key)
      configs.push({ api_key: preset.api_key, base_url: preset.base_url, model })
      if (configs.length >= 24) return configs
    }
  }
  return configs
}
import { PresetModal } from "@/components/PresetModal"

type SettingsSection = "appearance" | "api" | "connections" | "about"

const SECTION_KEYS: Array<SettingsSection> = ["appearance", "api", "connections", "about"]

const SECTION_ICONS: Record<SettingsSection, typeof Palette> = {
  appearance: Palette,
  api: Cpu,
  connections: Cloud,
  about: Info,
}

const SECTION_LABELS: Record<SettingsSection, string> = {
  appearance: "外观",
  api: "API 配置",
  connections: "云端连接",
  about: "关于",
}

const SECTION_DESCRIPTIONS: Record<SettingsSection, string> = {
  appearance: "主题和显示偏好",
  api: "LLM API 密钥、模型与连接测试",
  connections: "Notion 知识与 GitHub 能力仓库",
  about: "版本和技术信息",
}

const THEME_OPTIONS: Array<{ key: ThemeMode; label: string; desc: string; icon: typeof Sun }> = [
  { key: "light", label: "浅色", desc: "明亮的界面，适合白天使用", icon: Sun },
  { key: "dark", label: "深色", desc: "深色背景，减少眼部疲劳", icon: Moon },
  { key: "system", label: "跟随系统", desc: "根据系统设置自动切换", icon: Monitor },
]

export function SettingsPage({
  config,
  onUpdated,
  onExit,
}: {
  config: DaemonConfig | null
  onUpdated: () => void
  onExit: () => void
}) {
  const [activeSection, setActiveSection] = useState<SettingsSection>("appearance")
  const [theme, setTheme] = useState<ThemeMode>(() => readTheme())

  useEffect(() => {
    applyTheme(theme)
    persistTheme(theme)
  }, [theme])

  // system 模式下监听系统主题变化
  useEffect(() => {
    if (theme !== "system") return
    const mql = window.matchMedia("(prefers-color-scheme: dark)")
    const handler = () => applyTheme("system")
    mql.addEventListener("change", handler)
    return () => mql.removeEventListener("change", handler)
  }, [theme])

  return (
    <div className="flex min-h-0 flex-1 overflow-hidden bg-background">
      {/* 左侧导航 */}
      <aside className="flex w-[280px] shrink-0 flex-col border-r border-border/60 bg-sidebar/95 px-3 pb-6 pt-12 dark:border-white/10">
        <button
          type="button"
          onClick={onExit}
          className="mb-6 flex h-11 items-center gap-3 rounded-xl px-3 text-left text-muted-foreground transition hover:bg-background/40 hover:text-foreground dark:hover:bg-white/5"
        >
          <ArrowLeft className="size-5 shrink-0" />
          <span className="text-[15px] font-medium">返回</span>
        </button>

        <nav className="flex flex-col gap-1">
          <div className="px-3 pb-2 text-[12px] font-medium text-muted-foreground/75">设置</div>
          {SECTION_KEYS.map((key) => {
            const Icon = SECTION_ICONS[key]
            const active = key === activeSection
            return (
              <button
                key={key}
                type="button"
                onClick={() => setActiveSection(key)}
                className={cn(
                  "flex items-start gap-3 rounded-xl px-3 py-3 text-left transition",
                  active
                    ? "bg-background text-foreground shadow-sm ring-1 ring-border/60 dark:bg-white/10 dark:shadow-none dark:ring-white/10"
                    : "text-muted-foreground hover:bg-background/60 hover:text-foreground dark:hover:bg-white/5",
                )}
              >
                <Icon className={cn("mt-0.5 size-4 shrink-0", active ? "text-foreground/80" : "opacity-70")} />
                <div className="min-w-0">
                  <div className="text-sm font-medium">{SECTION_LABELS[key]}</div>
                  <div className="mt-0.5 text-xs text-muted-foreground">{SECTION_DESCRIPTIONS[key]}</div>
                </div>
              </button>
            )
          })}
        </nav>
      </aside>

      {/* 右侧内容 */}
      <section className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-[1280px] flex-col px-8 py-10 lg:px-10">
          {activeSection === "appearance" && (
            <AppearanceSection theme={theme} onSelect={setTheme} />
          )}
          {activeSection === "api" && (
            <ApiSection config={config} onUpdated={onUpdated} />
          )}
          {activeSection === "connections" && <ConnectionsSection />}
          {activeSection === "about" && <AboutSection />}
        </div>
      </section>
    </div>
  )
}

function SectionHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="mb-8">
      <h3 className="text-3xl font-semibold text-foreground">{title}</h3>
      <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>
    </div>
  )
}

function AppearanceSection({
  theme,
  onSelect,
}: {
  theme: ThemeMode
  onSelect: (m: ThemeMode) => void
}) {
  return (
    <>
      <SectionHeader title="外观" subtitle="自定义界面主题和显示偏好" />
      <div className="space-y-6">
        <div className="space-y-3">
          <div>
            <p className="text-sm font-medium text-foreground">主题模式</p>
            <p className="mt-1 text-sm text-muted-foreground">选择应用的配色方案</p>
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            {THEME_OPTIONS.map((opt) => {
              const selected = theme === opt.key
              return (
                <button
                  key={opt.key}
                  type="button"
                  onClick={() => onSelect(opt.key)}
                  className={cn(
                    "relative min-h-[200px] rounded-3xl border p-6 text-left transition",
                    selected
                      ? "border-primary/40 bg-primary/[0.07] shadow-sm ring-1 ring-primary/20 dark:border-white/15 dark:bg-white/[0.08] dark:shadow-none dark:ring-white/10"
                      : "border-border/70 bg-background hover:border-border hover:bg-muted/20 dark:border-white/10 dark:bg-white/[0.02] dark:hover:bg-white/[0.05]",
                  )}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="grid size-11 place-items-center rounded-2xl bg-muted text-muted-foreground dark:bg-white/10">
                      <opt.icon className="size-5" />
                    </div>
                    {selected && (
                      <span className="inline-flex size-7 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-4" />
                      </span>
                    )}
                  </div>
                  <div className="mt-8">
                    <div className="text-xl font-semibold text-foreground">{opt.label}</div>
                    <div className="mt-2 text-sm leading-7 text-muted-foreground">{opt.desc}</div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </>
  )
}

function ApiSection({
  config,
  onUpdated,
}: {
  config: DaemonConfig | null
  onUpdated: () => void
}) {
  const [savedMsg, setSavedMsg] = useState("")

  // 预设：localStorage 中保存多份命名配置（一个供应商 = 名称 + key + url + 多个模型）
  const [presets, setPresets] = useState<ApiPreset[]>([])
  const [activePresetId, setActiveId] = useState<string | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [editingPreset, setEditingPreset] = useState<ApiPreset | null>(null)
  const [applyingKey, setApplyingKey] = useState<string | null>(null)

  useEffect(() => {
    void (async () => {
      const [saved, active] = await Promise.all([loadPresets(), getActivePresetId()])
      setPresets(saved)
      setActiveId(active)
    })()
  }, [])

  const hasConfig = !!config

  // daemon 当前使用的模型集合（fast/smart/coder），用于高亮预设卡片中的模型标签
  const activeModels = useMemo(() => {
    if (!config) return new Set<string>()
    return new Set([config.fast_model, config.smart_model, config.coder_model].filter(Boolean))
  }, [config])

  // 同时更新 state 与 localStorage 的激活预设 id
  const updateActiveId = (id: string | null) => {
    setActiveId(id)
    void setActivePresetId(id)
  }

  // daemon 配置变化后，根据 base_url + has_key 反查匹配的预设
  useEffect(() => {
    if (!config) return
    const matched = findMatchingPreset(presets, config)
    if (matched) {
      updateActiveId(matched.preset.id)
    } else {
      updateActiveId(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config])

  // 应用预设的某个模型到 daemon
  // 若预设配置了完整的 role_mapping，则发送 fast/smart/coder 三个字段；否则用 model 字段
  const handleApplyModel = async (preset: ApiPreset, model: string) => {
    const key = `${preset.id}:${model}`
    setApplyingKey(key)
    setSavedMsg("")
    const body: Record<string, unknown> = {
      api_key: preset.api_key,
      base_url: preset.base_url,
      fallback_configs: fallbackConfigsFromPresets(presets),
    }
    const hasFullMapping =
      !!preset.role_mapping?.fast &&
      !!preset.role_mapping.smart &&
      !!preset.role_mapping.coder
    if (hasFullMapping) {
      body.fast_model = preset.role_mapping!.fast!
      body.smart_model = preset.role_mapping!.smart!
      body.coder_model = preset.role_mapping!.coder!
    } else {
      body.model = model
    }
    try {
      const resp = await window.daemon.updateConfig(body)
      const data = resp.data as Record<string, unknown> | undefined
      if (resp.ok && resp.status < 400 && data?.success !== false) {
        const pi = await window.daemon.syncPiConfig({
          apiKey: preset.api_key,
          baseUrl: preset.base_url,
          model,
          models: preset.models,
        })
        updateActiveId(preset.id)
        onUpdated()
        const label = hasFullMapping
          ? `fast=${preset.role_mapping!.fast} · smart=${preset.role_mapping!.smart} · coder=${preset.role_mapping!.coder}`
          : model
        setSavedMsg(pi.ok ? `已切换到「${preset.name}」· ${label} · pi 已同步` : `已切换到「${preset.name}」· ${label}（pi 同步失败：${pi.message}）`)
        setTimeout(() => setSavedMsg(""), 4000)
      } else {
        const err =
          (data?.error as string) ?? (data?.message as string) ?? `切换失败（HTTP ${resp.status}）`
        setSavedMsg(err)
      }
    } catch (e) {
      setSavedMsg(String(e))
    } finally {
      setApplyingKey(null)
    }
  }

  // 直接在「当前生效配置」区域修改三个角色模型并应用
  // 每个角色可携带独立的 api_key + base_url（支持跨供应商）
  const handleApplyRoles = async (roles: {
    fast: { model: string; apiKey?: string; baseUrl?: string }
    smart: { model: string; apiKey?: string; baseUrl?: string }
    coder: { model: string; apiKey?: string; baseUrl?: string }
  }) => {
    setApplyingKey("__roles__")
    setSavedMsg("")
    try {
      const body: Record<string, unknown> = {
        fast_model: roles.fast.model,
        smart_model: roles.smart.model,
        coder_model: roles.coder.model,
        fallback_configs: fallbackConfigsFromPresets(presets),
      }
      if (roles.fast.apiKey) body.fast_api_key = roles.fast.apiKey
      if (roles.fast.baseUrl) body.fast_base_url = roles.fast.baseUrl
      if (roles.smart.apiKey) body.smart_api_key = roles.smart.apiKey
      if (roles.smart.baseUrl) body.smart_base_url = roles.smart.baseUrl
      if (roles.coder.apiKey) body.coder_api_key = roles.coder.apiKey
      if (roles.coder.baseUrl) body.coder_base_url = roles.coder.baseUrl
      const resp = await window.daemon.updateConfig(body)
      const data = resp.data as Record<string, unknown> | undefined
      if (resp.ok && resp.status < 400 && data?.success !== false) {
        onUpdated()
        const fmtRole = (r: { model: string; baseUrl?: string }) =>
          r.baseUrl ? `${r.model}@${r.baseUrl}` : r.model
        setSavedMsg(
          `已应用：fast=${fmtRole(roles.fast)} · smart=${fmtRole(roles.smart)} · coder=${fmtRole(roles.coder)}`,
        )
        setTimeout(() => setSavedMsg(""), 4000)
      } else {
        const err =
          (data?.error as string) ??
          (data?.message as string) ??
          `应用失败（HTTP ${resp.status}）`
        setSavedMsg(err)
      }
    } catch (e) {
      setSavedMsg(String(e))
    } finally {
      setApplyingKey(null)
    }
  }

  const handleOpenNew = () => {
    setEditingPreset(null)
    setModalOpen(true)
  }

  const handleOpenEdit = (preset: ApiPreset) => {
    setEditingPreset(preset)
    setModalOpen(true)
  }

  const handleSaved = (preset: ApiPreset) => {
    void loadPresets().then(setPresets)
    setSavedMsg(`预设「${preset.name}」已保存`)
    setTimeout(() => setSavedMsg(""), 3000)
  }

  const handleDeletePreset = (id: string) => {
    const target = presets.find((p) => p.id === id)
    void removePreset(id).then(() => loadPresets().then(setPresets))
    if (activePresetId === id) {
      updateActiveId(null)
    }
    if (target) {
      setSavedMsg(`已删除预设「${target.name}」`)
      setTimeout(() => setSavedMsg(""), 3000)
    }
  }

  return (
    <>
      <SectionHeader
        title="API 配置"
        subtitle="管理 LLM 供应商预设，一键切换到 daemon — 下一次 LLM 调用即生效"
      />

      {!hasConfig && (
        <div className="mb-6 rounded-2xl border border-border/60 bg-muted/30 px-5 py-4 text-sm text-muted-foreground dark:border-white/10 dark:bg-white/[0.03]">
          未读到配置（daemon 可能未启动或用 CLI driver）— 预设可保存，但切换与测试需 daemon 在线
        </div>
      )}

      <div className="space-y-6">
        {/* 当前生效配置（可编辑角色模型） */}
        {hasConfig && (
          <ActiveConfigEditor
            config={config!}
            presets={presets}
            applyingKey={applyingKey}
            onApply={handleApplyRoles}
          />
        )}

        {/* 使用时段（active hours）：设置 LLM API 的开放时段 */}
        {hasConfig && <ActiveHoursCard config={config!} onUpdated={onUpdated} />}

        {/* 配置预设：供应商列表，点击模型一键切换 */}
        <div className="rounded-2xl border border-border/60 bg-card px-5 py-4 shadow-sm dark:border-white/10 dark:bg-[#242424]">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-foreground">配置预设</p>
              <p className="mt-1 text-xs text-muted-foreground">
                每个预设对应一个供应商（名称 + Key + URL + 多个模型），点击模型即切换到 daemon
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={handleOpenNew} className="gap-2">
              <Plus className="size-4" />
              新增预设
            </Button>
          </div>

          {/* 预设列表 */}
          {presets.length === 0 ? (
            <p className="mt-4 text-xs text-muted-foreground">
              暂无预设。点击「新增预设」打开弹窗配置供应商名称、API Key、Base URL 与多个模型（弹窗内可测试）。
            </p>
          ) : (
            <div className="mt-4 space-y-3">
              {presets.map((p) => {
                const active = p.id === activePresetId
                return (
                  <div
                    key={p.id}
                    className={cn(
                      "group rounded-xl border p-4 transition",
                      active
                        ? "border-primary/40 bg-primary/[0.04] ring-1 ring-primary/15 dark:border-white/15 dark:bg-white/[0.05] dark:ring-white/10"
                        : "border-border/60 bg-background hover:border-border hover:bg-muted/20 dark:border-white/10 dark:bg-white/[0.02] dark:hover:bg-white/[0.04]",
                    )}
                  >
                    {/* 头部：名称 + URL + 操作 */}
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          {active && (
                            <span
                              className="inline-flex size-2 shrink-0 rounded-full bg-pos"
                              title="当前生效"
                            />
                          )}
                          <span className="truncate text-sm font-semibold text-foreground">
                            {p.name}
                          </span>
                          <span className="shrink-0 rounded-md bg-muted/60 px-1.5 py-0.5 text-[10px] text-muted-foreground dark:bg-white/10">
                            {p.models.length} 模型
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
                          <span className="truncate font-mono">{p.base_url || "—"}</span>
                          <span className="shrink-0 font-mono">
                            {p.api_key
                              ? `${p.api_key.slice(0, 6)}••••${p.api_key.slice(-4)}`
                              : "—"}
                          </span>
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1">
                        <button
                          type="button"
                          onClick={() => handleOpenEdit(p)}
                          className="grid size-7 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground dark:hover:bg-white/10"
                          title="编辑预设"
                        >
                          <Cpu className="size-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeletePreset(p.id)}
                          className="grid size-7 place-items-center rounded-md text-muted-foreground/70 transition hover:bg-neg/10 hover:text-neg"
                          title="删除预设"
                        >
                          <Trash2 className="size-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* 模型列表：点击即应用 */}
                    <div className="mt-3 flex flex-wrap gap-2">
                      {p.models.map((m) => {
                        // 高亮所有被 daemon 当前使用的模型（fast/smart/coder 任一匹配）
                        const isActiveModel = active && activeModels.has(m)
                        const key = `${p.id}:${m}`
                        const applying = applyingKey === key
                        // 显示该模型被分配到的角色
                        const rm = p.role_mapping
                        const roles: string[] = []
                        if (rm?.fast === m) roles.push("fast")
                        if (rm?.smart === m) roles.push("smart")
                        if (rm?.coder === m) roles.push("coder")
                        return (
                          <button
                            key={m}
                            type="button"
                            onClick={() => handleApplyModel(p, m)}
                            disabled={applying || !hasConfig}
                            className={cn(
                              "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 font-mono text-xs transition disabled:opacity-50",
                              isActiveModel
                                ? "border-pos/40 bg-pos/10 text-pos dark:border-pos/30 dark:bg-pos/[0.12]"
                                : "border-border/60 bg-muted/30 text-foreground/80 hover:border-primary/40 hover:bg-primary/[0.06] hover:text-foreground dark:border-white/10 dark:bg-white/[0.04] dark:hover:bg-white/[0.08]",
                            )}
                            title={
                              isActiveModel
                                ? "当前生效"
                                : roles.length > 0
                                  ? `角色：${roles.join(" / ")}`
                                  : "点击切换到此模型"
                            }
                          >
                            {applying ? (
                              <RefreshCw className="size-3 animate-spin" />
                            ) : isActiveModel ? (
                              <Check className="size-3" />
                            ) : null}
                            {m}
                            {roles.length > 0 && (
                              <span className="ml-0.5 rounded bg-primary/15 px-1 py-0.5 text-[10px] text-primary dark:bg-primary/25">
                                {roles.join("/")}
                              </span>
                            )}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* 操作区 */}
        {savedMsg && (
          <div className="flex flex-wrap items-center gap-4">
            <span
              className={cn(
                "text-sm",
                savedMsg.startsWith("已") ? "text-pos" : "text-neg",
              )}
            >
              {savedMsg}
            </span>
          </div>
        )}
      </div>

      {/* 新增/编辑预设弹窗 */}
      <PresetModal
        open={modalOpen}
        preset={editingPreset}
        daemonAvailable={hasConfig}
        onClose={() => setModalOpen(false)}
        onSaved={handleSaved}
      />
    </>
  )
}

function AboutSection() {
  return (
    <>
      <SectionHeader title="关于" subtitle="应用版本和技术信息" />
      <div className="space-y-2">
        <InfoRow label="应用名称" value="Evolution Dashboard" />
        <InfoRow label="版本" value="0.1.0" />
        <InfoRow label="前端框架" value="Electron + React + Tailwind" />
        <InfoRow label="后端" value="composable-runtime daemon" />
        <InfoRow label="状态管理" value="React hooks" />
      </div>
    </>
  )
}

type ServiceConnection = {
  id: string
  role: string
  status: string
  installed: boolean
  authenticated: boolean
  healthy: boolean
  cli_path?: string | null
  version?: string | null
  account?: string | null
  account_name?: string | null
  workspace?: string | null
  scopes?: string[]
  warnings?: string[]
  error?: string | null
}

type IntegrationStatus = {
  checked_at: number
  knowledge_backend: string
  capability_backend: string
  notion: ServiceConnection
  github: ServiceConnection
  git: ServiceConnection
}

type CloudResource = { id: string; name: string; url?: string | null; kind: string; private?: boolean | null }
type CloudResourceState = {
  initialized_at?: number | null
  notion_root?: CloudResource | null
  notion_children?: CloudResource[]
  github_capability_repository?: CloudResource | null
  last_github_sync_at?: number | null
  last_notion_sync_at?: number | null
  github_sync_changed?: boolean
  last_sync_errors?: string[]
  warnings?: string[]
}

function ConnectionsSection() {
  const [status, setStatus] = useState<IntegrationStatus | null>(null)
  const [resources, setResources] = useState<CloudResourceState | null>(null)
  const [loading, setLoading] = useState(true)
  const [bootstrapping, setBootstrapping] = useState(false)

  const refresh = async () => {
    setLoading(true)
    try {
      const response = await window.daemon.integrationsStatus()
      if (response.ok && response.status === 200) {
        const data = response.data as { integrations?: IntegrationStatus; resources?: CloudResourceState }
        setStatus(data.integrations ?? null)
        setResources(data.resources ?? null)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const bootstrap = async () => {
    setBootstrapping(true)
    try {
      await window.daemon.bootstrapIntegrations()
      await refresh()
    } finally {
      setBootstrapping(false)
    }
  }

  const connections = status
    ? [
        { connection: status.notion, resource: resources?.notion_root, label: "Notion", role: "知识与长期记忆", Icon: BookOpen },
        { connection: status.github, resource: resources?.github_capability_repository, label: "GitHub", role: "能力代码与版本", Icon: GitFork },
        { connection: status.git, resource: null, label: "Git", role: "本地能力验证", Icon: GitBranch },
      ]
    : []

  return (
    <>
      <div className="mb-8 flex items-start justify-between gap-4">
        <SectionHeader title="云端连接" subtitle="本地自动检测，不读取或展示访问凭证" />
        <div className="mt-1 flex gap-2">
          <Button size="sm" variant="outline" className="gap-1.5" onClick={() => void refresh()} disabled={loading || bootstrapping}>
            <RefreshCw className={cn("size-3.5", loading && "animate-spin")} />重新检测
          </Button>
          <Button size="sm" className="gap-1.5" onClick={() => void bootstrap()} disabled={loading || bootstrapping}>
            <Cloud className={cn("size-3.5", bootstrapping && "animate-pulse")} />{bootstrapping ? "同步中" : "初始化并同步"}
          </Button>
        </div>
      </div>

      {loading && !status ? (
        <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
          <RefreshCw className="mr-2 size-4 animate-spin" />正在检测本地连接
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-3">
          {connections.map(({ connection, resource, label, role, Icon }) => (
            <div key={connection.id} className="rounded-lg border border-border/70 bg-background p-5 dark:border-white/10 dark:bg-white/[0.02]">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="grid size-10 place-items-center rounded-lg bg-muted text-muted-foreground dark:bg-white/10">
                    <Icon className="size-5" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">{label}</div>
                    <div className="mt-0.5 text-xs text-muted-foreground">{role}</div>
                  </div>
                </div>
                <span className={cn("inline-flex items-center gap-1 text-xs font-medium", connection.healthy ? "text-pos" : "text-warn")}>
                  {connection.healthy ? <Check className="size-3.5" /> : <CircleAlert className="size-3.5" />}
                  {connection.healthy ? "已连接" : connection.installed ? "未登录" : "未安装"}
                </span>
              </div>

              <div className="mt-5 space-y-2 border-t border-border/60 pt-4 text-xs">
                {connection.workspace && <div className="flex justify-between gap-3"><span className="text-muted-foreground">工作区</span><span className="truncate text-foreground">{connection.workspace}</span></div>}
                {connection.account && <div className="flex justify-between gap-3"><span className="text-muted-foreground">账号</span><span className="truncate text-foreground">{connection.account}</span></div>}
                {connection.account_name && connection.account_name !== connection.account && <div className="flex justify-between gap-3"><span className="text-muted-foreground">用户</span><span className="truncate text-foreground">{connection.account_name}</span></div>}
                {connection.version && <div className="flex justify-between gap-3"><span className="text-muted-foreground">CLI</span><span className="truncate text-foreground">{connection.version}</span></div>}
                {connection.scopes && connection.scopes.length > 0 && <div className="flex justify-between gap-3"><span className="text-muted-foreground">权限</span><span className="text-right text-foreground">{connection.scopes.join(" · ")}</span></div>}
                {resource && <div className="flex justify-between gap-3"><span className="text-muted-foreground">云端资源</span><span className="truncate text-foreground">{resource.name}{resource.private === true ? " · 私有" : ""}</span></div>}
                {connection.error && <div className="rounded-md bg-warn/10 px-2.5 py-2 text-warn">{connection.error}</div>}
                {connection.warnings?.map((warning) => <div key={warning} className="rounded-md bg-warn/10 px-2.5 py-2 text-warn">{warning}</div>)}
              </div>
            </div>
          ))}
        </div>
      )}

      {resources?.notion_children && resources.notion_children.length > 0 && <div className="mt-4 text-xs text-muted-foreground">Notion 结构：{resources.notion_children.map((resource) => resource.name).join(" · ")}</div>}
      {(resources?.last_github_sync_at || resources?.last_notion_sync_at) && <div className="mt-3 grid gap-1 text-xs text-muted-foreground sm:grid-cols-2">
        <div>GitHub 最近同步：{resources.last_github_sync_at ? new Date(resources.last_github_sync_at * 1000).toLocaleString() : "尚未同步"}</div>
        <div>Notion 最近同步：{resources.last_notion_sync_at ? new Date(resources.last_notion_sync_at * 1000).toLocaleString() : "尚未同步"}</div>
      </div>}
      {resources?.last_sync_errors?.map((error) => <div key={error} className="mt-3 rounded-md bg-neg/10 px-3 py-2 text-xs text-neg">{error}</div>)}
      {resources?.warnings?.map((warning) => <div key={warning} className="mt-3 rounded-md bg-warn/10 px-3 py-2 text-xs text-warn">{warning}</div>)}
      {status && <div className="mt-5 text-xs text-muted-foreground">上次检测：{new Date(status.checked_at * 1000).toLocaleString()}</div>}
    </>
  )
}

type RoleSelection = {
  model: string
  presetId: string | null
  apiKey?: string
  baseUrl?: string
}

function ActiveConfigEditor({
  config,
  presets,
  applyingKey,
  onApply,
}: {
  config: DaemonConfig
  presets: ApiPreset[]
  applyingKey: string | null
  onApply: (roles: {
    fast: { model: string; apiKey?: string; baseUrl?: string }
    smart: { model: string; apiKey?: string; baseUrl?: string }
    coder: { model: string; apiKey?: string; baseUrl?: string }
  }) => Promise<void> | void
}) {
  // 查找模型所属的预设（按 base_url + api_key 匹配）
  const findPresetForModel = (modelName: string, roleBaseUrl?: string): ApiPreset | null => {
    // 优先匹配 base_url 一致的预设
    if (roleBaseUrl) {
      const match = presets.find((p) => p.base_url === roleBaseUrl && p.models.includes(modelName))
      if (match) return match
    }
    // 回退：任意包含该模型的预设
    return presets.find((p) => p.models.includes(modelName)) ?? null
  }

  // 从 daemon config 初始化角色选择
  const initRole = (role: "fast" | "smart" | "coder"): RoleSelection => {
    const modelName = config[`${role}_model` as keyof DaemonConfig] as string
    const roleInfo = config.roles?.[role]
    const preset = findPresetForModel(modelName, roleInfo?.base_url)
    return {
      model: modelName,
      presetId: preset?.id ?? null,
      apiKey: preset?.api_key,
      baseUrl: preset?.base_url ?? roleInfo?.base_url,
    }
  }

  const [fastSel, setFastSel] = useState<RoleSelection>(() => initRole("fast"))
  const [smartSel, setSmartSel] = useState<RoleSelection>(() => initRole("smart"))
  const [coderSel, setCoderSel] = useState<RoleSelection>(() => initRole("coder"))

  // daemon 配置变化时同步本地状态
  useEffect(() => {
    setFastSel(initRole("fast"))
    setSmartSel(initRole("smart"))
    setCoderSel(initRole("coder"))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.fast_model, config.smart_model, config.coder_model, config.roles])

  // 所有预设中的模型选项，按预设分组：value 用 "presetId::modelName" 编码
  type ModelOption = { value: string; label: string; preset: ApiPreset }
  const allModelOptions: ModelOption[] = useMemo(() => {
    const opts: ModelOption[] = []
    for (const p of presets) {
      for (const m of p.models) {
        opts.push({ value: `${p.id}::${m}`, label: `${m} (${p.name})`, preset: p })
      }
    }
    return opts
  }, [presets])

  // 当前生效模型也作为兜底选项
  const fallbackOptions = useMemo(() => {
    const opts: ModelOption[] = []
    const seen = new Set(allModelOptions.map((o) => o.value))
    for (const [role, model] of [
      ["fast", config.fast_model],
      ["smart", config.smart_model],
      ["coder", config.coder_model],
    ] as const) {
      if (model && !seen.has(`__current__::${model}`)) {
        opts.push({
          value: `__current__::${model}`,
          label: `${model} (当前)`,
          preset: { id: "__current__", name: "当前", api_key: "", base_url: config.base_url, models: [model], created_at: 0, updated_at: 0 },
        })
      }
    }
    return opts
  }, [allModelOptions, config.fast_model, config.smart_model, config.coder_model, config.base_url])

  const selectOptions = [...allModelOptions, ...fallbackOptions]
  const uniqueSelectOptions = selectOptions.filter((option, index, options) =>
    options.findIndex((candidate) => candidate.value === option.value) === index,
  )

  const encodeSelection = (sel: RoleSelection): string => {
    if (sel.presetId) return `${sel.presetId}::${sel.model}`
    return `__current__::${sel.model}`
  }

  const decodeSelection = (value: string, role: "fast" | "smart" | "coder"): RoleSelection => {
    const [presetId, modelName] = value.split("::")
    if (presetId === "__current__") {
      return { model: modelName, presetId: null, baseUrl: config.base_url }
    }
    const preset = presets.find((p) => p.id === presetId)
    if (!preset) {
      const fallback = config[`${role}_model` as keyof DaemonConfig] as string
      return { model: fallback, presetId: null }
    }
    return {
      model: modelName,
      presetId: preset.id,
      apiKey: preset.api_key,
      baseUrl: preset.base_url,
    }
  }

  const dirty =
    fastSel.model !== config.fast_model ||
    smartSel.model !== config.smart_model ||
    coderSel.model !== config.coder_model ||
    fastSel.baseUrl !== (config.roles?.fast?.base_url ?? config.base_url) ||
    smartSel.baseUrl !== (config.roles?.smart?.base_url ?? config.base_url) ||
    coderSel.baseUrl !== (config.roles?.coder?.base_url ?? config.base_url)

  const applying = applyingKey === "__roles__"

  const handleApply = () => {
    if (!dirty || applying) return
    onApply({
      fast: { model: fastSel.model, apiKey: fastSel.apiKey, baseUrl: fastSel.baseUrl },
      smart: { model: smartSel.model, apiKey: smartSel.apiKey, baseUrl: smartSel.baseUrl },
      coder: { model: coderSel.model, apiKey: coderSel.apiKey, baseUrl: coderSel.baseUrl },
    })
  }

  const handleReset = () => {
    setFastSel(initRole("fast"))
    setSmartSel(initRole("smart"))
    setCoderSel(initRole("coder"))
  }

  const roles: Array<{
    key: "fast" | "smart" | "coder"
    label: string
    sel: RoleSelection
    setter: (sel: RoleSelection) => void
  }> = [
    { key: "fast", label: "fast", sel: fastSel, setter: setFastSel },
    { key: "smart", label: "smart", sel: smartSel, setter: setSmartSel },
    { key: "coder", label: "coder", sel: coderSel, setter: setCoderSel },
  ]

  return (
    <div className="rounded-2xl border border-border/60 bg-card px-5 py-4 shadow-sm dark:border-white/10 dark:bg-[#242424]">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-foreground">当前生效配置</p>
          <p className="mt-1 text-xs text-muted-foreground">
            为 fast/smart/coder 三个角色分别选择模型（可跨供应商），修改后点击「应用」即时生效
          </p>
        </div>
        <div className="flex items-center gap-2">
          {dirty && (
            <Button variant="ghost" size="sm" onClick={handleReset} disabled={applying}>
              撤销
            </Button>
          )}
          <Button size="sm" onClick={handleApply} disabled={!dirty || applying} className="gap-1.5">
            {applying ? <RefreshCw className="size-3.5 animate-spin" /> : null}
            应用
          </Button>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        {roles.map((r) => (
          <div key={r.key} className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-3">
            <div className="w-16 shrink-0">
              <span className="font-mono text-xs font-medium text-foreground">{r.label}</span>
            </div>
            <select
              value={encodeSelection(r.sel)}
              onChange={(e) => r.setter(decodeSelection(e.target.value, r.key))}
              disabled={applying}
              className="h-9 flex-1 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50"
            >
              {uniqueSelectOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            {r.sel.baseUrl && (
              <span className="truncate text-xs text-muted-foreground sm:max-w-[200px]" title={r.sel.baseUrl}>
                {r.sel.baseUrl}
              </span>
            )}
          </div>
        ))}
      </div>

      {uniqueSelectOptions.length === 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          提示：下方「配置预设」中添加供应商的模型列表后，可在此快速选择。
        </p>
      )}
    </div>
  )
}

function ActiveHoursCard({
  config,
  onUpdated,
}: {
  config: DaemonConfig
  onUpdated: () => void
}) {
  const aw = config.active_hours // {start,end} | null | undefined
  const [start, setStart] = useState(aw?.start ?? "23:00")
  const [end, setEnd] = useState(aw?.end ?? "09:00")
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState("")

  // daemon 配置变化后同步本地时间输入
  useEffect(() => {
    setStart(aw?.start ?? "23:00")
    setEnd(aw?.end ?? "09:00")
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aw?.start, aw?.end])

  const dirty = (aw?.start ?? "23:00") !== start || (aw?.end ?? "09:00") !== end

  const handleSave = async () => {
    setSaving(true)
    setMsg("")
    try {
      const resp = await window.daemon.updateConfig({ active_hours: { start, end } })
      const data = resp.data as Record<string, unknown> | undefined
      if (resp.ok && resp.status < 400 && data?.success !== false) {
        setMsg("已保存使用时段")
        onUpdated()
        setTimeout(() => setMsg(""), 4000)
      } else {
        setMsg(
          (data?.error as string) ??
            (data?.message as string) ??
            `保存失败（HTTP ${resp.status}）`,
        )
      }
    } catch (e) {
      setMsg(String(e))
    } finally {
      setSaving(false)
    }
  }

  const handleClear = async () => {
    setSaving(true)
    setMsg("")
    try {
      const resp = await window.daemon.updateConfig({ active_hours_clear: true })
      const data = resp.data as Record<string, unknown> | undefined
      if (resp.ok && resp.status < 400 && data?.success !== false) {
        setMsg("已清除使用时段")
        onUpdated()
        setTimeout(() => setMsg(""), 4000)
      } else {
        setMsg((data?.error as string) ?? `清除失败（HTTP ${resp.status}）`)
      }
    } catch (e) {
      setMsg(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-2xl border border-border/60 bg-card px-5 py-4 shadow-sm dark:border-white/10 dark:bg-[#242424]">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-foreground">使用时段</p>
          <p className="mt-1 text-xs text-muted-foreground">
            设置 LLM API 的开放时段；时段外自动暂停进化（支持跨天，如 23:00-09:00）
          </p>
        </div>
        <div className="flex items-center gap-2">
          {aw && (
            <Button variant="ghost" size="sm" onClick={handleClear} disabled={saving}>
              清除
            </Button>
          )}
          <Button size="sm" onClick={handleSave} disabled={!dirty || saving} className="gap-1.5">
            {saving ? <RefreshCw className="size-3.5 animate-spin" /> : null}
            保存
          </Button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">开始</span>
          <input
            type="time"
            value={start}
            onChange={(e) => setStart(e.target.value)}
            disabled={saving}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">结束</span>
          <input
            type="time"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            disabled={saving}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50"
          />
        </label>
        <span className="text-xs text-muted-foreground">
          {aw ? (
            <>
              当前：<span className="font-mono text-foreground">{aw.start}-{aw.end}</span>
            </>
          ) : (
            "未设置，全天可用"
          )}
        </span>
      </div>

      {msg && (
        <p className={cn("mt-3 text-sm", msg.startsWith("已") ? "text-pos" : "text-neg")}>
          {msg}
        </p>
      )}
    </div>
  )
}

function InfoLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/40 px-3 py-2 dark:bg-white/[0.04]">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="truncate font-mono text-xs text-foreground">{value || "—"}</span>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-card px-5 py-3.5 shadow-sm dark:border-white/10 dark:bg-[#242424]">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-medium text-foreground">{value}</span>
    </div>
  )
}
