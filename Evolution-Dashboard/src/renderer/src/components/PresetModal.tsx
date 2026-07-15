import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  Check,
  Eye,
  EyeOff,
  Info,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  addPreset,
  updatePreset,
  isRoleMappingComplete,
  LLM_ROLES,
  type ApiPreset,
  type LlmRole,
  type RoleMapping,
} from "@/lib/presets"

interface TestResult {
  status: "ok" | "error"
  result?: string
  error?: string
  model?: string
  elapsed_secs?: number
}

export interface PresetModalProps {
  open: boolean
  // 传入已有预设则为编辑模式，null 为新建
  preset: ApiPreset | null
  // daemon 是否可用（决定测试/保存是否可写 daemon）
  daemonAvailable: boolean
  onClose: () => void
  onSaved: (preset: ApiPreset) => void
}

export function PresetModal({
  open,
  preset,
  daemonAvailable,
  onClose,
  onSaved,
}: PresetModalProps) {
  const isEdit = !!preset
  const [name, setName] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [baseUrl, setBaseUrl] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [newModel, setNewModel] = useState("")
  // 角色映射：fast/smart/coder → model
  const [roleMapping, setRoleMapping] = useState<RoleMapping>({})
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState("")
  const [testing, setTesting] = useState(false)
  const [testModel, setTestModel] = useState("")
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  const modelInputRef = useRef<HTMLInputElement | null>(null)

  // 打开/切换 preset 时填充字段
  useEffect(() => {
    if (!open) return
    if (preset) {
      setName(preset.name)
      setApiKey(preset.api_key)
      setBaseUrl(preset.base_url)
      setModels([...preset.models])
      setRoleMapping(preset.role_mapping ? { ...preset.role_mapping } : {})
      setTestModel(preset.models[0] ?? "")
    } else {
      setName("")
      setApiKey("")
      setBaseUrl("")
      setModels([])
      setRoleMapping({})
      setTestModel("")
    }
    setNewModel("")
    setSaveMsg("")
    setTestResult(null)
  }, [open, preset])

  // 测试超时兜底
  useEffect(() => {
    if (!testing) return
    const id = setTimeout(() => {
      setTesting((prev) => {
        if (prev) {
          setTestResult({
            status: "error",
            error: "请求超时（35s），请检查 API URL 是否可达或网络连接",
          })
        }
        return false
      })
    }, 35000)
    return () => clearTimeout(id)
  }, [testing])

  const canSave = useMemo(() => {
    return name.trim().length > 0 && apiKey.trim().length > 0 && models.length > 0
  }, [name, apiKey, models])

  if (!open) return null

  const addModel = () => {
    const m = newModel.trim()
    if (!m) return
    if (models.includes(m)) {
      setNewModel("")
      return
    }
    const next = [...models, m]
    setModels(next)
    setNewModel("")
    if (!testModel) setTestModel(m)
    modelInputRef.current?.focus()
  }

  const removeModel = (m: string) => {
    const next = models.filter((x) => x !== m)
    setModels(next)
    if (testModel === m) setTestModel(next[0] ?? "")
    // 清理引用了被删模型的角色映射
    setRoleMapping((prev) => {
      const updated = { ...prev }
      if (updated.fast === m) delete updated.fast
      if (updated.smart === m) delete updated.smart
      if (updated.coder === m) delete updated.coder
      return updated
    })
  }

  const setRole = (role: LlmRole, model: string | "") => {
    setRoleMapping((prev) => {
      const next = { ...prev }
      if (model === "") delete next[role]
      else next[role] = model
      return next
    })
  }

  // 自动分配：把 models 前 3 个填入 fast/smart/coder（不足则留空）
  const autoAssignRoles = () => {
    const next: RoleMapping = {}
    if (models[0]) next.fast = models[0]
    if (models[1]) next.smart = models[1]
    if (models[2]) next.coder = models[2]
    setRoleMapping(next)
  }

  const handleTest = async () => {
    if (!apiKey.trim() || !testModel) return
    setTesting(true)
    setTestResult(null)
    const body: Record<string, string> = {
      api_key: apiKey.trim(),
      base_url: baseUrl.trim(),
      model: testModel,
    }
    try {
      const resp = await window.daemon.testLlm(body)
      const data = resp.data as Record<string, unknown> | undefined
      const status = (data?.status as string) ?? (resp.ok && resp.status < 400 ? "ok" : "error")
      setTestResult({
        status: status === "ok" ? "ok" : "error",
        result: data?.result as string | undefined,
        error:
          (data?.error as string) ??
          (resp.status === 0 ? "daemon 可能未启动" : `HTTP ${resp.status}`),
        model: data?.model as string | undefined,
        elapsed_secs: data?.elapsed_secs as number | undefined,
      })
    } catch (e) {
      setTestResult({ status: "error", error: String(e) })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    if (!canSave) return
    setSaving(true)
    setSaveMsg("")
    try {
      const payload = {
        name: name.trim(),
        api_key: apiKey.trim(),
        base_url: baseUrl.trim(),
        models,
        // 只在完整指定三个角色时才保存 role_mapping，否则视为未设置
        role_mapping:
          roleMapping.fast && roleMapping.smart && roleMapping.coder ? roleMapping : undefined,
      }
      let result: ApiPreset
      if (isEdit && preset) {
        await updatePreset(preset.id, payload)
        result = { ...preset, ...payload, updated_at: Date.now() }
      } else {
        result = await addPreset(payload)
      }
      onSaved(result)
      onClose()
    } catch (e) {
      setSaveMsg(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
      role="presentation"
    >
      <div
        className="flex max-h-[90vh] w-full max-w-[640px] flex-col overflow-hidden rounded-2xl border border-border/60 bg-card shadow-2xl dark:border-white/10 dark:bg-[#1c1c1c]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/60 px-6 py-4 dark:border-white/10">
          <div>
            <h3 className="text-lg font-semibold text-foreground">
              {isEdit ? "编辑预设" : "新增预设"}
            </h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              配置一个供应商：名称、API Key、Base URL 及多个模型
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="grid size-8 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground dark:hover:bg-white/10"
          >
            <X className="size-4" />
          </button>
        </div>

        {/* Body */}
        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
          <div className="space-y-5">
            {/* 供应商名称 */}
            <div>
              <Label className="mb-1.5 block">
                <span className="text-sm font-medium text-foreground">供应商名称</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  用于预设列表展示（如：智谱 / OpenAI / 自建代理）
                </span>
              </Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="例如：智谱"
                autoFocus
              />
            </div>

            {/* API Key */}
            <div>
              <Label className="mb-1.5 block">
                <span className="text-sm font-medium text-foreground">API Key</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  第三方 LLM API 密钥
                </span>
              </Label>
              <div className="flex items-center gap-2">
                <Input
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  className="grid size-9 shrink-0 place-items-center rounded-md border border-input bg-transparent text-muted-foreground transition hover:text-foreground"
                >
                  {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>
            </div>

            {/* Base URL */}
            <div>
              <Label className="mb-1.5 block">
                <span className="text-sm font-medium text-foreground">Base URL</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  OpenAI 兼容 API 的基础 URL
                </span>
              </Label>
              <Input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.iamhc.cn/v1"
              />
            </div>

            {/* 模型列表 */}
            <div>
              <Label className="mb-1.5 block">
                <span className="text-sm font-medium text-foreground">模型列表</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  该供应商下可用的模型，至少添加 1 个
                </span>
              </Label>

              {/* 已添加模型 */}
              {models.length > 0 && (
                <div className="mb-2 flex flex-wrap gap-2">
                  {models.map((m) => (
                    <span
                      key={m}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-muted/40 px-2.5 py-1 text-xs dark:border-white/10 dark:bg-white/[0.05]"
                    >
                      <span className="font-mono text-foreground/90">{m}</span>
                      <button
                        type="button"
                        onClick={() => removeModel(m)}
                        className="grid size-4 place-items-center rounded text-muted-foreground/60 transition hover:bg-neg/10 hover:text-neg"
                        title="移除"
                      >
                        <X className="size-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}

              {/* 添加模型输入 */}
              <div className="flex items-center gap-2">
                <Input
                  ref={modelInputRef}
                  value={newModel}
                  onChange={(e) => setNewModel(e.target.value)}
                  placeholder="模型名，如 glm-5.2"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault()
                      addModel()
                    }
                  }}
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={addModel}
                  disabled={!newModel.trim()}
                  className="shrink-0 gap-1.5"
                >
                  <Plus className="size-4" />
                  添加
                </Button>
              </div>
            </div>

            {/* 角色分配：为 fast/smart/coder 分别指定模型 */}
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <Label className="block">
                  <span className="text-sm font-medium text-foreground">角色分配（可选）</span>
                  <span className="mt-0.5 block text-xs text-muted-foreground">
                    为 fast/smart/coder 三个角色分别指定模型；未分配则切换时统一用第一个模型
                  </span>
                </Label>
                {models.length >= 1 && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={autoAssignRoles}
                    className="shrink-0 text-xs"
                  >
                    自动分配
                  </Button>
                )}
              </div>
              <div className="space-y-2">
                {LLM_ROLES.map((role) => (
                  <div key={role.key} className="flex items-center gap-3">
                    <div className="w-16 shrink-0">
                      <span className="font-mono text-xs font-medium text-foreground">
                        {role.label}
                      </span>
                    </div>
                    <select
                      value={roleMapping[role.key] ?? ""}
                      onChange={(e) => setRole(role.key, e.target.value)}
                      disabled={models.length === 0}
                      className="h-9 flex-1 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50"
                    >
                      <option value="">未分配</option>
                      {models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                    <span className="hidden shrink-0 text-xs text-muted-foreground sm:block">
                      {role.desc}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* 测试连接 */}
            <div className="rounded-xl border border-border/60 bg-muted/20 px-4 py-4 dark:border-white/10 dark:bg-white/[0.02]">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-foreground">测试连接</p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    选择一个模型发 ping 调用，验证 API 可达性与密钥有效性
                  </p>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <div className="flex min-w-[200px] flex-1 items-center gap-2">
                  <select
                    value={testModel}
                    onChange={(e) => setTestModel(e.target.value)}
                    disabled={models.length === 0 || testing}
                    className="h-9 flex-1 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none transition focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50"
                  >
                    {models.length === 0 && <option value="">请先添加模型</option>}
                    {models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleTest}
                  disabled={testing || !apiKey.trim() || !testModel || !daemonAvailable}
                  className="gap-1.5"
                >
                  {testing ? (
                    <RefreshCw className="size-4 animate-spin" />
                  ) : (
                    <Activity className="size-4" />
                  )}
                  {testing ? "测试中..." : "测试"}
                </Button>
              </div>

              {!daemonAvailable && (
                <p className="mt-2 text-xs text-neg">
                  daemon 未启动，无法测试（测试请求需通过 daemon 转发）
                </p>
              )}

              {/* 测试结果 */}
              {testResult && (
                <div
                  className={cn(
                    "mt-3 rounded-lg border px-3 py-2.5",
                    testResult.status === "ok"
                      ? "border-pos/30 bg-pos/5 dark:border-pos/20 dark:bg-pos/[0.08]"
                      : "border-neg/30 bg-neg/5 dark:border-neg/20 dark:bg-neg/[0.08]",
                  )}
                >
                  <div className="flex items-center gap-2">
                    {testResult.status === "ok" ? (
                      <Check className="size-4 text-pos" />
                    ) : (
                      <Info className="size-4 text-neg" />
                    )}
                    <span
                      className={cn(
                        "text-sm font-medium",
                        testResult.status === "ok" ? "text-pos" : "text-neg",
                      )}
                    >
                      {testResult.status === "ok" ? "连接成功" : "连接失败"}
                    </span>
                    {testResult.status === "ok" && testResult.elapsed_secs != null && (
                      <span className="text-xs text-muted-foreground">
                        耗时 {testResult.elapsed_secs}s
                        {testResult.model ? ` · ${testResult.model}` : ""}
                      </span>
                    )}
                  </div>
                  <p className="mt-1.5 text-sm text-foreground">
                    {testResult.status === "ok"
                      ? `LLM 回复: ${testResult.result ?? ""}`
                      : testResult.error}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between gap-3 border-t border-border/60 px-6 py-4 dark:border-white/10">
          <div className="text-xs text-muted-foreground">
            {saveMsg && <span className="text-neg">{saveMsg}</span>}
            {!saveMsg && (
              <span>
                {isEdit ? "修改后保存将覆盖原预设" : "保存后可在列表中点击模型一键切换"}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>
              取消
            </Button>
            <Button
              type="button"
              onClick={handleSave}
              disabled={!canSave || saving}
              className="gap-1.5"
            >
              {saving ? (
                <RefreshCw className="size-4 animate-spin" />
              ) : (
                <Check className="size-4" />
              )}
              {saving ? "保存中..." : isEdit ? "保存修改" : "保存预设"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
