// API 配置预设 — 由 Electron 主进程统一持久化，兼容迁移旧 localStorage。
// daemon 本身只保存一份"覆盖层"，预设是对其的封装：切换预设 = 把预设写入 daemon。
// 一个预设对应一个供应商：名称 + api_key + base_url + 多个模型。

const STORAGE_KEY = "evolution-dashboard:api-presets"
const ACTIVE_KEY = "evolution-dashboard:active-preset"

// 角色到模型的映射 — 可选，若三个角色都分配则切换时分别写入 daemon
export type RoleMapping = {
  fast?: string
  smart?: string
  coder?: string
}

export type LlmRole = "fast" | "smart" | "coder"

export const LLM_ROLES: Array<{ key: LlmRole; label: string; desc: string }> = [
  { key: "fast", label: "fast", desc: "测试输入/简单任务（便宜快速）" },
  { key: "smart", label: "smart", desc: "归因分析/目标生成（强推理）" },
  { key: "coder", label: "coder", desc: "代码生成/变异（代码能力强）" },
]

export interface ApiPreset {
  id: string
  // 供应商名称（如：智谱 / OpenAI / 自建代理）
  name: string
  api_key: string
  base_url: string
  // 该供应商下可用的模型列表（至少 1 个）
  models: string[]
  // 可选：为 fast/smart/coder 三个角色分别指定模型
  // 若完整指定三个角色，切换时分别写入；否则用 models[0] 作为通用 model
  role_mapping?: RoleMapping
  created_at: number
  updated_at: number
}

// 判断 role_mapping 是否完整（三个角色都有值且在 models 列表中）
export function isRoleMappingComplete(preset: ApiPreset): boolean {
  const rm = preset.role_mapping
  if (!rm) return false
  return (
    !!rm.fast && !!rm.smart && !!rm.coder &&
    preset.models.includes(rm.fast) &&
    preset.models.includes(rm.smart) &&
    preset.models.includes(rm.coder)
  )
}

function genId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

function normalize(raw: unknown): ApiPreset[] {
  try {
    const arr = (Array.isArray(raw) ? raw : []) as ApiPreset[]
    if (!Array.isArray(arr)) return []
    return arr
      .filter((p) => p && typeof p.name === "string")
      .map((p) => {
        // 兼容旧版单 model 字段
        const models = Array.isArray(p.models)
          ? p.models
          : typeof (p as unknown as { model?: string }).model === "string" &&
              (p as unknown as { model?: string }).model
            ? [(p as unknown as { model?: string }).model!]
            : []
        return { ...p, models }
      })
  } catch {
    return []
  }
}

export async function loadPresets(): Promise<ApiPreset[]> {
  const central = normalize(await window.daemon.loadPresets())
  let localRaw: unknown = []
  try { localRaw = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]") } catch {}
  const local = normalize(localRaw)
  const merged = [...central]
  for (const item of local) if (!merged.some((p) => p.id === item.id || (p.name === item.name && p.base_url === item.base_url))) merged.push(item)
  if (merged.length !== central.length) await window.daemon.savePresets(merged)
  return merged
}

export async function savePresets(list: ApiPreset[]): Promise<void> {
  await window.daemon.savePresets(list)
}

export async function addPreset(input: Omit<ApiPreset, "id" | "created_at" | "updated_at">): Promise<ApiPreset> {
  const list = await loadPresets()
  const now = Date.now()
  const preset: ApiPreset = {
    id: genId(),
    created_at: now,
    updated_at: now,
    ...input,
  }
  list.push(preset)
  await savePresets(list)
  return preset
}

export async function updatePreset(id: string, patch: Partial<Omit<ApiPreset, "id" | "created_at">>): Promise<void> {
  const list = (await loadPresets()).map((p) =>
    p.id === id ? { ...p, ...patch, updated_at: Date.now() } : p,
  )
  await savePresets(list)
}

export async function removePreset(id: string): Promise<void> {
  const list = (await loadPresets()).filter((p) => p.id !== id)
  await savePresets(list)
  if ((await getActivePresetId()) === id) await setActivePresetId(null)
}

export async function getActivePresetId(): Promise<string | null> {
  try { return window.localStorage.getItem(ACTIVE_KEY) || null } catch { return null }
}

export async function setActivePresetId(id: string | null): Promise<void> {
  if (id) window.localStorage.setItem(ACTIVE_KEY, id)
  else window.localStorage.removeItem(ACTIVE_KEY)
}

// 根据 daemon 当前生效配置匹配预设
// 优先匹配 role_mapping 完整的预设（三个角色模型都对得上），其次匹配 models 包含当前 model
export function findMatchingPreset(
  presets: ApiPreset[],
  current: { base_url: string; model: string; has_key: boolean } | null,
): { preset: ApiPreset; model: string } | null {
  if (!current) return null
  // 先尝试 role_mapping 完整匹配（最精确）
  for (const p of presets) {
    if (p.base_url !== current.base_url) continue
    if (!!p.api_key !== current.has_key) continue
    if (isRoleMappingComplete(p) && p.role_mapping!.fast === current.model) {
      return { preset: p, model: current.model }
    }
  }
  // 退而求其次：models 包含当前 model
  for (const p of presets) {
    if (p.base_url !== current.base_url) continue
    if (!!p.api_key !== current.has_key) continue
    if (p.models.includes(current.model)) {
      return { preset: p, model: current.model }
    }
  }
  return null
}
