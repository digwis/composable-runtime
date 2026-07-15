import { app, BrowserWindow, ipcMain, shell } from "electron"
import { readFile, writeFile, mkdir } from "node:fs/promises"
import { existsSync } from "node:fs"
import { spawn } from "node:child_process"
import path from "node:path"
import { fileURLToPath } from "node:url"

// Keep a small amount of lifecycle information outside the dev server output.
// This makes unexpected exits diagnosable after the window has disappeared.
const mainLogPath = path.join(app.getPath("logs"), "evolution-dashboard-main.log")
function logMain(message: string, error?: unknown) {
  const line = `[${new Date().toISOString()}] ${message}${error ? ` ${String(error)}` : ""}\n`
  void writeFile(mainLogPath, line, { flag: "a" }).catch(() => {})
  console.error(line.trim())
}

process.on("uncaughtException", (error) => logMain("uncaughtException", error))
process.on("unhandledRejection", (reason) => logMain("unhandledRejection", reason))

const gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!gotSingleInstanceLock) {
  logMain("second instance refused")
  app.quit()
}
app.on("second-instance", () => {
  if (!mainWindow) return
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.show()
  mainWindow.focus()
})

app.commandLine.appendSwitch("remote-debugging-port","9225")
app.commandLine.appendSwitch("remote-allow-origins","*")

const isDev = !!process.env.ELECTRON_RENDERER_URL
const currentDir = path.dirname(fileURLToPath(import.meta.url))

let mainWindow: BrowserWindow | null = null
let daemonStartup: Promise<void> | null = null

function createWindow() {
mainWindow = new BrowserWindow({
  width: 1440,
  height: 900,
  minWidth: 960,
  minHeight: 640,
  title: "",
  show: false,
  autoHideMenuBar: true,
  titleBarStyle: "hiddenInset",
  webPreferences: {
    preload: path.join(currentDir, "../preload/index.mjs"),
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: false,
  },
})

  mainWindow.on("ready-to-show", () => {
    mainWindow?.maximize()
    mainWindow?.show()
  })

  mainWindow.on("closed", () => {
    logMain("main window closed")
    mainWindow = null
  })
  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    logMain(`render process gone: ${details.reason} (exit ${details.exitCode})`)
  })
  mainWindow.webContents.on("unresponsive", () => logMain("renderer became unresponsive"))
  mainWindow.webContents.on("responsive", () => logMain("renderer became responsive"))
  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    if (level >= 2) logMain(`renderer console ${sourceId}:${line}`, message)
  })

  // 外部链接用系统浏览器打开
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: "deny" }
  })

  if (isDev) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL!)
  } else {
    mainWindow.loadFile(path.join(currentDir, "../renderer/index.html"))
  }
}

// daemon HTTP API 调用（默认 127.0.0.1:7331）
const DAEMON_PORT = Number(process.env.ORCH_HTTP_PORT) || 7331
const DAEMON_BASE = `http://127.0.0.1:${DAEMON_PORT}`

async function ensureDaemon(): Promise<void> {
  if (daemonStartup) return daemonStartup
  daemonStartup = (async () => {
    const probe = async (): Promise<number | null> => {
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 2500)
        const resp = await fetch(`${DAEMON_BASE}/api/status`, { signal: controller.signal })
        clearTimeout(timeout)
        return resp.status
      } catch {
        return null
      }
    }

    const initialStatus = await probe()
    if (initialStatus !== null) {
      // Any HTTP response, including 503, means the daemon is alive. A 503
      // only indicates that its evolution lock is busy.
      logMain(`daemon health check: HTTP ${initialStatus}`)
      return
    }

    // Give an already-starting daemon a chance to bind before creating another
    // process. This closes the race seen when the UI is restarted repeatedly.
    await new Promise((resolve) => setTimeout(resolve, 800))
    const retryStatus = await probe()
    if (retryStatus !== null) {
      logMain(`daemon became available during startup grace period: HTTP ${retryStatus}`)
      return
    }

    logMain("daemon health check failed; attempting startup")

    const candidates = [
    process.env.ORCH_BIN,
    path.resolve(currentDir, "../../../target/debug/orch"),
    path.resolve(currentDir, "../../../target/release/orch"),
  ].filter((value): value is string => !!value)
    const binary = candidates.find((value) => existsSync(value))
    if (!binary) {
      logMain("daemon binary not found")
      return
    }

    const home = app.getPath("home")
    const projectRoots = process.env.ORCH_PROJECT_ROOTS || [
    path.join(home, "项目", "Electron"),
    path.join(home, "项目"),
    path.join(home, "Projects"),
  ].join(":")
    const child = spawn(binary, [
    "daemon",
    "--model", process.env.ORCH_MODEL || "glm-5.2",
    "--base-url", process.env.ORCH_BASE_URL || "https://api.iamhc.cn/v1",
    "--interval", process.env.ORCH_DAEMON_INTERVAL || "300",
    "--max-rounds", "0",
  ], {
    detached: true,
    stdio: "ignore",
    env: { ...process.env, ORCH_PROJECT_ROOTS: projectRoots },
  })
    child.on("error", (error) => logMain("daemon spawn failed", error))
    child.on("exit", (code, signal) => logMain(`daemon exited: code=${code} signal=${signal}`))
    child.unref()
    logMain(`daemon started: ${binary}`)
  })().finally(() => {
    daemonStartup = null
  })
  return daemonStartup
}

async function fetchDaemon(pathname: string, init?: RequestInit): Promise<{ ok: boolean; status: number; data: unknown }> {
  try {
    const resp = await fetch(`${DAEMON_BASE}${pathname}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    })
    const text = await resp.text()
    let data: unknown = text
    try {
      data = JSON.parse(text)
    } catch {
      // 非 JSON（如 503 纯文本）
    }
    return { ok: resp.ok, status: resp.status, data }
  } catch (e) {
    return { ok: false, status: 0, data: { error: String(e), hint: "daemon 可能未启动" } }
  }
}

type PiSyncRequest = {
  apiKey: string
  baseUrl: string
  model: string
  models: string[]
}

type ApiPresetRecord = Record<string, unknown>

async function readPresetStore(): Promise<ApiPresetRecord[]> {
  const file = path.join(app.getPath("userData"), "api-presets.json")
  try {
    const value = JSON.parse(await readFile(file, "utf8"))
    return Array.isArray(value) ? value : []
  } catch { return [] }
}

async function writePresetStore(presets: ApiPresetRecord[]): Promise<void> {
  const dir = app.getPath("userData")
  await mkdir(dir, { recursive: true })
  await writeFile(path.join(dir, "api-presets.json"), `${JSON.stringify(presets, null, 2)}\n`, { mode: 0o600 })
}

async function syncPiConfig(req: PiSyncRequest): Promise<{ ok: boolean; message: string }> {
  if (!req.apiKey.trim() || !req.baseUrl.trim() || !req.model.trim()) {
    return { ok: false, message: "pi 同步需要 api key、base URL 和模型" }
  }
  const piDir = path.join(app.getPath("home"), ".pi", "agent")
  const modelsPath = path.join(piDir, "models.json")
  const settingsPath = path.join(piDir, "settings.json")
  await mkdir(piDir, { recursive: true })

  let modelsConfig: Record<string, any> = { providers: {} }
  try { modelsConfig = JSON.parse(await readFile(modelsPath, "utf8")) } catch {}
  if (!modelsConfig.providers || typeof modelsConfig.providers !== "object") modelsConfig.providers = {}
  const provider = "evolution-preset"
  modelsConfig.providers[provider] = {
    ...(modelsConfig.providers[provider] || {}),
    baseUrl: req.baseUrl,
    api: "openai-completions",
    apiKey: req.apiKey,
    authHeader: true,
    models: Array.from(new Set([req.model, ...req.models])).map((id) => ({
      id, name: id, input: ["text"], contextWindow: 128000, maxTokens: 16384,
    })),
  }
  await writeFile(modelsPath, `${JSON.stringify(modelsConfig, null, 2)}\n`, { mode: 0o600 })

  let settings: Record<string, any> = {}
  try { settings = JSON.parse(await readFile(settingsPath, "utf8")) } catch {}
  settings.defaultProvider = provider
  settings.defaultModel = req.model
  await writeFile(settingsPath, `${JSON.stringify(settings, null, 2)}\n`, { mode: 0o600 })
  return { ok: true, message: `pi 已同步到 ${provider}/${req.model}` }
}

function registerIpc() {
  ipcMain.handle("daemon:status", () => fetchDaemon("/api/status"))
  ipcMain.handle("daemon:capabilities", () => fetchDaemon("/api/capabilities"))
  ipcMain.handle("daemon:config", () => fetchDaemon("/api/config"))
  ipcMain.handle("daemon:integrations-status", () => fetchDaemon("/api/integrations/status"))
  ipcMain.handle("daemon:integrations-bootstrap", () =>
    fetchDaemon("/api/integrations/bootstrap", { method: "POST" }),
  )
  ipcMain.handle("daemon:config-update", (_e, body) =>
    fetchDaemon("/api/config", { method: "POST", body: JSON.stringify(body) }),
  )
  ipcMain.handle("daemon:config-reset", () =>
    fetchDaemon("/api/config", { method: "POST", body: JSON.stringify({ reset: true }) }),
  )
  ipcMain.handle("daemon:test-llm", (_e, body) =>
    fetchDaemon("/api/test_llm", { method: "POST", body: JSON.stringify(body) }),
  )
  ipcMain.handle("daemon:feedback", (_e, body) =>
    fetchDaemon("/api/feedback", { method: "POST", body: JSON.stringify(body) }),
  )
  ipcMain.handle("daemon:exec", (_e, body) =>
    fetchDaemon("/api/exec", { method: "POST", body: JSON.stringify(body) }),
  )
  ipcMain.handle("daemon:evolution", () => fetchDaemon("/api/evolution"))
  ipcMain.handle("daemon:llm-health", () => fetchDaemon("/api/llm_health"))
  ipcMain.handle("daemon:research", (_e, body: { urls?: string[]; query?: string; max_sources?: number; force_refresh?: boolean }) =>
    fetchDaemon("/api/research", { method: "POST", body: JSON.stringify(body) }))
  ipcMain.handle("daemon:workspace-graph", () => fetchDaemon("/api/workspace/graph"))
  ipcMain.handle("daemon:autonomy-status", () => fetchDaemon("/api/autonomy/status"))
  ipcMain.handle("daemon:autonomy-decisions", () => fetchDaemon("/api/autonomy/decisions"))
  ipcMain.handle("daemon:autonomy-prompts", () => fetchDaemon("/api/autonomy/prompts"))
  ipcMain.handle("daemon:learning-agenda", () => fetchDaemon("/api/learning-agenda"))
  ipcMain.handle("daemon:autonomy-approve", (_e, id: string) => fetchDaemon(`/api/autonomy/prompts/${encodeURIComponent(id)}/approve`, { method: "POST" }))
  ipcMain.handle("daemon:autonomy-reject", (_e, id: string) => fetchDaemon(`/api/autonomy/prompts/${encodeURIComponent(id)}/reject`, { method: "POST" }))
  ipcMain.handle("daemon:autonomy-dismiss", (_e, id: string) => fetchDaemon(`/api/autonomy/prompts/${encodeURIComponent(id)}/dismiss`, { method: "POST" }))
  ipcMain.handle("daemon:autonomy-pause", () => fetchDaemon("/api/autonomy/pause", { method: "POST" }))
  ipcMain.handle("daemon:autonomy-resume", () => fetchDaemon("/api/autonomy/resume", { method: "POST" }))
  ipcMain.handle("daemon:explore-project", (_e, body: { project_path: string; objective: string; max_variants?: number }) => fetchDaemon("/api/explorer", { method: "POST", body: JSON.stringify(body) }))
  ipcMain.handle("daemon:experiment-batches", () => fetchDaemon("/api/experiments"))
  ipcMain.handle("daemon:run-experiments", (_e, body: unknown) => fetchDaemon("/api/experiments", { method: "POST", body: JSON.stringify(body) }))
  ipcMain.handle("pi:sync-config", (_e, body: PiSyncRequest) => syncPiConfig(body))
  ipcMain.handle("presets:load", () => readPresetStore())
  ipcMain.handle("presets:save", (_e, presets: ApiPresetRecord[]) => writePresetStore(presets).then(() => true))
  ipcMain.handle("daemon:port", () => DAEMON_PORT)
  ipcMain.handle("daemon:projects", () => fetchDaemon("/api/projects"))
  ipcMain.handle("daemon:project-memory", (_e, body: { project_path: string; vision: string; priorities: string[] }) =>
    fetchDaemon("/api/projects/memory", { method: "POST", body: JSON.stringify(body) }))
  ipcMain.handle("daemon:project-execute", (_e, body) => fetchDaemon("/api/projects/execute", { method: "POST", body: JSON.stringify(body) }))
  ipcMain.handle("daemon:project-proposal-feedback", (_e, body: { id: string; project_path: string; title: string; task: string; category?: string; useful: boolean }) =>
    fetchDaemon(`/api/projects/proposals/${encodeURIComponent(body.id)}/feedback`, { method: "POST", body: JSON.stringify(body) }))
  ipcMain.handle("daemon:project-tasks", () => fetchDaemon("/api/projects/tasks"))
  ipcMain.handle("daemon:project-runs", () => fetchDaemon("/api/projects/runs"))
  ipcMain.handle("daemon:worker-pool-status", () => fetchDaemon("/api/workers/status"))
  ipcMain.handle("daemon:project-run-events", (_e, id: string) =>
    fetchDaemon(`/api/projects/runs/${encodeURIComponent(id)}/events`),
  )
  ipcMain.handle("daemon:project-run-retry", (_e, id: string) =>
    fetchDaemon(`/api/projects/runs/${encodeURIComponent(id)}/retry`, { method: "POST" }),
  )
  ipcMain.handle("daemon:project-task-feedback", (_e, body: { id: string; useful: boolean; note?: string }) =>
    fetchDaemon(`/api/projects/tasks/${encodeURIComponent(body.id)}/feedback`, {
      method: "POST",
      body: JSON.stringify({ useful: body.useful, note: body.note || "" }),
    }),
  )
  ipcMain.handle("daemon:project-task-outcome", (_e, body: { id: string; horizon_days: 7 | 30; status: "adopted" | "still_using" | "rolled_back"; note?: string }) =>
    fetchDaemon(`/api/projects/tasks/${encodeURIComponent(body.id)}/outcome`, {
      method: "POST",
      body: JSON.stringify({ horizon_days: body.horizon_days, status: body.status, note: body.note || "" }),
    }),
  )
}

app.whenReady().then(() => {
  registerIpc()
  void ensureDaemon()
  createWindow()
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})
