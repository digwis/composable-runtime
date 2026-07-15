import { contextBridge, ipcRenderer } from "electron"

export interface DaemonResponse {
  ok: boolean
  status: number
  data: unknown
}

const api = {
  status: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:status"),
  capabilities: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:capabilities"),
  config: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:config"),
  integrationsStatus: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:integrations-status"),
  bootstrapIntegrations: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:integrations-bootstrap"),
  updateConfig: (body: Record<string, unknown>): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:config-update", body),
  resetConfig: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:config-reset"),
  testLlm: (body: Record<string, unknown>): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:test-llm", body),
  feedback: (body: { capability: string; useful: boolean; note?: string }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:feedback", body),
  exec: (body: { capability: string; action: string; input?: unknown }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:exec", body),
  evolution: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:evolution"),
  llmHealth: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:llm-health"),
  research: (body: { urls?: string[]; query?: string; max_sources?: number; force_refresh?: boolean }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:research", body),
  workspaceGraph: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:workspace-graph"),
  autonomyStatus: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-status"),
  autonomyDecisions: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-decisions"),
  autonomyPrompts: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-prompts"),
  learningAgenda: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:learning-agenda"),
  approveAutonomyPrompt: (id: string): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-approve", id),
  rejectAutonomyPrompt: (id: string): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-reject", id),
  dismissAutonomyPrompt: (id: string): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-dismiss", id),
  pauseAutonomy: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-pause"),
  resumeAutonomy: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:autonomy-resume"),
  exploreProject: (body: { project_path: string; objective: string; max_variants?: number }): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:explore-project", body),
  experimentBatches: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:experiment-batches"),
  runExperiments: (body: { project_path: string; objective: string; variants: Array<{ id: string; title: string; task: string }>; verify_command?: string; benchmark_command?: string }): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:run-experiments", body),
  syncPiConfig: (body: { apiKey: string; baseUrl: string; model: string; models: string[] }) =>
    ipcRenderer.invoke("pi:sync-config", body) as Promise<{ ok: boolean; message: string }>,
  port: (): Promise<number> => ipcRenderer.invoke("daemon:port"),
  projects: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:projects"),
  updateProjectMemory: (body: { project_path: string; vision: string; priorities: string[] }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:project-memory", body),
  executeProject: (body: { project_path: string; task: string; proposal_id?: string; verify_command?: string }) =>
    ipcRenderer.invoke("daemon:project-execute", body) as Promise<DaemonResponse>,
  projectProposalFeedback: (id: string, body: { project_path: string; title: string; task: string; category?: string; useful: boolean }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:project-proposal-feedback", { id, ...body }),
  projectTasks: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:project-tasks"),
  projectRuns: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:project-runs"),
  workerPoolStatus: (): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:worker-pool-status"),
  projectRunEvents: (id: string): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:project-run-events", id),
  retryProjectRun: (id: string): Promise<DaemonResponse> => ipcRenderer.invoke("daemon:project-run-retry", id),
  projectTaskFeedback: (id: string, body: { useful: boolean; note?: string }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:project-task-feedback", { id, ...body }),
  projectTaskOutcome: (id: string, body: { horizon_days: 7 | 30; status: "adopted" | "still_using" | "rolled_back"; note?: string }): Promise<DaemonResponse> =>
    ipcRenderer.invoke("daemon:project-task-outcome", { id, ...body }),
  loadPresets: (): Promise<unknown[]> => ipcRenderer.invoke("presets:load"),
  savePresets: (presets: unknown[]): Promise<boolean> => ipcRenderer.invoke("presets:save", presets),
}

contextBridge.exposeInMainWorld("daemon", api)

export type DaemonApi = typeof api
