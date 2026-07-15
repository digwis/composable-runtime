import { Type } from '@earendil-works/pi-ai';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { appendFile } from 'node:fs/promises';

type AllowedCapability = {
  capability: string;
  action: string;
  input: Record<string, unknown>;
};

type RuntimeResponse = {
  success?: boolean;
  payload?: Record<string, unknown>;
  error?: string;
};

function allowedCapabilities(): AllowedCapability[] {
  try {
    const value = JSON.parse(process.env.ORCH_PI_CAPABILITIES || '[]');
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function scopedInput(defaults: Record<string, unknown>, supplied: unknown): Record<string, unknown> {
  const input = supplied && typeof supplied === 'object' && !Array.isArray(supplied)
    ? supplied as Record<string, unknown>
    : {};
  const merged = { ...defaults, ...input };
  const worktree = process.env.ORCH_PI_WORKTREE || '';
  if (!worktree) return merged;
  for (const key of ['path', 'cwd', 'repo_path', 'source_dir']) {
    if (key in merged) merged[key] = worktree;
  }
  if ('build_dir' in merged) merged.build_dir = worktree + '/build';
  return merged;
}

async function recordTrace(value: Record<string, unknown>): Promise<void> {
  const path = process.env.ORCH_PI_CAPABILITY_TRACE;
  if (!path) return;
  try {
    await appendFile(path, JSON.stringify(value) + '\n', 'utf8');
  } catch {
    // Trace persistence must not turn a successful capability call into a task failure.
  }
}

export default function runtimeCapabilityExtension(pi: ExtensionAPI) {
  pi.registerTool({
    name: 'runtime_capabilities',
    label: 'Runtime Capabilities',
    description: 'List evolved capabilities approved by the runtime for this project task.',
    promptSnippet: 'List reusable runtime capabilities approved for the current project task',
    promptGuidelines: [
      'Use runtime_capabilities before runtime_capability_call when reusable project evidence would reduce repeated analysis.',
    ],
    parameters: Type.Object({}),
    async execute() {
      const capabilities = allowedCapabilities();
      return {
        content: [{ type: 'text', text: JSON.stringify(capabilities, null, 2) }],
        details: { capabilities },
      };
    },
  });

  pi.registerTool({
    name: 'runtime_capability_call',
    label: 'Runtime Capability',
    description: 'Execute one approved evolved capability through the local Capability Runtime.',
    promptSnippet: 'Run an approved evolved capability in the current isolated worktree',
    promptGuidelines: [
      'Use runtime_capability_call only for capability and action pairs returned by runtime_capabilities.',
      'Treat a failed runtime_capability_call as evidence of risk and never report it as successful.',
    ],
    parameters: Type.Object({
      capability: Type.String({ description: 'Approved capability name' }),
      action: Type.String({ description: 'Approved action name' }),
      input: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
    }),
    async execute(_toolCallId, params, signal, onUpdate) {
      const started = Date.now();
      const allowed = allowedCapabilities();
      const selected = allowed.find(
        item => item.capability === params.capability && item.action === params.action,
      );
      if (!selected) {
        const message = 'Capability action is not approved for this project task.';
        await recordTrace({
          capability: params.capability,
          action: params.action,
          phase: 'agent_dynamic',
          input: params.input || {},
          output_summary: message,
          success: false,
          elapsed_ms: Date.now() - started,
        });
        return { content: [{ type: 'text', text: message }], details: { success: false } };
      }

      onUpdate?.({ content: [{ type: 'text', text: 'Running evolved capability...' }] });
      const input = scopedInput(selected.input || {}, params.input);
      const base = process.env.ORCH_DAEMON_URL || 'http://127.0.0.1:7331';
      try {
        const response = await fetch(base + '/api/exec', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ capability: params.capability, action: params.action, input }),
          signal,
        });
        const body = await response.json() as RuntimeResponse;
        const payload = body.payload || body;
        const success = response.ok && body.success === true && payload.success !== false;
        const summary = JSON.stringify(payload).slice(0, 1200);
        await recordTrace({
          capability: params.capability,
          action: params.action,
          phase: 'agent_dynamic',
          input,
          output_summary: summary,
          success,
          elapsed_ms: Date.now() - started,
        });
        return {
          content: [{ type: 'text', text: JSON.stringify({ success, payload }, null, 2) }],
          details: { success, payload },
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await recordTrace({
          capability: params.capability,
          action: params.action,
          phase: 'agent_dynamic',
          input,
          output_summary: message,
          success: false,
          elapsed_ms: Date.now() - started,
        });
        return {
          content: [{ type: 'text', text: 'Runtime capability failed: ' + message }],
          details: { success: false, error: message },
        };
      }
    },
  });
}
