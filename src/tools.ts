/**
 * Model-facing tools for dsh-comfyui, registered into the host `tools`
 * registry. `comfyui_run` submits a workflow and returns media results
 * (synchronously or as a background job); `comfyui_object_info` exposes the
 * server's node definitions; `comfyui_workflow` lists and runs saved
 * workflows from the panel-managed workflow library.
 */
import type { Context } from '@deepseek-ai/cordis'
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve as resolvePath, join } from 'node:path'
import type { Config } from './config.js'
import { ComfyUIClient, ComfyUIError, collectMedia, type ComfyUIHistoryEntry } from './comfyui.js'
import { TEMPLATES, findTemplate, cloneWorkflow, applyTemplateInputs } from './templates.js'
import type { AssetRecord, LoadSlot, StoredWorkflow } from './store.js'
import type { GraphAnalysis } from './analyze.js'
import type { RunProgress } from './progress.js'
import type { QueuedRun } from './queue.js'
import { refreshParameterMetadata, type Workflow, type WorkflowParameter } from './params.js'
import type { HostHint } from './host-hint.js'
import { SKILL_MAIN, joinFrontmatter, type WorkflowSkillPacks } from './skillpack.js'
import {
  formatRunLabel,
  nextJobNumber,
  resolveRunPrefix,
  safeRelativePath,
  uniqueOutputPath,
  upsertRun,
  type LedgerRecord,
} from './ledger.js'
import {
  extractNodeErrors,
  preflightFailure,
  preflightRefusalFailure,
  preflightUnavailable,
  preflightWorkflow,
  undeclaredArgumentKeys,
  type RunFailure,
} from './preflight.js'

/** A workflow saved on the ComfyUI server (userdata/workflows), with extract status. */
export interface ComfyUIComfyWorkflow {
  name: string
  size?: number
  modified?: number
  /** Whether at least one runnable API workflow was extracted from this graph. */
  extracted: boolean
  /** Runnable API workflows extracted from this graph (运行主题). */
  derived: Array<{ libraryId: string; name: string }>
}

/** Live runtime the tools, routes, and proxy share. */
export interface ComfyUIRuntime {
  getConfig(): Config
  /** Resolve the API key per request (credentials store, then environment). */
  getApiKey(): Promise<string | undefined>
  createClient(apiKey: string | undefined): ComfyUIClient
  /** Absolute media proxy URL base: explicit config > detected request host > loopback. */
  proxyBase(): string | undefined
  /** Directory the run ledger (`runs.json`) and fetched media land in:
   * explicit config, else the plugin data directory. */
  downloadDir(): string
  /** Remembers the origin browsers use to reach this server (Host header). */
  hostHint: HostHint
  /** Whether the settings service can persist config writes. */
  settingsWritable(): boolean
  updateConfig(patch: Record<string, unknown>): Promise<{ ok: true } | { ok: false; error: string }>
  /** Queue a workflow and track it in the queue tracker. `meta.parameters`
   * applies adjustable parameters (values/random seeds) before submitting. */
  queue(workflow: unknown, meta: {
    workflowName: string | null
    workflowId?: string | null
    source: string
    parameters?: WorkflowParameter[]
    values?: Record<string, unknown>
  }): Promise<string>
  /** Stop tracking a prompt (used when a tool call fails before completion). */
  untrack(promptId: string): void
  /** Every prompt this plugin queued and is still waiting on. */
  trackedRuns(): QueuedRun[]
  /** Live generation progress for one prompt (from the ComfyUI WebSocket). */
  queueProgress(promptId: string): RunProgress | undefined
  /** Saved workflows from the library. */
  listWorkflows(): Promise<StoredWorkflow[]>
  getWorkflow(id: string): Promise<StoredWorkflow | undefined>
  /** Create or update a workflow in the library. */
  saveWorkflow(input: {
    id?: string
    name: string
    description: string
    workflow: unknown
    parameters?: WorkflowParameter[]
    tags?: string[]
    source?: 'user' | 'comfyui'
    comfyuiFile?: string
  }): Promise<{ ok: true; workflow: StoredWorkflow } | { ok: false; error: string }>
  /** Delete a workflow from the library; false when it did not exist. */
  deleteWorkflow(id: string): Promise<boolean>
  /** Per-workflow skill packs (SKILL.md bundles under `<dataDir>/skills/`). */
  skillPacks: WorkflowSkillPacks
  /** Force TTS-Audio-Suite to rescan its voice library (best-effort; false
   * when the server has no such endpoint). Call before re-deriving parameter
   * snapshots so object_info reports newly added voices. */
  refreshVoiceLibrary(): Promise<boolean>
  /** Pixel sizes of panel-uploaded files, keyed by file name. */
  listMediaSizes(): Promise<Record<string, { width: number; height: number }>>
  /** Record the pixel size of one uploaded file. */
  saveMediaSize(name: string, size: { width: number; height: number }): Promise<void>
  /** File name recorded for a content hash (dedup index), if any. */
  lookupMediaHash(hash: string): Promise<string | undefined>
  /** Record a content hash → file name pair for dedup. */
  saveMediaHash(hash: string, name: string): Promise<void>
  /** The load-area slots, in order; `null` is an empty slot the user added.
   * Filled slots are the default source media for loader parameters. */
  loadSlots(): Promise<Array<LoadSlot>>
  /** Persist the load-area slots. */
  saveSlots(slots: Array<LoadSlot>): Promise<void>
  /** The asset index (newest first). */
  listAssets(): Promise<AssetRecord[]>
  /** Remove one asset record from the index, returning what was removed. */
  deleteAsset(promptId: string): Promise<AssetRecord | undefined>
  /** Move completed tracked runs into the asset index. */
  sweep(): Promise<AssetRecord[]>
  /** Workflows the user saved on the ComfyUI server, with extract status. */
  listComfyWorkflows(): Promise<ComfyUIComfyWorkflow[]>
  /** Read one ComfyUI-side saved workflow graph (UI format, not runnable as-is). */
  getComfyWorkflow(file: string): Promise<unknown>
  /** Analyze one ComfyUI-side graph: connected components, groups, dangling nodes. */
  analyzeComfyWorkflow(file: string): Promise<GraphAnalysis | { ok: false; error: string }>
  /** Extract runnable API workflows from a ComfyUI-side graph (整体/按分量/主流程). */
  extractComfyWorkflow(input: {
    file: string
    mode: 'all' | 'split' | 'main'
  }): Promise<{ ok: true; saved: StoredWorkflow[]; analysis: GraphAnalysis; warnings: string[] } | { ok: false; error: string }>
}

/** Execution identity handed to tool execute. */
interface ToolRunContext {
  agent?: unknown
  signal: AbortSignal
}

/**
 * Which skill packs each agent has already read, for the `requireSkill` gate.
 *
 * Keyed by the live `Agent` handle the tools registry passes as `exec.agent`:
 * one per session and stable across turns, so a WeakMap entry lives exactly as
 * long as the session and needs no eviction pass. An execution without an
 * agent (a direct internal call) is never gated.
 */
const skillPackReads = new WeakMap<object, Set<string>>()

/** Upper bound on the text one `comfyui_skill action: read` returns. The pack
 * caps files at 256 KB, which is still far more than a tool result should
 * inject in one go; a longer file comes back truncated with a marker. */
const MAX_TOOL_READ_CHARS = 40_000

function markSkillPackRead(agent: unknown, workflowId: string): void {
  if (typeof agent !== 'object' || agent === null) return
  const seen = skillPackReads.get(agent)
  if (seen === undefined) skillPackReads.set(agent, new Set([workflowId]))
  else seen.add(workflowId)
}

function hasReadSkillPack(agent: unknown, workflowId: string): boolean {
  if (typeof agent !== 'object' || agent === null) return true
  return skillPackReads.get(agent)?.has(workflowId) === true
}

/** One media item returned by comfyui_run (JSON-safe). */
export interface RunMediaItem {
  filename: string
  subfolder: string
  type: string
  node: string
  index: number
  kind: 'image' | 'video' | 'audio' | 'other'
  url: string
}

/** Synchronous completion result. */
export interface RunResult {
  kind: 'sync'
  promptId: string
  /** Governance key of this run (`<批次>-<lane>-<job>`); the ledger row's key. */
  runLabel: string
  status: 'completed' | 'interrupted' | 'error'
  elapsedMs: number
  media: RunMediaItem[]
  summary: string
  /** Seed actually written into the submitted workflow (never -1, never null). */
  seed: number | null
  /** Every seed-typed input of the submitted workflow, by `nodeId.inputKey`. */
  seeds: Record<string, number> | null
  /** Terminal records are mirrored into the ledger for replay. */
  ledger: string | null
  /** Present when the run was refused or failed before/while executing. */
  error: RunFailure | null
}

/** Background mode result: collect later with job_output. */
export interface BackgroundResult {
  kind: 'background'
  jobId: string
  promptId: string
  /** Governance key of this run, matching its ledger row. */
  runLabel: string
  label: string
  /** Seed written into the submitted graph (already resolved, not deferred). */
  seed: number | null
  /** Directory holding the run ledger. */
  ledger: string
}

/** A minimal ToolDefinition for ctx.tools.register. */
interface ToolDefinition {
  name: string
  description: string
  parameters: Record<string, unknown>
  output: {
    schema: Record<string, unknown>
    render(args: unknown, value: unknown): unknown[]
    presentationMeta?(args: unknown, value: unknown): unknown
  }
  timeoutMs?: number
  execute(args: Record<string, unknown>, exec: ToolRunContext): Promise<unknown>
  /** The runtime this definition is wired to. Not read by the host; it exists
   * so the checks can reach the runtime (`ComfyUIRuntime.queue`, where the
   * submit-time preflight lives) instead of rebuilding it. */
  runtime?: ComfyUIRuntime
}

interface JobsService {
  start(spec: {
    kind: string
    label: string
    owner?: unknown
    run(): {
      cancel(reason?: string): void
      done: Promise<{ status: 'completed' | 'killed' | 'failed'; detail?: string; output?: string }>
      readOutput?(): string
    }
  }): string
}

const TOOL_TIMEOUT_MS = 3_600_000

/**
 * Everything one `comfyui_run` does before it may touch the server: draw the
 * governance identity, open the queued ledger row, run the early preflight and
 * resolve the seeds.
 *
 * The order matters and is the fix for the sequence race (B-2):
 *
 * 1. `resolveSeeds` is synchronous and must run first, because the row has to
 *    record the values that were submitted.
 * 2. The early preflight is the last thing allowed to await, so a refused
 *    graph is refused before any sequence number is consumed.
 * 3. `openRun` then draws the number and writes the queued row in ONE
 *    synchronous stretch — nothing awaitable may be introduced between them,
 *    or two concurrent runs read the same number and collapse into one row.
 * 4. The caller must submit immediately after this returns: any new `await`
 *    between here and the submit widens that window again.
 *
 * It returns either the identity to continue with, or a finished refusal
 * result — the only two outcomes a caller has to handle.
 */
async function preflightRun(
  runtime: ComfyUIRuntime,
  client: ComfyUIClient,
  requestedLabel: unknown,
  workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>,
  requestedSeed: number | undefined,
  params: Record<string, unknown>,
  label: string,
): Promise<{ ok: false; result: RunResult } | { ok: true; runDir: string; runLabel: string; seed: number | null; seeds: Record<string, number> }> {
  const { seed, seeds } = resolveSeeds(workflow, requestedSeed)
  let objectInfo: Record<string, unknown>
  try {
    objectInfo = await client.objectInfo()
  } catch (error) {
    const failure = preflightUnavailable(error instanceof Error ? error.message : String(error))
    const { runDir, runLabel } = openRun(runtime, requestedLabel, { params, workflowName: label, seed: seed ?? undefined, seeds })
    return { ok: false, result: refusedRun(runLabel, runDir, params, failure, seed, seeds) }
  }
  const preflight = preflightWorkflow(workflow, objectInfo)
  const { runDir, runLabel } = openRun(runtime, requestedLabel, { params, workflowName: label, seed: seed ?? undefined, seeds })
  if (!preflight.ok) {
    return { ok: false, result: refusedRun(runLabel, runDir, params, preflightFailure(preflight, runDir), seed, seeds) }
  }
  return { ok: true, runDir, runLabel, seed, seeds }
}

function missing(args: Record<string, unknown>, name: string): boolean {
  return args[name] === undefined || args[name] === null
}

function requireOneOf(args: Record<string, unknown>, names: readonly string[]): string | undefined {
  const present = names.filter((name) => !missing(args, name))
  if (present.length === 0) return `exactly one of ${names.join(', ')} is required`
  if (present.length > 1) return `only one of ${names.join(', ')} may be given`
  return undefined
}

/**
 * The refusal for an argument this tool never declares, or `undefined` when
 * every key is known. The message lists what IS accepted: the host does not
 * validate `parameters` on the raw channel, so without this the model gets no
 * feedback at all and the typo is silently dropped.
 */
function undeclaredArgumentFailure(tool: ToolDefinition, args: Record<string, unknown>): RunFailure | undefined {
  const undeclared = undeclaredArgumentKeys(tool.parameters, args)
  if (undeclared.length === 0) return undefined
  const declared = Object.keys((tool.parameters['properties'] ?? {}) as Record<string, unknown>)
  return {
    code: 'UNKNOWN_ARGUMENT',
    message: `${tool.name}: 未声明的参数 ${undeclared.join(', ')} — 本工具接受的参数：${declared.join(', ')}。宿主不校验 raw 通道的 parameters，拼错的键只会被静默忽略，故在此显式拒绝。`,
    nodeErrors: null,
  }
}

function buildWorkflow(args: Record<string, unknown>): { workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>; label: string } {
  const template = args.template
  if (typeof template === 'string') {
    const found = findTemplate(template)
    if (found === undefined) {
      throw new Error(`comfyui_run: unknown template "${template}" — use one of ${TEMPLATES.map((t) => t.id).join(', ')}`)
    }
    const workflow = cloneWorkflow(found.workflow)
    const inputs = args.inputs
    if (inputs !== undefined) {
      if (typeof inputs !== 'object' || inputs === null) {
        throw new Error('comfyui_run: inputs must be an object keyed by node id')
      }
      applyTemplateInputs(workflow, inputs as Record<string, Record<string, unknown>>)
    }
    return { workflow, label: `comfyui ${template}` }
  }
  const workflow = args.workflow
  if (typeof workflow !== 'object' || workflow === null) {
    throw new Error('comfyui_run: workflow must be an object')
  }
  const inputs = args.inputs
  if (inputs !== undefined) {
    if (typeof inputs !== 'object' || inputs === null) {
      throw new Error('comfyui_run: inputs must be an object keyed by node id')
    }
    applyTemplateInputs(workflow as Record<string, { class_type: string; inputs: Record<string, unknown> }>, inputs as Record<string, Record<string, unknown>>)
  }
  return { workflow: workflow as Record<string, { class_type: string; inputs: Record<string, unknown> }>, label: 'comfyui custom workflow' }
}

function summarizeMedia(media: RunMediaItem[]): string {
  if (media.length === 0) return 'no media outputs'
  const images = media.filter((item) => item.kind === 'image').length
  const videos = media.filter((item) => item.kind === 'video').length
  const audio = media.filter((item) => item.kind === 'audio').length
  const others = media.length - images - videos - audio
  const parts: string[] = []
  if (images > 0) parts.push(`${images} image(s)`)
  if (videos > 0) parts.push(`${videos} video(s)`)
  if (audio > 0) parts.push(`${audio} audio file(s)`)
  if (others > 0) parts.push(`${others} other file(s)`)
  return parts.join(', ')
}

/** A node input name that carries a sampling seed (`seed`, `noise_seed`, …). */
function isSeedKey(key: string): boolean {
  return /seed/i.test(key)
}

/** One seed-typed input found in a workflow. */
interface SeedSlot {
  nodeId: string
  inputKey: string
  /** The value the workflow carried before resolution. */
  authored: number
}

function seedSlotsOf(workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>): SeedSlot[] {
  const slots: SeedSlot[] = []
  for (const [nodeId, node] of Object.entries(workflow)) {
    for (const [inputKey, value] of Object.entries(node.inputs ?? {})) {
      if (!isSeedKey(inputKey)) continue
      if (typeof value !== 'number' || !Number.isFinite(value)) continue
      slots.push({ nodeId, inputKey, authored: value })
    }
  }
  return slots
}

/**
 * Write down the actual value of every seed-typed input before submitting.
 *
 * ComfyUI runs whatever seed the graph carries, so a workflow shipping a fixed
 * authored seed (the built-in templates default to `0`) reproduces one image
 * forever while `-1` leaves the value to server-side randomness — neither is
 * replayable after the fact. Resolution happens here, in the implementation,
 * and the result goes into the ledger, in this order:
 *
 * 1. the call's `seed` argument, when given — an explicit request wins over the
 *    graph's authored value, which is what makes `comfyui_run({ seed })` and
 *    `comfyui_run({ inputs: { "3": { seed } } })` behave the same way;
 * 2. otherwise the authored value, when it is a concrete seed (`>= 0` and not
 *    the `-1` "randomize" sentinel);
 * 3. otherwise one seed is drawn, and written into every seed input that
 *    shared the same authored value, so a graph's sampler and any sibling seed
 *    still replay from a single recorded number.
 */
function resolveSeeds(
  workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>,
  requested: number | undefined,
): { seed: number | null; seeds: Record<string, number> } {
  const slots = seedSlotsOf(workflow)
  if (slots.length === 0) return { seed: null, seeds: {} }
  const byAuthored = new Map<number, SeedSlot[]>()
  for (const slot of slots) {
    const group = byAuthored.get(slot.authored)
    if (group === undefined) byAuthored.set(slot.authored, [slot])
    else group.push(slot)
  }
  const seeds: Record<string, number> = {}
  let primary: number | null = null
  for (const [authored, group] of byAuthored) {
    const value = requested ?? (authored >= 0 && authored !== -1 ? authored : Math.floor(Math.random() * 2 ** 32))
    for (const slot of group) {
      workflow[slot.nodeId]!.inputs[slot.inputKey] = value
      seeds[`${slot.nodeId}.${slot.inputKey}`] = value
    }
    if (primary === null) primary = value
  }
  return { seed: primary, seeds }
}

/** The `filename_prefix` node inputs of a workflow, for the run's output naming. */
function filenamePrefixes(workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>): string[] {
  const prefixes: string[] = []
  for (const node of Object.values(workflow)) {
    const value = node.inputs?.['filename_prefix']
    if (typeof value === 'string' && value !== '') prefixes.push(value)
  }
  return prefixes
}

/** Media refs of a completed run, reduced to the ledger's file shape. */
function ledgerFiles(media: RunMediaItem[]): LedgerRecord['media'] {
  return media.map(({ filename, subfolder, type }) => ({ filename, subfolder, type }))
}

/** Render one run result: status line, seed, media, then any failure detail. */
function renderRunResult(_args: unknown, value: unknown): unknown[] {
  const result = value as RunResult | BackgroundResult
  if (result.kind === 'background') {
    return [{
      type: 'text',
      text: `ComfyUI generation started in the background (job ${result.jobId}, prompt ${result.promptId}). Collect the result with job_output.`,
    }]
  }
  const lines = [
    `ComfyUI ${result.status} (prompt ${result.promptId}, run ${result.runLabel}) in ${result.elapsedMs} ms — ${summarizeMedia(result.media)}`,
  ]
  if (result.error !== null) {
    lines.push(`  error ${result.error.code}: ${result.error.message}`)
    for (const node of result.error.nodeErrors ?? []) {
      lines.push(`    node ${node.node} (${node.classType}).${node.input}: ${node.reason}`)
    }
  }
  if (result.seed !== null) {
    const distinct = [...new Set(Object.values(result.seeds ?? {}))]
    lines.push(`  seed: ${result.seed}${distinct.length > 1 ? ` (${JSON.stringify(result.seeds)})` : ''} — 已写入台账，可复跑`)
  }
  for (const item of result.media) {
    lines.push(`  ${item.kind}: ${item.url}`)
  }
  if (result.ledger !== null) lines.push(`  台账: ${result.ledger}`)
  return [{ type: 'text', text: lines.join('\n') }]
}

/**
 * Text of the first seed-keyed exception ComfyUI reported, so a failed run
 * still explains itself in one line.
 *
 * The empty state returns `''`, NOT a placeholder sentence. Every caller reads
 * this through `|| <other source>` (or adds its own fallback), and a non-empty
 * constant here made that `||` a dead branch: `firstErrorText(errors) ||
 * error.message` could never fall through, so a ComfyUI run that failed for a
 * reason with no node-level detail (a transport error, an abort, a successful
 * execution whose post-processing threw) was recorded as the flat
 * `'unknown error'` and the real cause was destroyed at the reporting site.
 * Returning the empty string keeps "there is nothing to say here" and "the
 * placeholder wording" in the caller, where the fallback wording lives.
 */
function firstErrorText(errors: RunFailure['nodeErrors']): string {
  const first = errors?.[0]
  if (first === undefined) return ''
  return `${first.classType !== '' ? `${first.classType} ` : ''}${first.reason}`
}

/**
 * The terminal state one `waitAndRecord` hands back to its caller.
 *
 * `failure` is the whole failure the recorder already wrote to the ledger,
 * carried through so the caller reports the same text the row holds. It is
 * nullable for every non-failure terminal (`completed`, `interrupted`), where
 * the caller keeps its own wording.
 */
type TerminalState = {
  status: RunResult['status']
  media: RunMediaItem[]
  durationMs: number
  errors: RunFailure['nodeErrors']
  failure: RunFailure | null
}

/**
 * Wait for a submitted prompt, then record its terminal state **in the same
 * ledger row** the queued record opened (identity key first, promptId second)
 * and hand the collected media to `onTerminal`.
 *
 * The terminal record reuses the whole queued snapshot, so a replay reads the
 * seed and params that were actually submitted rather than the workflow as
 * authored.
 */
async function waitAndRecord(
  runtime: ComfyUIRuntime,
  client: ComfyUIClient,
  promptId: string,
  config: Config,
  signal: AbortSignal,
  timeoutMs: number | undefined,
  record: LedgerRecord,
  runDir: string,
  onTerminal: (state: TerminalState) => void,
): Promise<ComfyUIHistoryEntry | undefined> {
  const startedAt = Date.now()
  try {
    const entry = await client.waitForCompletion({
      promptId,
      timeoutMs: timeoutMs ?? config.timeoutMs,
      pollIntervalMs: config.pollIntervalMs,
      signal,
    })
    const media = collectMedia({ promptId, entry, maxItems: config.maxMediaItems, proxyBase: runtime.proxyBase() })
    const durationMs = Date.now() - startedAt
    upsertRun(runDir, { ...record, promptId, status: 'completed', durationMs, media: ledgerFiles(media) })
    onTerminal({ status: 'completed', media, durationMs, errors: null, failure: null })
    return entry
  } catch (error) {
    runtime.untrack(promptId)
    const durationMs = Date.now() - startedAt
    if (error instanceof Error && error.name === 'ComfyUIError' && error.message.includes('interrupted')) {
      upsertRun(runDir, { ...record, promptId, status: 'interrupted', durationMs })
      onTerminal({ status: 'interrupted', media: [], durationMs, errors: null, failure: null })
      return undefined
    }
    // A failed run is a terminal state too: the row is closed out with the
    // node-level detail ComfyUI reported, so it is not left reading "queued".
    const entry = await client.getHistory(promptId).catch(() => undefined)
    const errors = extractNodeErrors(entry)
    const failure: RunFailure = {
      code: 'EXECUTION_FAILED',
      message: firstErrorText(errors) || (error instanceof Error ? error.message : String(error)),
      nodeErrors: errors,
    }
    upsertRun(runDir, { ...record, promptId, status: 'failed', durationMs, error: failure })
    // `failure` is handed over WHOLE, not rebuilt by the caller from `errors`.
    // Node-less failures (a transport error, an abort, a post-processing throw)
    // carry `nodeErrors: null` and a message that exists only here — a caller
    // that reassembles the message out of `errors` alone silently replaces the
    // real cause with a placeholder, which is how the tool result and its
    // ledger row came to disagree about the same run.
    onTerminal({ status: 'error', media: [], durationMs, errors, failure })
    return entry
  }
}

/**
 * Open one run's ledger row: `runLabel` and the sequence number are drawn from
 * the ledger and the `queued` row is written, all in ONE synchronous stretch.
 *
 * This function is deliberately synchronous and must stay that way. Reading the
 * next sequence number and writing the row that reserves it are two ledger
 * accesses; any `await` between them lets a second run (another agent, another
 * lane, sharing this plugin instance and download directory) read the same
 * number and open the same key, so the two runs' records collapse into one row
 * and the cursor is consumed twice. Everything that needs awaiting — resolving
 * the API key, building the workflow, resolving seeds — happens before this
 * call, never inside it.
 */
function openRun(
  runtime: ComfyUIRuntime,
  requestedLabel: unknown,
  entry: { params: Record<string, unknown>; workflowName?: string | null; seed?: number; seeds?: Record<string, number> },
): { runDir: string; runLabel: string; requested: string; job: number } {
  const runDir = runtime.downloadDir()
  const requested = typeof requestedLabel === 'string' && requestedLabel.trim() !== ''
    ? requestedLabel.trim()
    : resolveRunPrefix()
  // An explicit label ending in digits is already a sequence; anything else is
  // a prefix and gets the next number this ledger has not used.
  const explicit = /-\d+$/.test(requested)
  const job = explicit ? 0 : nextJobNumber(runDir, requested)
  const runLabel = explicit ? requested : formatRunLabel(requested, job)
  const base: LedgerRecord = {
    runLabel,
    seed: entry.seed ?? undefined,
    seeds: entry.seeds ?? {},
    params: entry.params,
    workflowName: entry.workflowName ?? null,
    ts: new Date().toISOString(),
    status: 'queued',
  }
  // The queued row is written before the submit, so an interrupt or a crash
  // between here and the terminal state still leaves the attempt on record.
  upsertRun(runDir, base)
  return { runDir, runLabel, requested, job }
}

/** A run result with no terminal state yet (used for the refused paths). */
function emptyResult(runLabel: string, runDir: string | null, failure: RunFailure): RunResult {
  return {
    kind: 'sync',
    promptId: runLabel,
    runLabel,
    status: 'error',
    elapsedMs: 0,
    media: [],
    summary: 'not submitted',
    seed: null,
    seeds: null,
    ledger: runDir,
    error: failure,
  }
}

/**
 * A run that never reached the server: the refusal (or the failed submit) is
 * its own ledger row, so "what did this run key try to do" stays answerable.
 */
function refusedRun(
  runLabel: string,
  runDir: string,
  params: Record<string, unknown>,
  failure: RunFailure,
  seed: number | null,
  seeds: Record<string, number>,
): RunResult {
  upsertRun(runDir, {
    runLabel,
    seed: seed ?? undefined,
    seeds,
    params,
    ts: new Date().toISOString(),
    status: 'failed',
    durationMs: 0,
    error: failure,
  })
  const result = emptyResult(runLabel, runDir, failure)
  return {
    ...result,
    seed,
    seeds: Object.keys(seeds).length > 0 ? seeds : null,
  }
}

/**
 * The async takeover failed **after** the prompt was submitted: the job could
 * not be started, but ComfyUI already has the prompt and may still finish it.
 * The run is therefore not "not submitted", and the row cannot stay `queued`.
 *
 * Two things have to come out of this failure. The row the reservation opened
 * must be closed out (the sequence number is spent), and it must carry the
 * **real** `promptId` — that is the only handle `comfyui_fetch_output` has on a
 * run whose job never existed, and without it a submitted prompt is
 * unreachable. The message keeps the `background jobs unavailable` fragment
 * callers match on, and says in so many words that the submit already happened,
 * so it is never read as "nothing was sent".
 */
function jobsTakeoverFailure(tool: string, promptId: string, runLabel: string, cause: unknown): RunFailure {
  const reason = cause instanceof Error ? cause.message : String(cause)
  return {
    code: 'JOBS_UNAVAILABLE',
    message: `${tool}: background jobs unavailable — 后台任务接管失败(${reason})。注意：prompt 已提交到 ComfyUI、可能仍会执行完成，promptId=${promptId}（runLabel=${runLabel}），可用 comfyui_fetch_output 以该 promptId 取回结果。接管能力需加载 @deepseek-ai/dsh-jobs-local 与 @deepseek-ai/dsh-tool-jobs。`,
    nodeErrors: null,
  }
}

function runDefinition(runtime: ComfyUIRuntime, ctx: Context): ToolDefinition {
  return {
    name: 'comfyui_run',
    description: [
      'Submit a workflow to the configured ComfyUI server and return the generated media (images/videos).',
      'Provide exactly one of `workflow` (ComfyUI API-format object: node id → { class_type, inputs }) or `template` (built-in: txt2img | img2img | video).',
      'Use `inputs` to override node inputs by id, e.g. {"6": {"text": "a red cat"}} for the positive prompt in the templates.',
      'Templates: txt2img — 4 checkpoint, 5 EmptyLatentImage (width/height), 6 positive text, 7 negative text, 3 KSampler (seed/steps/cfg/denoise), 9 SaveImage. img2img — 10 LoadImage (image), 11 VAEEncode, 6 text, 3 KSampler (denoise). video — Wan 2.1, needs ComfyUI-WanVideoWrapper custom nodes (10 UNETLoader, 13 WanTextEncode, 14 WanImageToVideo, 15 KSampler, 17 SaveVideo).',
      'Inspect available node types with comfyui_object_info before hand-writing a workflow.',
      'Before anything is submitted the workflow is preflighted against the server node definitions: an unregistered class type, or a loader naming a model that is not on disk, refuses the run with a structured error listing the missing items and the values that ARE available. Nothing is ever downloaded to satisfy a missing model.',
      '`seed` writes one concrete sampling seed into every seed input the graph carries and records it in the run ledger, so a run is reproducible afterwards; a seed already present in the workflow (or passed through `inputs`) is kept as-is. `run_label` fixes the run identity (`<批次>-<lane>-<job>`, default `<prefix>-<NNNN>`); it is the ledger key, and `queued` records are overwritten in place by the terminal state.',
      'Every run is recorded in `<downloadDir>/runs.json` with its runLabel, resolved seeds, status and media; comfyui_fetch_output downloads those files to disk.',
      '`mode: sync` (default) waits and returns media URLs; `mode: async` starts a background job and returns a job id for job_output.',
    ].join(' '),
    parameters: {
      type: 'object',
      // Closed deliberately, for two reasons that are still true: it documents
      // the exact argument surface, and it is what the PTC SDK rendering
      // (`jsonSchemaToTs`) turns into the tool's input type. It does NOT make
      // a misspelled argument fail: the host never validates a raw-channel
      // `parameters` — not at register time (`dsh-tools` only runs
      // `assertSupportedJsonSchema` on `output.schema`) and not at call time
      // (only the returned value is validated). The real risk runs the other
      // way: if a keyword here is outside the host's supported subset, that
      // PTC rendering throws, the failure is swallowed, and this tool's input
      // type silently degrades to `unknown`. Enforcement of the declared key
      // set lives in `execute` (see `undeclaredArgumentFailure`).
      additionalProperties: false,
      properties: {
        workflow: { type: 'object', additionalProperties: true, description: 'ComfyUI API-format workflow: node id → { class_type, inputs }. Alternative to `template`.' },
        template: { type: 'string', enum: ['txt2img', 'img2img', 'video'], description: 'Built-in workflow template id. Alternative to `workflow`.' },
        inputs: { type: 'object', additionalProperties: true, description: 'Per-node input overrides keyed by node id, e.g. {"3": {"seed": 42, "steps": 30}, "6": {"text": "prompt"}}.' },
        mode: { type: 'string', enum: ['sync', 'async'], default: 'sync', description: 'sync waits and returns media; async returns a background job id.' },
        timeout_ms: { type: 'number', description: 'Generation wait budget in ms (default 180000). Video needs minutes.' },
        seed: { type: 'integer', description: 'One concrete sampling seed for this run, written into every seed input the graph carries and recorded in the ledger (reproducible replay). Explicit here it wins over the graph\'s authored value; omitted, the authored value is kept (the built-in templates default to 0, so pass a seed to vary the result).' },
        run_label: { type: 'string', description: 'Governance identity of this run (`<批次>-<lane>-<job>`), used as the ledger key; default `<COMFYUI_RUN_PREFIX 或 comfyui>-<NNNN>` with the sequence drawn from the ledger.' },
      },
      required: [],
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: renderRunResult,
      presentationMeta(_args, value) {
        const result = value as RunResult | BackgroundResult
        if (result.kind === 'background') {
          return { kind: 'background', jobId: result.jobId, promptId: result.promptId, label: result.label }
        }
        return {
          kind: 'sync',
          promptId: result.promptId,
          status: result.status,
          elapsedMs: result.elapsedMs,
          media: result.media,
          summary: result.summary,
        }
      },
    },
    timeoutMs: TOOL_TIMEOUT_MS,
    async execute(args, exec) {
      const problem = requireOneOf(args, ['workflow', 'template'])
      if (problem !== undefined) throw new Error(`comfyui_run: ${problem}`)
      const undeclaredArgs = undeclaredArgumentFailure(this, args)
      if (undeclaredArgs !== undefined) return emptyResult('', null, undeclaredArgs)
      const mode = args.mode === undefined ? 'sync' : args.mode
      if (mode !== 'sync' && mode !== 'async') throw new Error(`comfyui_run: mode must be sync or async, got ${String(mode)}`)
      // Background mode is decided before anything is committed: without a jobs
      // service the run is refused here, so no ledger row is opened and no
      // sequence number is consumed for a run that could never be collected.
      // Wording kept verbatim for callers that match on it.
      if (mode === 'async' && ctx.get('jobs') === undefined) {
        throw new Error('comfyui_run: background jobs unavailable — load @deepseek-ai/dsh-jobs-local and @deepseek-ai/dsh-tool-jobs')
      }
      const config = runtime.getConfig()
      const apiKey = await runtime.getApiKey()
      const client = runtime.createClient(apiKey)
      const { workflow, label } = buildWorkflow(args)
      const waitMs = typeof args.timeout_ms === 'number' ? args.timeout_ms : config.timeoutMs

      const requestedSeed = typeof args.seed === 'number' && Number.isFinite(args.seed) && args.seed !== -1
        ? Math.floor(args.seed)
        : undefined
      // The ledger stores the run's own settings, not the workflow body: the
      // graph is addressed by runLabel (and its seed by `seed`), which is what
      // a replay needs without copying a whole workflow into the bookkeeping.
      const params: Record<string, unknown> = {
        ...(typeof args.template === 'string' ? { template: args.template } : {}),
        ...(typeof args.mode === 'string' ? { mode: args.mode } : {}),
        ...(typeof args.timeout_ms === 'number' ? { timeout_ms: args.timeout_ms } : {}),
        node_overrides: args.inputs === undefined ? 0 : Object.keys(args.inputs as Record<string, unknown>).length,
      }
      // Identity, queued row, early preflight, seeds — in that order, with the
      // reserve-then-write pair kept inside one synchronous function (B-2).
      const prepared = await preflightRun(runtime, client, args.run_label, workflow, requestedSeed, params, label)
      if (!prepared.ok) return prepared.result
      const { runDir, runLabel, seed, seeds } = prepared
      const base: LedgerRecord = {
        runLabel,
        seed: seed ?? undefined,
        seeds,
        params,
        workflowName: label,
        ts: new Date().toISOString(),
        status: 'queued',
      }
      const startedAt = Date.now()
      let promptId: string
      try {
        promptId = await runtime.queue(workflow, { workflowName: label, source: 'tool' })
      } catch (error) {
        // The submit throat refuses a graph it cannot preflight (`PREFLIGHT*`):
        // that is a refusal, not a submit failure, and it carries its own code
        // and detail. Anything else really was a failed submit.
        const refusal = preflightRefusalFailure(error)
        if (refusal !== undefined) {
          // The row the reservation opened is closed out as a refusal: it must
          // carry the refusal's own code, not a generic submit failure.
          return refusedRun(runLabel, runDir, params, refusal, seed, seeds)
        }
        const failure: RunFailure = {
          code: 'SUBMIT_FAILED',
          message: error instanceof Error ? error.message : String(error),
          nodeErrors: null,
        }
        upsertRun(runDir, { ...base, status: 'failed', durationMs: Date.now() - startedAt, error: failure })
        return {
          kind: 'sync',
          promptId: runLabel,
          runLabel,
          status: 'error',
          elapsedMs: Date.now() - startedAt,
          media: [],
          summary: 'not submitted',
          seed,
          seeds: Object.keys(seeds).length > 0 ? seeds : null,
          ledger: runDir,
          error: failure,
        }
      }
      // Seed and prefixes are written into the submitted graph before submit,
      // so the ledger snapshot is the graph's actual values.
      base.promptId = promptId

      if (mode === 'async') {
        let jobId: string
        try {
          const jobs = ctx.get('jobs') as JobsService
          jobId = jobs.start({
            kind: 'comfyui',
            label,
            ...(exec.agent !== undefined ? { owner: exec.agent } : {}),
            run: () => {
              let terminal: RunResult | undefined
              const done = (async () => {
                const record = { ...base }
                await waitAndRecord(
                  runtime, client, promptId, config, new AbortController().signal, waitMs, record, runDir,
                  (state) => {
                    terminal = {
                      kind: 'sync',
                      promptId,
                      runLabel,
                      status: state.status,
                      elapsedMs: state.durationMs,
                      media: state.media,
                      summary: summarizeMedia(state.media),
                      seed,
                      seeds: Object.keys(seeds).length > 0 ? seeds : null,
                      ledger: runDir,
                      error: state.failure,
                    }
                  },
                )
                if (terminal === undefined) {
                  return { status: 'failed' as const, detail: 'comfyui', output: `run ${runLabel} produced no terminal state` }
                }
                return terminal.status === 'error'
                  ? { status: 'failed' as const, detail: 'comfyui', output: JSON.stringify(terminal) }
                  : { status: 'completed' as const, output: JSON.stringify(terminal) }
              })()
              return {
                cancel: () => { void client.interrupt().catch(() => undefined) },
                done,
              }
            },
          })
        } catch (error) {
          // Submitted but not taken over: the row is closed out with the real
          // promptId before the caller hears about it, so no `queued` row is
          // left behind and the submitted prompt stays fetchable (BI-1). The
          // error is re-thrown rather than returned — the caller must not read
          // a failed takeover as a background run that started.
          const failure = jobsTakeoverFailure('comfyui_run', promptId, runLabel, error)
          upsertRun(runDir, { ...base, status: 'failed', durationMs: Date.now() - startedAt, error: failure })
          throw new Error(failure.message)
        }
        const result: BackgroundResult = { kind: 'background', jobId, promptId, runLabel, label, seed, ledger: runDir }
        return result
      }

      let outcome: TerminalState | undefined
      await waitAndRecord(runtime, client, promptId, config, exec.signal, waitMs, base, runDir, (state) => { outcome = state })
      const terminal: TerminalState = outcome ?? { status: 'error', media: [], durationMs: Date.now() - startedAt, errors: null, failure: null }
      return {
        kind: 'sync',
        promptId,
        runLabel,
        status: terminal.status,
        elapsedMs: terminal.durationMs,
        media: terminal.media,
        summary: summarizeMedia(terminal.media),
        seed,
        seeds: Object.keys(seeds).length > 0 ? seeds : null,
        ledger: runDir,
        error: terminal.failure ?? (terminal.errors === null && terminal.status !== 'error' ? null : {
          code: 'EXECUTION_FAILED',
          message: firstErrorText(terminal.errors) || (terminal.status === 'interrupted' ? 'interrupted before completion' : 'unknown error'),
          nodeErrors: terminal.errors,
        }),
      }
    },
  }
}

interface FieldSummary {
  name: string
  type: string
  required: boolean
  default?: unknown
  options?: string[]
}

function summarizeFields(fields: Record<string, unknown> | undefined, required: boolean): FieldSummary[] {
  const out: FieldSummary[] = []
  for (const [name, spec] of Object.entries(fields ?? {})) {
    if (!Array.isArray(spec)) continue
    const [typeOrList, options] = spec as [unknown, unknown?]
    const optionsRecord = typeof options === 'object' && options !== null ? options as Record<string, unknown> : undefined
    const entry: FieldSummary = { name, type: 'unknown', required }
    if (Array.isArray(typeOrList)) {
      entry.type = 'enum'
      entry.options = (typeOrList as unknown[]).slice(0, 6).map(String)
    } else if (typeof typeOrList === 'string') {
      entry.type = typeOrList
    }
    if (optionsRecord !== undefined && 'default' in optionsRecord) {
      entry.default = optionsRecord.default
    }
    out.push(entry)
    if (out.length >= 14) break
  }
  return out
}

function objectInfoDefinition(runtime: ComfyUIRuntime): ToolDefinition {
  return {
    name: 'comfyui_object_info',
    description: 'List the node definitions the configured ComfyUI server supports (class types, required and optional inputs). Use it to build valid API-format workflows for comfyui_run. Optional `filter` narrows by class-name substring, e.g. "KSampler", "VAE", "LoadImage".',
    parameters: {
      type: 'object',
      // Closed deliberately, for two reasons that are still true: it documents
      // the exact argument surface, and it is what the PTC SDK rendering
      // (`jsonSchemaToTs`) turns into the tool's input type. It does NOT make
      // a misspelled argument fail: the host never validates a raw-channel
      // `parameters` — not at register time (`dsh-tools` only runs
      // `assertSupportedJsonSchema` on `output.schema`) and not at call time
      // (only the returned value is validated). The real risk runs the other
      // way: if a keyword here is outside the host's supported subset, that
      // PTC rendering throws, the failure is swallowed, and this tool's input
      // type silently degrades to `unknown`. Enforcement of the declared key
      // set lives in `execute` (see `undeclaredArgumentFailure`).
      additionalProperties: false,
      properties: {
        filter: { type: 'string', description: 'Optional substring filter on node class names.' },
      },
      required: [],
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render(_args, value) {
        const data = value as { total: number; shown: number; nodes: Array<{ class_type: string; display_name?: string; description?: string }>; hint?: string }
        const lines = [`ComfyUI nodes: ${data.total} total, showing ${data.shown}`]
        for (const node of data.nodes) {
          lines.push(`- ${node.class_type}${node.display_name !== undefined ? ` (${node.display_name})` : ''}${node.description !== undefined && node.description !== '' ? `: ${node.description}` : ''}`)
        }
        if (data.hint !== undefined) lines.push(data.hint)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    timeoutMs: 60_000,
    async execute(args) {
      // `output.schema` here is closed and has no refusable shape, so the
      // refusal is thrown with the same code in it rather than faked as an
      // empty listing — an empty answer would read as "the server has no
      // nodes", which is the opposite of what happened.
      const undeclaredArgs = undeclaredArgumentFailure(this, args)
      if (undeclaredArgs !== undefined) throw new Error(undeclaredArgs.message)
      const client = runtime.createClient(await runtime.getApiKey())
      const raw = await client.objectInfo()
      const entries = Object.entries(raw as Record<string, {
        display_name?: string
        description?: string
        input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> }
      }>)
      const filter = typeof args.filter === 'string' ? args.filter.trim().toLowerCase() : undefined
      const filtered = filter === undefined || filter === ''
        ? entries
        : entries.filter(([name]) => name.toLowerCase().includes(filter))
      const nodes = filtered.slice(0, 60).map(([classType, def]) => ({
        class_type: classType,
        display_name: def.display_name,
        description: (def.description ?? '').slice(0, 200),
        required: summarizeFields(def.input?.required, true),
        optional: summarizeFields(def.input?.optional, false),
      }))
      return {
        total: entries.length,
        shown: nodes.length,
        filter: filter ?? null,
        hint: filtered.length > nodes.length ? `filter matched ${filtered.length} nodes, showing first ${nodes.length} — narrow the filter for more` : undefined,
        nodes,
      }
    },
  }
}

/** List and run saved workflows from the panel-managed library. */
function workflowDefinition(runtime: ComfyUIRuntime, ctx: Context): ToolDefinition {
  return {
    name: 'comfyui_workflow',
    description: [
      'List and run saved ComfyUI workflows from the plugin workflow library (运行主题: API-format workflows extracted from a graph or pasted directly).',
      '`action: list` returns every runnable workflow with its id, name, description (what the workflow does), and input notes.',
      'It also lists workflows the user saved on the ComfyUI server (衍生主题: UI graph format canvases). A graph may hold SEVERAL independent flows; each is extracted into its own runnable workflow in the panel (整体/按分量/主流程). A graph with no extracted workflow yet cannot run — tell the user to open the ComfyUI panel and 提取 it first.',
      '`action: run` runs one saved workflow by id — pass only the id plus parameter overrides; the plugin submits the saved workflow JSON itself (never copy the JSON into your reply). It waits for media by default; add `mode: "async"` to run in the background and collect the result with job_output.',
      '`action: get` returns one saved workflow\'s complete API-format JSON by id for inspection/diagnostics only — it consumes many tokens and is not the run path.',
      '`action: refresh` re-derives one saved workflow\'s parameter snapshot (options / numberKind / min/max/step) from the current node definitions and saves it back. Run it after the TTS-Audio-Suite voice library or node definitions changed: saved workflows snapshot their parameter options at save time, so a freshly added voice is not accepted by `action: run` until the snapshot catches up. It force-rescans the TTS voice library first, then updates only the fields derived from object_info — the parameter set (including user-added advanced parameters) is preserved. Returns `changed` with the parameter names that actually changed.',
    ].join(' '),
    parameters: {
      type: 'object',
      // Closed deliberately, for two reasons that are still true: it documents
      // the exact argument surface, and it is what the PTC SDK rendering
      // (`jsonSchemaToTs`) turns into the tool's input type. It does NOT make
      // a misspelled argument fail: the host never validates a raw-channel
      // `parameters` — not at register time (`dsh-tools` only runs
      // `assertSupportedJsonSchema` on `output.schema`) and not at call time
      // (only the returned value is validated). The real risk runs the other
      // way: if a keyword here is outside the host's supported subset, that
      // PTC rendering throws, the failure is swallowed, and this tool's input
      // type silently degrades to `unknown`. Enforcement of the declared key
      // set lives in `execute` (see `undeclaredArgumentFailure`).
      additionalProperties: false,
      properties: {
        action: { type: 'string', enum: ['list', 'run', 'get', 'refresh'], description: 'list returns the workflow library; run executes one workflow by id (direct call to the saved JSON); get returns one workflow\'s full JSON for inspection; refresh re-derives one workflow\'s parameter snapshot from the current node definitions and saves it back.' },
        id: { type: 'string', description: 'Workflow id (required for action: run and get).' },
        mode: { type: 'string', enum: ['sync', 'async'], description: 'run mode (default sync); async starts a background job and returns its id for job_output. Video/audio workflows should use async — generation takes minutes and sync may time out.' },
        timeout_ms: { type: 'number', description: 'Generation wait budget in ms (default 900000 = 15 min). Video needs minutes; raise this for long videos.' },
        run_label: { type: 'string', description: 'Governance identity for this run (`<批次>-<lane>-<job>`), used as the run-ledger key; default `<COMFYUI_RUN_PREFIX 或 comfyui>-<NNNN>`.' },
        parameters: {
          type: 'object',
          additionalProperties: true,
          description: 'Optional per-run values for the workflow\'s adjustable parameters (see the workflow\'s `inputs` note from action: list — e.g. {"prompt": "a red cat", "seed": 42}). Omitted parameters keep their defaults; seed-type parameters randomize when the workflow marks them 随机.',
        },
      },
      required: ['action'],
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render(_args, value) {
        const data = value as {
          action: string
          env?: { baseUrl: string; comfyuiDirs: string[] }
          workflows?: Array<{ id: string; name: string; description: string; skill?: { summary: string; files: number; required: boolean }; parameters?: Array<{ name: string; label: string; type: string; default?: string | number | boolean; random?: boolean; numberKind?: 'int' | 'float'; options?: Array<string | number>; upload?: 'image' | 'video' | 'audio' | 'media'; subfolder?: string }> }>
          comfyuiWorkflows?: Array<{ name: string; extracted: boolean; derived: Array<{ libraryId: string; name: string }> }>
          loadArea?: { slots: number; loaded: number; items: Array<{ name: string; kind: string; source: string }> }
          result?: RunResult
          id?: string
          name?: string
          workflowName?: string
          workflow?: unknown
          background?: BackgroundResult
          changed?: string[]
          parameterCount?: number
          error?: RunFailure | null
          skill?: { workflowName: string; summary: string; body: string; resourceBase: string; files: string[] }
        }
        if (data.action === 'error') {
          return [{ type: 'text', text: data.error?.message ?? 'comfyui_workflow: 调用被拒绝' }]
        }
        if (data.action === 'skill') {
          const pack = data.skill
          if (pack === undefined) return [{ type: 'text', text: `ComfyUI workflow ${data.name ?? data.id} 没有技能包。` }]
          // Same framing the host uses for its own skills: a named block, the
          // resource base, then the body verbatim. The reference files stay on
          // disk until the body sends the model after one of them.
          const others = pack.files.filter((file) => file !== 'SKILL.md')
          return [{
            type: 'text',
            text: [
              `<skill_content name="${pack.workflowName}">`,
              '<skill_resources>',
              `Base directory for this skill: ${pack.resourceBase}`,
              others.length > 0
                ? `Files in this pack: ${others.join(', ')} — resolve them against the base directory and read one only when the instructions below point at it.`
                : 'This pack has no reference files.',
              '</skill_resources>',
              '',
              '<skill_instructions>',
              pack.body,
              '</skill_instructions>',
              '</skill_content>',
            ].join('\n'),
          }]
        }
        if (data.action === 'get') {
          return [{ type: 'text', text: `ComfyUI workflow ${data.name ?? data.id}: ${JSON.stringify(data.workflow)}` }]
        }
        if (data.background !== undefined) {
          return [{ type: 'text', text: `ComfyUI workflow started in the background (job ${data.background.jobId}, prompt ${data.background.promptId}). Collect the result with job_output.` }]
        }
        if (data.action === 'refresh') {
          const changed = data.changed ?? []
          const changedText = changed.length > 0
            ? `更新了 ${changed.length} 个参数的 options / 数值声明：${changed.join('、')}`
            : 'options 与数值声明与最新节点定义一致，没有变化'
          return [{ type: 'text', text: `ComfyUI workflow ${data.name ?? data.id}：参数快照已刷新（${data.parameterCount ?? 0} 个参数原样保留），${changedText}` }]
        }
        if (data.action === 'list') {
          const lines: string[] = []
          const env = data.env
          if (env !== undefined) {
            // The model's local-env one-liner: the ComfyUI server address and
            // the user's ComfyUI install dirs, so it never has to ask where
            // files live (TTS-Audio-Suite voice library etc.).
            lines.push(`ComfyUI 服务器: ${env.baseUrl}；本机 ComfyUI 目录: ${env.comfyuiDirs.length > 0 ? env.comfyuiDirs.join('；') : '未配置（设置页 comfyuiDirs 可填写）'}`)
          }
          lines.push(`Saved ComfyUI workflows (${data.workflows?.length ?? 0}):`)
          for (const workflow of data.workflows ?? []) {
            lines.push(`- ${workflow.id} — ${workflow.name}${workflow.description !== '' ? `: ${workflow.description}` : ''}`)
            const skill = workflow.skill
            if (skill !== undefined) {
              const extra = skill.files > 1 ? `，另有 ${skill.files - 1} 篇参考文档` : ''
              lines.push(`  技能包${skill.required ? '（运行前必读）' : ''}: ${skill.summary}${extra} — 运行前先 action: skill { id: "${workflow.id}" }`)
            }
            for (const param of workflow.parameters ?? []) {
              const def = typeof param.default === 'string' ? `"${param.default}"` : String(param.default)
              const options = Array.isArray(param.options) && param.options.length > 0 ? `，可选: ${param.options.join(' / ')}` : ''
              const upload = param.upload !== undefined
                ? `，上传类型: ${param.upload}${param.upload === 'media' ? `（${param.subfolder ?? ''}/，空值=移除该参考位）` : ''}`
                : ''
              lines.push(`  ${param.name}(${param.label}${param.random === true ? '，随机' : ''}，默认 ${def}${options}${upload})`)
            }
          }
          const loadArea = data.loadArea
          if (loadArea !== undefined && loadArea.slots > 0) {
            lines.push(`用户加载区（${loadArea.slots} 个加载位，已放入 ${loadArea.loaded} 个素材，未显式传值的加载参数按顺序取用）:`)
            for (const [index, item] of loadArea.items.entries()) {
              lines.push(`  ${index + 1}. ${item.name}（${item.kind}）`)
            }
          }
          const comfyui = data.comfyuiWorkflows ?? []
          if (comfyui.length > 0) {
            lines.push(`ComfyUI 端保存的图工作流（${comfyui.length} 个，UI 图格式，不能直接运行）:`)
            for (const workflow of comfyui) {
              if (workflow.extracted) {
                lines.push(`- ${workflow.name} — 已提取 ${workflow.derived.length} 个运行工作流：${workflow.derived.map((d) => `${d.name}(${d.libraryId})`).join('、')}`)
              } else {
                lines.push(`- ${workflow.name} — 未提取：如需运行，请转告用户先在 ComfyUI 面板里“提取”它（可选择整体/按分量/主流程）`)
              }
            }
          }
          return [{ type: 'text', text: lines.join('\n') }]
        }
        const result = data.result
        if (result === undefined) {
          if (data.error !== undefined && data.error !== null) {
            return [{ type: 'text', text: `ComfyUI workflow ${data.workflowName ?? data.id} 未提交: ${data.error.code} ${data.error.message}` }]
          }
          return [{ type: 'text', text: 'ComfyUI workflow run returned no result.' }]
        }
        const lines = [`ComfyUI workflow ${result.status} (prompt ${result.promptId}, run ${result.runLabel}) in ${result.elapsedMs} ms — ${summarizeMedia(result.media)}`]
        if (result.error !== null) lines.push(`  error ${result.error.code}: ${result.error.message}`)
        if (result.seed !== null) lines.push(`  seed: ${result.seed} — 已写入台账，可复跑`)
        for (const item of result.media) lines.push(`  ${item.kind}: ${item.url}`)
        if (result.ledger !== null) lines.push(`  台账: ${result.ledger}`)
        return [{ type: 'text', text: lines.join('\n') }]
      },
      presentationMeta(_args, value) {
        return value
      },
    },
    timeoutMs: TOOL_TIMEOUT_MS,
    async execute(args, exec) {
      const action = args.action
      if (action !== 'list' && action !== 'run' && action !== 'get' && action !== 'refresh' && action !== 'skill') {
        throw new Error(`comfyui_workflow: action must be list, run, skill, get, or refresh, got ${String(action)}`)
      }
      const undeclaredArgs = undeclaredArgumentFailure(this, args)
      if (undeclaredArgs !== undefined) {
        return { action: 'error', error: undeclaredArgs }
      }
      if (action === 'list') {
        const [workflows, comfyui, slots] = await Promise.all([
          runtime.listWorkflows(),
          runtime.listComfyWorkflows().catch(() => []),
          runtime.loadSlots().catch(() => [] as LoadSlot[]),
        ])
        // What the user currently has loaded in the panel's load area. Unset
        // loader parameters take these files in slot order, so the model can
        // see how many references a run will pick up without asking.
        const loaded = slots.filter((slot): slot is NonNullable<LoadSlot> => slot !== null)
        // Rung one of the skill-pack disclosure ladder: a workflow that has a
        // pack contributes ONE summary line here, not its body. The model
        // reaches for `action: skill` only after this listing points it at a
        // specific workflow, so an unused pack costs nothing.
        const packs = new Map<string, { summary: string; files: number; required: boolean }>()
        await Promise.all(workflows
          .filter((workflow) => workflow.skillDir !== undefined && workflow.skillDir !== '')
          .map(async (workflow) => {
            const pack = await runtime.skillPacks.infoFor(workflow).catch(() => undefined)
            if (pack === undefined) return
            packs.set(workflow.id, {
              summary: pack.summary !== '' ? pack.summary : `${workflow.name} 的使用说明`,
              files: pack.files.length,
              required: pack.required,
            })
          }))
        return {
          action: 'list',
          // Local environment the model can rely on: the ComfyUI server
          // address and the user's ComfyUI install dirs (configured in the
          // settings page comfyuiDirs). Fresh on every call, so config
          // changes show up without restarting DSH.
          env: {
            baseUrl: runtime.getConfig().baseUrl,
            comfyuiDirs: runtime.getConfig().comfyuiDirs,
          },
          loadArea: {
            slots: slots.length,
            loaded: loaded.length,
            items: loaded.map(({ name, kind, source }) => ({ name, kind, source })),
          },
          workflows: workflows.map(({ id, name, description, parameters, updatedAt }) => ({
            id,
            name,
            description,
            ...(packs.has(id) ? { skill: packs.get(id)! } : {}),
            parameters: (parameters ?? []).map(({ name: pname, label, type, default: def, random, numberKind, options, upload }) => {
              // DSH validates tool output as lossless JSON: JSON.stringify drops
              // undefined keys, so omit optional fields instead of passing undefined.
              const entry: Record<string, unknown> = { name: pname, label, type, default: def }
              if (random !== undefined) entry.random = random
              // Tells the model whether decimals are accepted; absent means the
              // node's declared type is unknown and a float is safe.
              if (numberKind !== undefined) entry.numberKind = numberKind
              if (options !== undefined) entry.options = options
              if (upload !== undefined) entry.upload = upload
              return entry
            }),
            updatedAt,
          })),
          comfyuiWorkflows: comfyui.map(({ name, extracted, derived }) => ({ name, extracted, derived })),
        }
      }
      const id = args.id
      if (typeof id !== 'string' || id === '') {
        throw new Error('comfyui_workflow: id is required for action: run, get, refresh and skill')
      }
      const saved = await runtime.getWorkflow(id)
      if (saved === undefined) {
        throw new Error(`comfyui_workflow: workflow "${id}" not found — run action: list first`)
      }
      if (action === 'get') {
        return {
          action: 'get',
          id: saved.id,
          name: saved.name,
          description: saved.description,
          parameters: saved.parameters ?? [],
          workflow: saved.workflow,
        }
      }
      if (action === 'skill') {
        const pack = await runtime.skillPacks.load(saved.id)
        if (!pack.ok) {
          throw new Error(`comfyui_workflow: ${pack.error}`)
        }
        // Reading the pack is what opens the `requireSkill` gate below.
        markSkillPackRead(exec.agent, saved.id)
        return {
          action: 'skill',
          id: saved.id,
          name: saved.name,
          skill: {
            workflowName: pack.value.workflowName,
            summary: pack.value.summary,
            body: pack.value.body,
            resourceBase: pack.value.resourceBase,
            files: pack.value.files,
          },
        }
      }
      if (action === 'refresh') {
        // The library snapshot is captured at save time and never re-reads
        // object_info, so a voice library that grew since then would reject
        // the new voice at run time. Force the TTS rescan first (the object_info
        // COMBO does not self-heal), then re-derive the derived fields only.
        await runtime.refreshVoiceLibrary()
        const client = runtime.createClient(await runtime.getApiKey())
        const objectInfo = await client.objectInfo().catch(() => undefined)
        const { parameters, changed } = refreshParameterMetadata(
          saved.parameters ?? [],
          objectInfo,
          saved.workflow,
        )
        const result = await runtime.saveWorkflow({
          id: saved.id,
          name: saved.name,
          description: saved.description,
          workflow: saved.workflow,
          parameters,
          source: saved.source,
          comfyuiFile: saved.comfyuiFile,
          tags: saved.tags,
        })
        if (!result.ok) {
          throw new Error(`comfyui_workflow: refresh failed: ${result.error}`)
        }
        return {
          action: 'refresh',
          id: saved.id,
          name: saved.name,
          parameterCount: parameters.length,
          changed,
        }
      }
      // The 必读 gate: workflows the user marked `requireSkill` refuse to run
      // until this session has actually loaded their pack. The reminder in the
      // listing is advisory; this is the part that holds.
      if (saved.requireSkill === true && saved.skillDir !== undefined && !hasReadSkillPack(exec.agent, saved.id)) {
        throw new Error(`comfyui_workflow: 工作流 "${saved.name}" 标记了运行前必读技能包 — 先调用 action: skill { id: "${saved.id}" } 读完再运行。`)
      }
      const config = runtime.getConfig()
      const client = runtime.createClient(await runtime.getApiKey())
      const values = typeof args.parameters === 'object' && args.parameters !== null
        ? (args.parameters as Record<string, unknown>)
        : {}
      const mode = args.mode === undefined ? 'sync' : args.mode
      if (mode !== 'sync' && mode !== 'async') {
        throw new Error(`comfyui_workflow: mode must be sync or async, got ${String(mode)}`)
      }
      // Background mode is decided before anything is committed: without a jobs
      // service the run is refused here, so no ledger row is opened and no
      // sequence number is consumed for a run that could never be collected.
      if (mode === 'async' && ctx.get('jobs') === undefined) {
        throw new Error('comfyui_workflow: background jobs unavailable — load @deepseek-ai/dsh-jobs-local and @deepseek-ai/dsh-tool-jobs')
      }
      const waitMs = typeof args.timeout_ms === 'number' ? args.timeout_ms : config.timeoutMs
      // The run identity is the same governance key comfyui_run uses, and it is
      // what makes this path land on the ledger instead of only in history.
      // Row plus sequence number in one synchronous stretch (B-2): nothing is
      // awaited between drawing the number and reserving it.
      const { runDir, runLabel } = openRun(runtime, args.run_label, {
        params: { workflowId: saved.id, workflowName: saved.name, values },
        workflowName: saved.name,
      })
      const base: LedgerRecord = {
        runLabel,
        params: { workflowId: saved.id, workflowName: saved.name, values },
        workflowName: saved.name,
        ts: new Date().toISOString(),
        status: 'queued',
      }

      const startedAt = Date.now()
      let promptId: string
      try {
        promptId = await runtime.queue(saved.workflow, {
          workflowName: saved.name,
          workflowId: saved.id,
          source: 'workflow-tool',
          parameters: saved.parameters,
          values,
        })
      } catch (error) {
        // A refusal from the submit throat is recorded as its own code, so a
        // caller can tell "this graph was rejected" from "the submit broke".
        const refusal = preflightRefusalFailure(error)
        if (refusal !== undefined) {
          upsertRun(runDir, { ...base, status: 'failed', durationMs: Date.now() - startedAt, error: refusal })
          return { action: 'run', id, workflowName: saved.name, runLabel, error: refusal }
        }
        const failure: RunFailure = {
          code: 'SUBMIT_FAILED',
          message: error instanceof Error ? error.message : String(error),
          nodeErrors: null,
        }
        upsertRun(runDir, { ...base, status: 'failed', durationMs: Date.now() - startedAt, error: failure })
        return { action: 'run', id, workflowName: saved.name, runLabel, error: failure }
      }
      const record: LedgerRecord = { ...base, promptId }

      if (mode === 'async') {
        let jobId: string
        try {
          const jobs = ctx.get('jobs') as JobsService
          jobId = jobs.start({
            kind: 'comfyui',
            label: saved.name,
            ...(exec.agent !== undefined ? { owner: exec.agent } : {}),
            run: () => {
              const done = (async () => {
                let terminal: RunResult | undefined
                await waitAndRecord(
                  runtime, client, promptId, config, new AbortController().signal, waitMs, record, runDir,
                  (state) => {
                    terminal = {
                      kind: 'sync',
                      promptId,
                      runLabel,
                      status: state.status,
                      elapsedMs: state.durationMs,
                      media: state.media,
                      summary: summarizeMedia(state.media),
                      seed: null,
                      seeds: null,
                      ledger: runDir,
                      error: state.failure,
                    }
                  },
                )
                if (terminal === undefined) {
                  return { status: 'failed' as const, detail: 'comfyui', output: `run ${runLabel} produced no terminal state` }
                }
                return terminal.status === 'error'
                  ? { status: 'failed' as const, detail: 'comfyui', output: JSON.stringify(terminal) }
                  : { status: 'completed' as const, output: JSON.stringify(terminal) }
              })()
              return {
                cancel: () => { void client.interrupt().catch(() => undefined) },
                done,
              }
            },
          })
        } catch (error) {
          // Second path, same contract as `comfyui_run`: close the row out with
          // the real promptId, then re-throw instead of reporting a background
          // job that was never started.
          const failure = jobsTakeoverFailure('comfyui_workflow', promptId, runLabel, error)
          upsertRun(runDir, { ...record, status: 'failed', durationMs: Date.now() - startedAt, error: failure })
          throw new Error(failure.message)
        }
        const background: BackgroundResult = { kind: 'background', jobId, promptId, runLabel, label: saved.name, seed: null, ledger: runDir }
        return { action: 'run', id, workflowName: saved.name, background }
      }
      let outcome: TerminalState | undefined
      await waitAndRecord(runtime, client, promptId, config, exec.signal, waitMs, record, runDir, (state) => { outcome = state })
      const terminal: TerminalState = outcome ?? { status: 'error', media: [], durationMs: Date.now() - startedAt, errors: null, failure: null }
      const result: RunResult = {
        kind: 'sync',
        promptId,
        runLabel,
        status: terminal.status,
        elapsedMs: terminal.durationMs,
        media: terminal.media,
        summary: summarizeMedia(terminal.media),
        seed: null,
        seeds: null,
        ledger: runDir,
        error: terminal.failure ?? (terminal.errors === null && terminal.status !== 'error' ? null : {
          code: 'EXECUTION_FAILED',
          message: firstErrorText(terminal.errors) || (terminal.status === 'interrupted' ? 'interrupted before completion' : 'unknown error'),
          nodeErrors: terminal.errors,
        }),
      }
      return { action: 'run', id, workflowName: saved.name, runLabel, result }
    },
  }
}

/**
 * The runtime one registered tool is wired to.
 *
 * The tools are the only handle a caller gets on the runtime, and the checks
 * that matter most — the submit-time preflight in `ComfyUIRuntime.queue`, for
 * one — live on the runtime rather than in any tool definition, so the self
 * checks cannot reach them from the tool surface alone. Reading the tag
 * `comfyUIToolDefinitions` applies keeps that one source instead of a second
 * runtime assembled in a test harness, which would drift from the real one.
 */
export function introspectComfyUIRuntime(tool: ToolDefinition | { name: string }): ComfyUIRuntime {
  const runtime = (tool as ToolDefinition).runtime
  if (runtime === undefined) throw new Error(`tool "${(tool as { name: string }).name}" is not wired to a runtime`)
  return runtime
}

/**
 * `comfyui_skill`: read and write one workflow's skill pack.
 *
 * The pack is documentation the agent is expected to consult before running a
 * workflow (`comfyui_workflow action: skill`), and this tool is the other half:
 * the agent can also author it — record a pitfall it just hit, add a style
 * reference, lay out its own folders — with the same validation, size caps, and
 * path containment the panel goes through.
 *
 * Destroying a pack is deliberately absent. The files are hand-written by the
 * user and have no other copy, so removing the whole thing stays a panel
 * gesture behind an explicit confirmation; the agent can delete a file it owns
 * but cannot wipe the directory.
 */
function skillDefinition(runtime: ComfyUIRuntime): ToolDefinition {
  return {
    name: 'comfyui_skill',
    description: [
      "Read and write one workflow's skill pack: the SKILL.md the agent reads before running that workflow, plus its reference files, scripts, templates and assets.",
      '`action: list` returns the pack listing (files, sub-directories, byte sizes) and its absolute directory.',
      '`action: read` returns one file (path relative to the pack, e.g. `references/styles.md`); reading `SKILL.md` also satisfies the 必读 gate that blocks `comfyui_workflow action: run` for workflows marked required.',
      '`action: write` creates or overwrites one file (`content`); pass `summary` alongside when writing SKILL.md to set the one-line summary the workflow listing shows. `action: append` adds to the end of an existing file instead — the right choice for recording a newly discovered pitfall without rewriting the document.',
      '`action: mkdir` creates a sub-directory, `action: rename` moves a file within the pack, `action: delete` removes one file (SKILL.md cannot be renamed or deleted).',
      '`action: enable` attaches a pack to a workflow that has none (seeding SKILL.md), and `action: require` toggles whether running that workflow demands the pack be read first.',
      'Write documentation the next agent run will need: when to use the workflow, which parameter values matter, what fails. Keep SKILL.md short and put bulk material in separate files — SKILL.md is loaded whole, the other files only when it points at them.',
    ].join(' '),
    parameters: {
      type: 'object',
      // Closed deliberately, for two reasons that are still true: it documents
      // the exact argument surface, and it is what the PTC SDK rendering
      // (`jsonSchemaToTs`) turns into the tool's input type. It does NOT make
      // a misspelled argument fail: the host never validates a raw-channel
      // `parameters` — not at register time (`dsh-tools` only runs
      // `assertSupportedJsonSchema` on `output.schema`) and not at call time
      // (only the returned value is validated). The real risk runs the other
      // way: if a keyword here is outside the host's supported subset, that
      // PTC rendering throws, the failure is swallowed, and this tool's input
      // type silently degrades to `unknown`. Enforcement of the declared key
      // set lives in `execute` (see `undeclaredArgumentFailure`).
      additionalProperties: false,
      properties: {
        action: {
          type: 'string',
          enum: ['list', 'read', 'write', 'append', 'mkdir', 'rename', 'delete', 'enable', 'require'],
          description: 'list the pack; read/write/append/rename/delete one file; mkdir a sub-directory; enable a pack on a workflow; require toggles the read-before-run gate.',
        },
        workflow_id: { type: 'string', description: 'Workflow id from comfyui_workflow action: list.' },
        path: { type: 'string', description: 'Pack-relative file path for read/write/append/rename/delete, e.g. "SKILL.md" or "references/styles.md". One level of sub-directory only.' },
        content: { type: 'string', description: 'File text for write and append.' },
        summary: { type: 'string', description: 'One-line summary stored in SKILL.md frontmatter; this is the line the workflow listing shows, so make it say when to use the workflow.' },
        to: { type: 'string', description: 'New path for rename (a bare name keeps the current directory).' },
        name: { type: 'string', description: 'Sub-directory name for mkdir (letters, digits, CJK, underscore, dash).' },
        required: { type: 'boolean', description: 'For action: require — true refuses to run the workflow until the pack has been read in this session.' },
      },
      required: ['action', 'workflow_id'],
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render(_args, value) {
        const data = value as {
          action: string
          workflowName?: string
          path?: string
          content?: string
          truncated?: boolean
          dir?: string
          summary?: string
          files?: Array<{ path: string; size: number }>
          dirs?: string[]
          required?: boolean
        }
        if (data.action === 'read') {
          return [{ type: 'text', text: `${data.path} (${data.workflowName}):\n${data.content ?? ''}${data.truncated === true ? '\n…（文件过大，已截断）' : ''}` }]
        }
        const lines: string[] = []
        if (data.action === 'list') {
          lines.push(`技能包 ${data.workflowName}（${data.dir}）${data.required === true ? ' — 运行前必读' : ''}`)
          if (data.summary !== undefined && data.summary !== '') lines.push(`摘要: ${data.summary}`)
          for (const file of data.files ?? []) lines.push(`  ${file.path} (${file.size} B)`)
          const empty = (data.dirs ?? []).filter((dir) => !(data.files ?? []).some((file) => file.path.startsWith(`${dir}/`)))
          if (empty.length > 0) lines.push(`  空目录: ${empty.map((dir) => `${dir}/`).join('、')}`)
          return [{ type: 'text', text: lines.join('\n') }]
        }
        lines.push(`技能包 ${data.workflowName} 已更新（${data.action}${data.path !== undefined ? ` ${data.path}` : ''}）`)
        for (const file of data.files ?? []) lines.push(`  ${file.path} (${file.size} B)`)
        return [{ type: 'text', text: lines.join('\n') }]
      },
      presentationMeta(_args, value) {
        return value
      },
    },
    timeoutMs: TOOL_TIMEOUT_MS,
    async execute(args, exec) {
      const action = args.action
      const id = args.workflow_id
      const undeclaredArgs = undeclaredArgumentFailure(this, args)
      if (undeclaredArgs !== undefined) {
        return { action: 'rejected', workflowId: typeof id === 'string' ? id : '', error: undeclaredArgs }
      }
      if (typeof id !== 'string' || id === '') {
        throw new Error('comfyui_skill: workflow_id is required')
      }
      const saved = await runtime.getWorkflow(id)
      if (saved === undefined) {
        throw new Error(`comfyui_skill: workflow "${id}" not found — run comfyui_workflow action: list first`)
      }
      const path = typeof args.path === 'string' ? args.path : ''
      const content = typeof args.content === 'string' ? args.content : ''

      const listing = async (label: string, extra?: Record<string, unknown>): Promise<Record<string, unknown>> => {
        const pack = await runtime.skillPacks.info(id)
        if (pack === undefined) throw new Error(`comfyui_skill: 工作流 "${saved.name}" 没有技能包 — 先用 action: enable 挂一个`)
        return {
          action: label,
          workflowId: id,
          workflowName: saved.name,
          dir: pack.dir,
          summary: pack.summary,
          required: pack.required,
          files: pack.files.map(({ path: file, size }) => ({ path: file, size })),
          dirs: pack.dirs,
          ...extra,
        }
      }

      if (action === 'enable') {
        const result = await runtime.skillPacks.enable(id)
        if (!result.ok) throw new Error(`comfyui_skill: ${result.error}`)
        return listing('enable')
      }
      if (action === 'list') return listing('list')
      if (action === 'require') {
        const result = await runtime.skillPacks.setRequired(id, args.required === true)
        if (!result.ok) throw new Error(`comfyui_skill: ${result.error}`)
        return listing('require')
      }
      if (action === 'read') {
        if (path === '') throw new Error('comfyui_skill: path is required for action: read')
        const file = await runtime.skillPacks.readFile(id, path)
        if (!file.ok) throw new Error(`comfyui_skill: ${file.error}`)
        // Reading the main document is the same gesture `comfyui_workflow
        // action: skill` performs, so it opens the run gate too.
        if (path === SKILL_MAIN) markSkillPackRead(exec.agent, id)
        const truncated = file.value.length > MAX_TOOL_READ_CHARS
        return {
          action: 'read',
          workflowId: id,
          workflowName: saved.name,
          path,
          content: truncated ? file.value.slice(0, MAX_TOOL_READ_CHARS) : file.value,
          truncated,
        }
      }
      if (action === 'write' || action === 'append') {
        if (path === '') throw new Error(`comfyui_skill: path is required for action: ${action}`)
        let text = content
        if (action === 'append') {
          const existing = await runtime.skillPacks.readFile(id, path)
          const before = existing.ok ? existing.value : ''
          text = before === '' ? content : `${before.replace(/\s*$/, '')}\n\n${content}`
        }
        const summary = typeof args.summary === 'string' ? args.summary : undefined
        const result = path === SKILL_MAIN && summary !== undefined
          ? await runtime.skillPacks.writeFile(id, path, joinFrontmatter(summary, text))
          : await runtime.skillPacks.writeFile(id, path, text)
        if (!result.ok) throw new Error(`comfyui_skill: ${result.error}`)
        return listing(action, { path })
      }
      if (action === 'mkdir') {
        const name = typeof args.name === 'string' ? args.name : ''
        const result = await runtime.skillPacks.makeDir(id, name)
        if (!result.ok) throw new Error(`comfyui_skill: ${result.error}`)
        return listing('mkdir', { path: `${name}/` })
      }
      if (action === 'rename') {
        const to = typeof args.to === 'string' ? args.to : ''
        if (path === '' || to === '') throw new Error('comfyui_skill: path and to are required for action: rename')
        const bucket = path.includes('/') ? `${path.split('/')[0] ?? ''}/` : ''
        const target = to.includes('/') ? to : `${bucket}${to}`
        const result = await runtime.skillPacks.renameFile(id, path, target)
        if (!result.ok) throw new Error(`comfyui_skill: ${result.error}`)
        return listing('rename', { path: target })
      }
      if (action === 'delete') {
        if (path === '') throw new Error('comfyui_skill: path is required for action: delete')
        const result = await runtime.skillPacks.deleteFile(id, path)
        if (!result.ok) throw new Error(`comfyui_skill: ${result.error}`)
        return listing('delete', { path })
      }
      throw new Error(`comfyui_skill: unknown action ${String(action)}`)
    },
  }
}

/** A client pointed at the run's configured server or an explicit override. */
function clientFor(
  runtime: ComfyUIRuntime,
  apiKey: string | undefined,
  override: { host?: unknown; port?: unknown; timeoutMs?: unknown },
): ComfyUIClient {
  const config = runtime.getConfig()
  const host = typeof override.host === 'string' && override.host !== '' ? override.host : undefined
  const port = typeof override.port === 'number' && Number.isFinite(override.port) ? Math.floor(override.port) : undefined
  const baseUrl = host === undefined && port === undefined
    ? config.baseUrl
    : `http://${host ?? '127.0.0.1'}:${port ?? 8188}`
  const connectTimeoutMs = typeof override.timeoutMs === 'number' && Number.isFinite(override.timeoutMs)
    ? Math.max(1, Math.floor(override.timeoutMs))
    : config.connectTimeoutMs
  return new ComfyUIClient(baseUrl, apiKey, connectTimeoutMs, config.maxMediaBytes)
}

/** The declared `ckpt_name` values of the server, or null when undeclared. */
function checkpointNames(objectInfo: Record<string, unknown>): string[] | null {
  const definition = objectInfo['CheckpointLoaderSimple'] as { input?: { required?: Record<string, unknown> } } | undefined
  const spec = definition?.input?.required?.['ckpt_name']
  if (!Array.isArray(spec) || !Array.isArray(spec[0])) return null
  return (spec[0] as unknown[]).filter((value): value is string => typeof value === 'string')
}

/** The first device row of `/system_stats`, if the server reports one. */
function firstDevice(stats: unknown): { name: string | null; type: string | null; vramFree: number | null } | null {
  if (typeof stats !== 'object' || stats === null) return null
  const devices = (stats as { devices?: unknown }).devices
  if (!Array.isArray(devices) || devices.length === 0) return null
  const device = devices[0]
  if (typeof device !== 'object' || device === null) return null
  const record = device as Record<string, unknown>
  return {
    name: typeof record['name'] === 'string' ? record['name'] : null,
    type: typeof record['type'] === 'string' ? record['type'] : null,
    vramFree: typeof record['vram_free'] === 'number' ? record['vram_free'] : null,
  }
}

/**
 * `comfyui_probe`: the capability gate to run before submitting anything.
 *
 * One call answers "is the server up, what is it, and can it load the models I
 * am about to name": readiness from `/system_stats`, the device/VRAM row, the
 * `CheckpointLoaderSimple` model list, and the queue backlog. An unreachable or
 * non-ComfyUI server is a normal answer (`ready: false` plus a readable
 * `{code, message, status}`) rather than a thrown error, so a caller can gate
 * on it without catching.
 */
function probeDefinition(runtime: ComfyUIRuntime): ToolDefinition {
  return {
    name: 'comfyui_probe',
    description: [
      'Probe the configured ComfyUI server before submitting anything: readiness (/system_stats), device and free VRAM, the checkpoint models the server can actually load (CheckpointLoaderSimple), and the queue backlog.',
      'Returns structured JSON. A server that is down, wrong, or too old answers `ready: false` with a readable `error` ({code, message, status}) instead of failing.',
      'Use it as the pre-submit gate: a workflow naming a checkpoint that is not in `ckpts` will be refused by comfyui_run, and this call is how you find that out (and what to tell the user to install) before a run.',
    ].join(' '),
    parameters: {
      type: 'object',
      // Closed deliberately, for two reasons that are still true: it documents
      // the exact argument surface, and it is what the PTC SDK rendering
      // (`jsonSchemaToTs`) turns into the tool's input type. It does NOT make
      // a misspelled argument fail: the host never validates a raw-channel
      // `parameters` — not at register time (`dsh-tools` only runs
      // `assertSupportedJsonSchema` on `output.schema`) and not at call time
      // (only the returned value is validated). The real risk runs the other
      // way: if a keyword here is outside the host's supported subset, that
      // PTC rendering throws, the failure is swallowed, and this tool's input
      // type silently degrades to `unknown`. Enforcement of the declared key
      // set lives in `execute` (see `undeclaredArgumentFailure`).
      additionalProperties: false,
      properties: {
        host: { type: 'string', description: 'Override the target host (default: the configured ComfyUI server).' },
        port: { type: 'integer', description: 'Override the target port (default: the configured ComfyUI server, 8188).' },
        timeoutMs: { type: 'integer', description: 'Per-request timeout in ms (default: the configured connectTimeoutMs).' },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          ready: { type: 'boolean' },
          comfyVersion: { oneOf: [{ type: 'string' }, { type: 'null' }] },
          device: {
            oneOf: [
              {
                type: 'object',
                additionalProperties: false,
                properties: {
                  name: { oneOf: [{ type: 'string' }, { type: 'null' }] },
                  type: { oneOf: [{ type: 'string' }, { type: 'null' }] },
                  vramFree: { oneOf: [{ type: 'integer' }, { type: 'null' }] },
                },
              },
              { type: 'null' },
            ],
          },
          ckpts: { oneOf: [{ type: 'array', items: { type: 'string' } }, { type: 'null' }] },
          queueLength: { oneOf: [{ type: 'integer' }, { type: 'null' }] },
          baseUrl: { type: 'string' },
          ts: { type: 'string' },
          error: {
            oneOf: [
              {
                type: 'object',
                additionalProperties: false,
                properties: {
                  code: { type: 'string' },
                  message: { type: 'string' },
                  status: { oneOf: [{ type: 'integer' }, { type: 'null' }] },
                },
              },
              { type: 'null' },
            ],
          },
        },
      },
      render(_args, value) {
        const data = value as {
          ready: boolean
          comfyVersion: string | null
          device: { name: string | null; type: string | null; vramFree: number | null } | null
          ckpts: string[] | null
          queueLength: number | null
          baseUrl: string
          error: { code: string; message: string; status: number | null } | null
        }
        if (!data.ready) {
          return [{ type: 'text', text: `ComfyUI 不可用（${data.baseUrl}）: ${data.error?.code ?? 'ERROR'} ${data.error?.message ?? ''}` }]
        }
        const lines = [
          `ComfyUI ready: ${data.baseUrl}${data.comfyVersion !== null ? ` (v${data.comfyVersion})` : ''}`,
          `device: ${data.device?.name ?? 'unknown'}${data.device?.type !== null && data.device?.type !== undefined ? ` [${data.device.type}]` : ''}${data.device?.vramFree !== null && data.device?.vramFree !== undefined ? `, vram free ${data.device.vramFree}` : ''}`,
          `queue: ${data.queueLength ?? 0} task(s)`,
          `checkpoints (${data.ckpts?.length ?? 0}): ${(data.ckpts ?? []).slice(0, 12).join(', ') || 'none'}`,
        ]
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    timeoutMs: 120_000,
    async execute(args) {
      // `output.schema` here is closed and has no refusable shape, so the
      // refusal is thrown with the same code in it rather than faked as an
      // unreachable server — a probe that never ran is not a probe result.
      const undeclaredArgs = undeclaredArgumentFailure(this, args)
      if (undeclaredArgs !== undefined) throw new Error(undeclaredArgs.message)
      const client = clientFor(runtime, await runtime.getApiKey(), args)
      const ts = new Date().toISOString()
      try {
        const stats = await client.systemStats()
        const objectInfo = await client.objectInfo()
        const queue = await client.getQueue()
        const version = (stats as { system?: { comfyui_version?: unknown } } | undefined)?.system?.comfyui_version
        return {
          ready: true,
          comfyVersion: typeof version === 'string' ? version : null,
          device: firstDevice(stats),
          ckpts: checkpointNames(objectInfo),
          queueLength: queue.queue_running.length + queue.queue_pending.length,
          baseUrl: client.baseUrl,
          ts,
          error: null,
        }
      } catch (error) {
        // An unreachable server is an answer, not an exception: the probe
        // exists to be run *before* committing to a run.
        const status = error instanceof ComfyUIError ? error.status ?? null : null
        return {
          ready: false,
          comfyVersion: null,
          device: null,
          ckpts: null,
          queueLength: null,
          baseUrl: client.baseUrl,
          ts,
          error: {
            code: error instanceof ComfyUIError ? 'UNREACHABLE' : 'ERROR',
            message: (error instanceof Error ? error.message : String(error)).slice(0, 300),
            status,
          },
        }
      }
    },
  }
}

/**
 * `comfyui_fetch_output`: bring a run's media onto this machine.
 *
 * `comfyui_run` returns proxy URLs, which serve the media out of ComfyUI's
 * output directory and stop working the moment that file is moved or the
 * history entry is cleared. This tool makes a durable local copy instead, named
 * through `uniqueOutputPath` so a repeated fetch never overwrites an earlier
 * one, and records the absolute paths on the same ledger row as the run that
 * produced them.
 *
 * Two addressing modes, exactly one of which is required (the pairing cannot
 * be expressed in the parameters schema — the host's schema subset allows
 * neither a root `oneOf` beside `properties` nor a conditional — so the
 * check lives in `execute`, and the description states it).
 */
function fetchOutputDefinition(runtime: ComfyUIRuntime): ToolDefinition {
  return {
    name: 'comfyui_fetch_output',
    description: [
      'Download a completed run\'s media from ComfyUI onto this machine and record the paths in the run ledger.',
      'Address the media EITHER by `promptId` (every output of that prompt, via GET /history/<promptId>) OR by `filename` (one file, using `subfolder` and `type`); passing both, or neither, is refused.',
      'Files land in `targetDir` (default: the configured download directory) and never overwrite an existing file — the stem is bumped (`image.png` → `image.01.png`). The absolute paths and sizes are written onto the matching `runs.json` row, so a run and its local copies stay linked.',
    ].join(' '),
    parameters: {
      type: 'object',
      // Closed deliberately, for two reasons that are still true: it documents
      // the exact argument surface, and it is what the PTC SDK rendering
      // (`jsonSchemaToTs`) turns into the tool's input type. It does NOT make
      // a misspelled argument fail: the host never validates a raw-channel
      // `parameters` — not at register time (`dsh-tools` only runs
      // `assertSupportedJsonSchema` on `output.schema`) and not at call time
      // (only the returned value is validated). The real risk runs the other
      // way: if a keyword here is outside the host's supported subset, that
      // PTC rendering throws, the failure is swallowed, and this tool's input
      // type silently degrades to `unknown`. Enforcement of the declared key
      // set lives in `execute` (see `undeclaredArgumentFailure`).
      additionalProperties: false,
      properties: {
        promptId: { type: 'string', description: 'Download every output of this prompt (from GET /history/<promptId>). Alternative to `filename`.' },
        filename: { type: 'string', description: 'Download this one file. Alternative to `promptId`; pair it with `subfolder`/`type` when the file is not at the output root.' },
        subfolder: { type: 'string', description: 'Subfolder the file lives in (default: the output root).' },
        type: { type: 'string', description: 'ComfyUI file type/context of the file (default: output).' },
        targetDir: { type: 'string', description: 'Directory to download into (default: the configured download directory).' },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          files: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                filename: { type: 'string' },
                subfolder: { type: 'string' },
                type: { type: 'string' },
                absPath: { type: 'string' },
                size: { type: 'integer' },
              },
            },
          },
          targetDir: { type: 'string' },
          promptId: { oneOf: [{ type: 'string' }, { type: 'null' }] },
          ledger: {
            type: 'object',
            additionalProperties: false,
            properties: {
              file: { type: 'string' },
              records: { type: 'integer' },
              note: { oneOf: [{ type: 'string' }, { type: 'null' }] },
            },
          },
          error: {
            oneOf: [
              {
                type: 'object',
                additionalProperties: false,
                properties: {
                  code: { type: 'string' },
                  message: { type: 'string' },
                },
              },
              { type: 'null' },
            ],
          },
        },
      },
      render(_args, value) {
        const data = value as {
          files: Array<{ absPath: string; size: number }>
          targetDir: string
          ledger: { file: string; records: number; note: string | null }
          error: { code: string; message: string } | null
        }
        if (data.error !== null) {
          return [{ type: 'text', text: `comfyui_fetch_output 失败 ${data.error.code}: ${data.error.message}` }]
        }
        const lines = [`下载 ${data.files.length} 个文件到 ${data.targetDir}:`]
        for (const file of data.files) lines.push(`  ${file.absPath} (${file.size} B)`)
        lines.push(`台账: ${data.ledger.file}（${data.ledger.records} 条${data.ledger.note !== null ? `；${data.ledger.note}` : ''}）`)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    timeoutMs: 600_000,
    async execute(args) {
      const hasPrompt = typeof args.promptId === 'string' && args.promptId !== ''
      const hasFilename = typeof args.filename === 'string' && args.filename !== ''
      const runDir = typeof args.targetDir === 'string' && args.targetDir !== ''
        ? resolvePath(args.targetDir)
        : runtime.downloadDir()
      const fail = (code: string, message: string, promptId: string | null): Record<string, unknown> => ({
        files: [],
        targetDir: runDir,
        promptId,
        ledger: { file: join(runDir, 'runs.json'), records: 0, note: null },
        error: { code, message },
      })
      // The refusal rides the existing structured error shape, so a caller sees
      // one error vocabulary (`code` + `message`) for both a bad target and an
      // argument the tool never declared.
      const undeclaredArgs = undeclaredArgumentFailure(this, args)
      if (undeclaredArgs !== undefined) {
        return fail(undeclaredArgs.code, undeclaredArgs.message, null)
      }
      if (hasPrompt === hasFilename) {
        return fail('BAD_ARGS', 'promptId 与 filename 二选一：给且仅给一个', null)
      }
      const client = runtime.createClient(await runtime.getApiKey())

      let sources: Array<{ filename: string; subfolder: string; type: string }>
      let promptId: string
      if (hasPrompt) {
        promptId = String(args.promptId)
        const entry = await client.getHistory(promptId)
        if (entry === undefined) return fail('NO_HISTORY', `history 无记录: ${promptId}（prompt 未提交或已被清除）`, promptId)
        sources = collectMedia({ promptId, entry, maxItems: Number.MAX_SAFE_INTEGER, proxyBase: undefined })
          .map(({ filename, subfolder, type }) => ({ filename, subfolder, type }))
        if (sources.length === 0) return fail('NO_OUTPUTS', `history ${promptId} 无媒体输出`, promptId)
      } else {
        const filename = String(args.filename)
        promptId = `view:${filename}`
        sources = [{
          filename,
          subfolder: typeof args.subfolder === 'string' ? args.subfolder : '',
          type: typeof args.type === 'string' ? args.type : 'output',
        }]
      }

      const files: Array<{ filename: string; subfolder: string; type: string; absPath: string; size: number }> = []
      try {
        mkdirSync(runDir, { recursive: true })
        for (const source of sources) {
          const buffer = await client.fetchView(source)
          const relative = safeRelativePath(source.subfolder !== '' ? `${source.subfolder}/${source.filename}` : source.filename)
          const absPath = uniqueOutputPath(runDir, relative)
          writeFileSync(absPath, buffer.bytes)
          files.push({ ...source, absPath, size: buffer.bytes.byteLength })
        }
      } catch (error) {
        return fail('FETCH_FAILED', error instanceof Error ? error.message : String(error), promptId)
      }

      const ledger = upsertRun(runDir, {
        promptId,
        ts: new Date().toISOString(),
        status: 'completed',
        files,
      })
      return {
        files,
        targetDir: runDir,
        promptId,
        ledger: { file: join(runDir, 'runs.json'), records: ledger.records, note: ledger.note },
        error: null,
      }
    },
  }
}

/**
 * Every tool this plugin registers, in registration order.
 *
 * The list is exported so the schema self-check can enumerate the tools from
 * the same source that registers them: a hand-written list in the check would
 * quietly stop covering a tool the moment one was added.
 */
export function comfyUIToolDefinitions(ctx: Context, runtime: ComfyUIRuntime): ToolDefinition[] {
  const definitions: ToolDefinition[] = [
    runDefinition(runtime, ctx),
    objectInfoDefinition(runtime),
    workflowDefinition(runtime, ctx),
    skillDefinition(runtime),
    probeDefinition(runtime),
    fetchOutputDefinition(runtime),
  ]
  // One runtime, one source: tagging the definitions here beats re-deriving
  // the runtime in every caller that needs it (tools, routes, self-checks).
  for (const definition of definitions) definition.runtime = runtime
  return definitions
}

/** Register the plugin tools; returns disposers. */
export function registerComfyUITools(ctx: Context, runtime: ComfyUIRuntime): Array<() => void> {
  const tools = (ctx as unknown as { tools: { register(definition: ToolDefinition): () => void } }).tools
  const disposers: Array<() => void> = []
  // One bad definition must not take the whole loader chain down: the registry
  // validates `output.schema` at register time and throws on an unsupported
  // shape, which is exactly the failure this plugin used to escalate into a
  // dead host. Each registration is isolated; a failure is reported and the
  // remaining tools still go in.
  const register = (definition: ToolDefinition): void => {
    try {
      disposers.push(tools.register(definition))
    } catch (error) {
      console.warn(`[dsh-comfyui] 工具 "${definition.name}" 注册失败，已跳过: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  for (const definition of comfyUIToolDefinitions(ctx, runtime)) register(definition)
  return disposers
}
