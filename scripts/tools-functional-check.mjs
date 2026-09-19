#!/usr/bin/env node
/**
 * Functional smoke check for the four hardening items, against a MOCK ComfyUI
 * server on loopback (never the real host, never 3080).
 *
 * The schema self-check proves the tools are *shaped* correctly; this proves
 * they *behave*: the probe's ready/unreachable branches, the submit-time
 * preflight refusing a missing model without queueing anything, the ledger's
 * queued → terminal same-key overwrite with a written-down seed, and the
 * fetch path's non-overwriting file layout.
 *
 * Usage: node scripts/tools-functional-check.mjs
 * Exit 0 = every assertion held.
 */
import { createServer } from 'node:http'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const DSH_TOOLS = 'file:///D:/dsh/npm/0.1.6-alpha.1/node_modules/@deepseek-ai/dsh-tools/lib/index.js'
const { jsonSchemaToTs } = await import(DSH_TOOLS)
const { ComfyUIClient } = await import('../lib/comfyui.js')
const { comfyUIToolDefinitions, introspectComfyUIRuntime } = await import('../lib/tools.js')
const { readLedger } = await import('../lib/ledger.js')
const { apply: applyPlugin } = await import('../lib/index.js')

const failures = []
const checks = []
function assert(label, condition, detail = '') {
  checks.push({ label, ok: Boolean(condition), detail })
  if (!condition) failures.push(`${label}${detail !== '' ? ` — ${detail}` : ''}`)
}

/** First element of a tool's `files` array, or undefined — keeps a failed
 * expectation from turning into a TypeError that hides the real reason. */
function firstFile(result) {
  return Array.isArray(result?.files) ? result.files[0] : undefined
}

/** Media the mock server serves through /view. */
const MEDIA = Buffer.from('dsh-comfyui-smoke-bytes')
const OBJECT_INFO = {
  CheckpointLoaderSimple: { input: { required: { ckpt_name: [['sd_xl_base_1.0.safetensors']] } } },
  KSampler: { input: { required: { seed: ['INT', { default: 0 }] } } },
  SaveImage: { input: { required: { filename_prefix: ['STRING', { default: 'dsh-comfyui' }] } } },
}
// The built-in templates use core nodes beyond the ones the assertions below
// care about; they are declared bare so the preflight passes them without
// inventing option lists it would then have to satisfy.
for (const classType of ['EmptyLatentImage', 'CLIPTextEncode', 'VAEDecode', 'LoadImage', 'VAEEncode',
  'UNETLoader', 'CLIPLoader', 'VAELoader', 'WanTextEncode', 'WanImageToVideo', 'WanVideoDecode', 'SaveVideo']) {
  OBJECT_INFO[classType] = {}
}
let queueSubmissions = 0

const server = createServer((request, response) => {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  const json = (body) => {
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(body))
  }
  if (url.pathname === '/system_stats') {
    return json({ system: { comfyui_version: '0.3.99-smoke' }, devices: [{ name: 'cuda:0 NVIDIA Smoke', type: 'cuda', vram_free: 12_345 }] })
  }
  if (url.pathname === '/object_info') return json(OBJECT_INFO)
  if (url.pathname === '/queue') return json({ queue_running: [], queue_pending: [[1, 'pending-prompt']] })
  if (url.pathname === '/prompt') {
    queueSubmissions += 1
    return json({ prompt_id: `mock-prompt-${queueSubmissions}` })
  }
  if (url.pathname.startsWith('/history/')) {
    const promptId = decodeURIComponent(url.pathname.slice('/history/'.length))
    // A prompt the mock never produced has no outputs, which is how the
    // NO_OUTPUTS path is exercised.
    const outputs = promptId === 'plain-prompt'
      ? {}
      : { '9': { images: [{ filename: 'out.png', subfolder: '', type: 'output' }] } }
    return json({
      [promptId]: {
        status: { status_str: 'success', completed: true },
        outputs,
        // A real history entry carries the prompt it ran, which is what the
        // panel's rerun action reads back.
        prompt: badCheckpointWorkflow(1),
      },
    })
  }
  if (url.pathname === '/view') {
    response.writeHead(200, { 'content-type': 'image/png' })
    return response.end(MEDIA)
  }
  response.writeHead(404, { 'content-type': 'application/json' })
  response.end(JSON.stringify({ error: `unexpected ${url.pathname}` }))
})

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const port = server.address().port
const downloads = mkdtempSync(join(tmpdir(), 'dsh-comfyui-smoke-'))

function toolByName(name, runtime) {
  const tool = comfyUIToolDefinitions({ get: () => undefined }, runtime).find((entry) => entry.name === name)
  if (tool === undefined) throw new Error(`tool ${name} not registered`)
  return tool
}

/**
 * The plugin's REAL submit throat, mounted from `lib/index.js`.
 *
 * `runtime.queue` is where the preflight now lives, and the routes call it
 * directly, so the third path and the parameter-rewrite path can only be
 * exercised against this object, not against the mock runtime above (whose
 * `queue` is a stub). Returns the runtime plus the disposers that stop the
 * progress socket and the route mounts.
 */
async function mountPlugin(baseUrl) {
  const dataDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-mount-'))
  const disposers = []
  const capturedRuntime = { value: undefined }
  const tools = []
  const routes = []
  // The fiber the plugin mounts its routes and media proxy on. In cordis the
  // sub-fiber reaches the isolated service through its own `get`, so the stub
  // has to serve it there too — that is the path `mountComfyUIRoutes` uses.
  const webServerStub = {
    register(route) { routes.push(route); return () => {} },
    tapIndex: () => () => {},
  }
  const webCtx = {
    get: (name) => (name === 'webServer' ? webServerStub : undefined),
    effect(callback) {
      const dispose = callback()
      if (typeof dispose === 'function') disposers.push(dispose)
      return () => {}
    },
    logger: { warn() {}, info() {}, error() {} },
  }
  const ctx = {
    // The plugin reads the registry as `ctx.tools` (its cordis inject also
    // declares the name), so the harness provides the property itself. The
    // runtime is not on the definition object, so it is read back through the
    // module's own introspection door (the same runtime the tools close over).
    tools: {
      register(tool) {
        tools.push(tool)
        return () => {}
      },
    },
    get(name) {
      if (name === 'webServer') return webServerStub
      return undefined
    },
    effect(callback) {
      const dispose = callback()
      if (typeof dispose === 'function') disposers.push(dispose)
      return () => {}
    },
    inject(names, callback) {
      if (Array.isArray(names) && names.includes('webServer')) callback(webCtx)
    },
    logger: { warn() {}, info() {}, error() {} },
  }
  await applyPlugin(ctx, { baseUrl, dataDir, downloadDir: join(dataDir, 'runs'), pollIntervalMs: 50, timeoutMs: 20_000, connectTimeoutMs: 5_000 })
  const runtime = introspectComfyUIRuntime(tools[0])
  const toolFor = (name) => tools.find((tool) => tool.name === name)
  const routeOf = (path) => routes.find((route) => route.path === path)?.handler
  return { runtime, toolFor, dataDir, routeOf, routePaths: () => routes.map((route) => route.path), dispose: () => { for (const dispose of disposers) dispose() } }
}

/** A request/response pair for invoking a mounted route handler directly. */
function fakeRouteCall(body) {
  const request = {
    method: 'POST',
    url: '/',
    headers: { host: '127.0.0.1' },
    async *[Symbol.asyncIterator]() {
      yield Buffer.from(JSON.stringify(body), 'utf8')
    },
  }
  const captured = { status: 0, body: null }
  const response = {
    writeHead(status) { captured.status = status },
    end(payload) { captured.body = payload === undefined ? null : JSON.parse(String(payload)) },
  }
  return { request, response, captured }
}

/** One `POST` of a route handler mounted on a stub web server. */
async function callRoute(handler, body) {
  const { request, response, captured } = fakeRouteCall(body)
  await handler(request, response)
  return captured
}

/** A workflow whose KSampler seed and missing checkpoint are both addressable. */
function badCheckpointWorkflow(seed) {
  return {
    '3': { class_type: 'KSampler', inputs: { seed, steps: 20, cfg: 6.5 } },
    '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'missing.safetensors' } },
    '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'dsh-comfyui' } },
  }
}

function runtimeFor(baseUrl, downloadDir) {
  return {
    getConfig: () => ({
      baseUrl, apiKeyEnv: 'COMFYUI_API_KEY', connectTimeoutMs: 5_000, timeoutMs: 20_000,
      pollIntervalMs: 50, maxMediaItems: 12, maxMediaBytes: 64 * 1024 * 1024,
      dataDir: downloads, maxAssets: 200, skillsDir: '', mediaHost: '', outputDir: '',
      downloadDir, comfyuiDirs: [],
    }),
    getApiKey: async () => undefined,
    createClient: (apiKey) => new ComfyUIClient(baseUrl, apiKey, 5_000, 64 * 1024 * 1024),
    hostHint: { origin: () => undefined, remember: () => {} },
    proxyBase: () => undefined,
    downloadDir: () => downloadDir,
    settingsWritable: () => false,
    updateConfig: async () => ({ ok: false, error: 'smoke' }),
    queue: async (workflow) => {
      queueSubmissions += 1
      submissions.push(workflow)
      return `mock-prompt-${queueSubmissions}`
    },
    untrack: () => {},
    trackedRuns: () => [],
    queueProgress: () => undefined,
    listWorkflows: async () => [],
    getWorkflow: async () => undefined,
    saveWorkflow: async () => ({ ok: false, error: 'smoke' }),
    deleteWorkflow: async () => false,
    skillPacks: {},
    refreshVoiceLibrary: async () => false,
    listMediaSizes: async () => ({}),
    saveMediaSize: async () => {},
    lookupMediaHash: async () => undefined,
    saveMediaHash: async () => {},
    loadSlots: async () => [],
    saveSlots: async () => {},
    listAssets: async () => [],
    deleteAsset: async () => undefined,
    sweep: async () => [],
    listComfyWorkflows: async () => [],
    getComfyWorkflow: async () => undefined,
    analyzeComfyWorkflow: async () => ({ ok: false, error: 'smoke' }),
    extractComfyWorkflow: async () => ({ ok: false, error: 'smoke' }),
  }
}

const runtime = runtimeFor(`http://127.0.0.1:${port}`, downloads)
const exec = { signal: new AbortController().signal }

// --- D1: probe, ready branch -------------------------------------------------
const probe = await toolByName('comfyui_probe', runtime).execute({}, exec)
assert('D1 probe ready', probe.ready === true, JSON.stringify(probe.error))
assert('D1 probe version/device/ckpts/queue', probe.comfyVersion === '0.3.99-smoke'
  && probe.device?.name === 'cuda:0 NVIDIA Smoke'
  && Array.isArray(probe.ckpts) && probe.ckpts[0] === 'sd_xl_base_1.0.safetensors'
  && probe.queueLength === 1, JSON.stringify({ v: probe.comfyVersion, d: probe.device, c: probe.ckpts, q: probe.queueLength }))

// --- D1: probe, unreachable branch (an answer, not a throw) ------------------
const deadProbe = await toolByName('comfyui_probe', runtimeFor('http://127.0.0.1:1', downloads)).execute({}, exec)
assert('D1 probe unreachable returns ready:false with readable error',
  deadProbe.ready === false && typeof deadProbe.error?.code === 'string' && deadProbe.error.message.length > 0,
  JSON.stringify(deadProbe.error))

// --- D4: preflight refuses a missing checkpoint, and queues nothing ----------
const submissions = []
const before = queueSubmissions
const refused = await toolByName('comfyui_run', runtime).execute({
  template: 'txt2img',
  inputs: { '4': { ckpt_name: 'not-on-disk.safetensors' } },
  run_label: 'dsh-comfyui-fork-hardening-harden-9001',
}, exec)
assert('D4 missing model refused with PREFLIGHT', refused.status === 'error' && refused.error?.code === 'PREFLIGHT', JSON.stringify(refused.error))
assert('D4 refusal names the missing value', JSON.stringify(refused.error?.nodeErrors ?? []).includes('not-on-disk.safetensors'), JSON.stringify(refused.error?.nodeErrors))
assert('D4 nothing was queued', queueSubmissions === before, `queueSubmissions ${before} -> ${queueSubmissions}`)
const refusedLedger = readLedger(downloads).records.find((entry) => entry.runLabel === 'dsh-comfyui-fork-hardening-harden-9001')
assert('D4 refusal recorded as failed row (keyed by runLabel)', refusedLedger?.status === 'failed' && refusedLedger.promptId === undefined,
  JSON.stringify(refusedLedger))

// --- D4: unknown class type is refused too -----------------------------------
const unknownClass = await toolByName('comfyui_run', runtime).execute({
  workflow: { '1': { class_type: 'NoSuchNode', inputs: {} } },
  run_label: 'noclass-0001',
}, exec)
assert('D4 unknown class refused', unknownClass.error?.code === 'PREFLIGHT'
  && JSON.stringify(unknownClass.error.nodeErrors).includes('UNKNOWN_CLASS'), JSON.stringify(unknownClass.error))

// --- D5 + D6: queued -> terminal same-key overwrite, seed written down -------
const runArgs = { template: 'txt2img', run_label: 'dsh-comfyui-fork-hardening-harden-9002', inputs: { '6': { text: 'smoke' } } }
const first = await toolByName('comfyui_run', runtime).execute(runArgs, exec)
assert('D5 run completed', first.status === 'completed', JSON.stringify(first.error))
assert('D6 seed written down (not -1, not null)', typeof first.seed === 'number' && first.seed >= 0 && first.seed !== -1, String(first.seed))
const submittedSeeds = submissions.map((workflow) => workflow['3']?.inputs?.seed)
assert('D6 submitted graph carries the recorded seed', submittedSeeds.length > 0 && submittedSeeds.every((seed) => seed === first.seed), JSON.stringify({ actual: submittedSeeds, recorded: first.seed }))

const afterFirst = readLedger(downloads)
const rowCount = afterFirst.records.filter((entry) => entry.runLabel === runArgs.run_label).length
const row = afterFirst.records.find((entry) => entry.runLabel === runArgs.run_label)
assert('D5 one row per run (not queued + terminal)', rowCount === 1, `rows=${rowCount}`)
assert('D5 row keyed by runLabel and closed at terminal status', row?.runLabel === runArgs.run_label && row.status === 'completed', JSON.stringify(row))
assert('D5 same row carries the promptId after submit', typeof row?.promptId === 'string' && row.promptId.startsWith('mock-prompt-'), String(row?.promptId))
assert('D6 ledger row carries the actual seed', row?.seed === first.seed, JSON.stringify({ ledger: row?.seed, result: first.seed }))

// --- D5: sequence increments under one prefix --------------------------------
const auto = runtimeFor(`http://127.0.0.1:${port}`, downloads)
const autoRunA = await toolByName('comfyui_run', auto).execute({ template: 'txt2img' }, exec)
const autoRunB = await toolByName('comfyui_run', auto).execute({ template: 'txt2img' }, exec)
assert('D5 default runLabel is <prefix>-<NNNN> and increments',
  /-0001$/.test(autoRunA.runLabel) && /-0002$/.test(autoRunB.runLabel),
  `${autoRunA.runLabel} / ${autoRunB.runLabel}`)

// --- D5: corrupt ledger is tolerated ----------------------------------------
const corruptDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-corrupt-'))
const { writeFileSync } = await import('node:fs')
writeFileSync(join(corruptDir, 'runs.json'), '{ this is not json', 'utf8')
const corruptRun = await toolByName('comfyui_run', runtimeFor(`http://127.0.0.1:${port}`, corruptDir)).execute({ template: 'txt2img' }, exec)
// Reading the ledger AFTER the run only sees the valid table the tool wrote,
// so the tolerance claim is sampled on a freshly corrupted copy instead.
const noteDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-corrupt-read-'))
writeFileSync(join(noteDir, 'runs.json'), '{ this is not json', 'utf8')
const corruptRead = readLedger(noteDir)
assert('D5 corrupt ledger degrades to empty table with a note',
  corruptRead.note !== null && corruptRead.records.length === 0, JSON.stringify(corruptRead))
rmSync(noteDir, { recursive: true, force: true })
assert('D5 corrupt ledger does not break the run (rewritten as a valid table)',
  corruptRun.status === 'completed' && readLedger(corruptDir).note === null && readLedger(corruptDir).records.length >= 1,
  JSON.stringify({ status: corruptRun.status, error: corruptRun.error, ledger: readLedger(corruptDir) }))

// --- D6: an explicit seed is honoured and replays identically ---------------
const replayArgs = (label) => ({ template: 'txt2img', seed: 424242, run_label: label })
submissions.length = 0
const replayA = await toolByName('comfyui_run', runtime).execute(replayArgs('replay-0001'), exec)
const replayB = await toolByName('comfyui_run', runtime).execute(replayArgs('replay-0002'), exec)
const replaySeeds = submissions.map((workflow) => workflow['3']?.inputs?.seed)
assert('D6 explicit seed is written through, never overridden',
  replayA.seed === 424242 && replayB.seed === 424242 && replaySeeds.every((seed) => seed === 424242),
  JSON.stringify({ result: [replayA.seed, replayB.seed], submitted: replaySeeds }))
const replayRows = readLedger(downloads).records.filter((entry) => (entry.runLabel ?? '').startsWith('replay-'))
assert('D6 the replay seed is on both ledger rows (落盘即可复跑)',
  replayRows.length === 2 && replayRows.every((entry) => entry.seed === 424242),
  JSON.stringify(replayRows.map((entry) => ({ runLabel: entry.runLabel, seed: entry.seed }))))

// --- D1: fetch by promptId, then again (no overwrite) ------------------------
const fetchTool = toolByName('comfyui_fetch_output', runtime)
const fetchDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-fetch-'))
const fetch1 = await fetchTool.execute({ promptId: String(row.promptId), targetDir: fetchDir }, exec)
const fetch2 = await fetchTool.execute({ promptId: String(row.promptId), targetDir: fetchDir }, exec)
const file1 = firstFile(fetch1)
const file2 = firstFile(fetch2)
assert('D1 fetch by promptId writes a file', file1 !== undefined && existsSync(file1.absPath), JSON.stringify(fetch1.error ?? fetch1.files))
assert('D1 fetched bytes match the server', file1 !== undefined && readFileSync(file1.absPath).equals(MEDIA))
assert('D5 second fetch does not overwrite (stem.NN)',
  file1 !== undefined && file2 !== undefined && file1.absPath !== file2.absPath && existsSync(file1.absPath) && existsSync(file2.absPath),
  `${file1?.absPath} / ${file2?.absPath}`)
const fetchByName = await fetchTool.execute({ filename: 'out.png', subfolder: '', type: 'output', targetDir: fetchDir }, exec)
assert('D1 fetch by filename writes a file', firstFile(fetchByName) !== undefined && existsSync(firstFile(fetchByName).absPath), JSON.stringify(fetchByName.error ?? fetchByName.files))
const bothArgs = await fetchTool.execute({ promptId: 'x', filename: 'y', targetDir: fetchDir }, exec)
assert('D1 promptId/filename exclusivity enforced in execute', bothArgs.error?.code === 'BAD_ARGS', JSON.stringify(bothArgs.error))
const noArgs = await fetchTool.execute({ targetDir: fetchDir }, exec)
assert('D1 neither promptId nor filename refused', noArgs.error?.code === 'BAD_ARGS', JSON.stringify(noArgs.error))
const noOutputs = await fetchTool.execute({ promptId: 'plain-prompt', targetDir: fetchDir }, exec)
assert('D1 a prompt with no media answers NO_OUTPUTS', noOutputs.error?.code === 'NO_OUTPUTS', JSON.stringify(noOutputs.error))
const fetchLedger = readLedger(fetchDir).records.find((entry) => entry.promptId === row.promptId)
assert('D1 fetch records absPath/size on the ledger row',
  Array.isArray(fetchLedger?.files) && fetchLedger.files[0]?.absPath === file2?.absPath && fetchLedger.files[0]?.size === MEDIA.length,
  JSON.stringify({ ledger: fetchLedger?.files, lastFetch: file2?.absPath }))
assert('D1 both fetches are recorded on the same row (no new row per fetch)',
  readLedger(fetchDir).records.filter((entry) => entry.promptId === row.promptId).length === 1,
  JSON.stringify(readLedger(fetchDir).records.map((entry) => entry.promptId)))

// =============================================================================
// 返工新增用例（N-1…N-14）：B-1 三条提交路径 / B-2 并发占号 / B-3 异步缺席 /
// B-4 两侧零违禁键 / F-1 参数白名单。每条都先红后绿可复现（见 exec/fix.md）。
// =============================================================================

// --- N-1/N-2/N-6 + B-1: the submit throat itself refuses ---------------------
// `comfyui_workflow action: run` and the panel routes reach `/prompt` only
// through `runtime.queue`, so this is the check that covers them: a bad graph
// handed to the throat must refuse with the same code and missing-value set as
// the tool-level preflight, and nothing may reach the server.
const badGraph = {
  '3': { class_type: 'KSampler', inputs: { seed: 1, steps: 20, cfg: 6.5 } },
  '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'missing.safetensors' } },
}
// The mock runtime above has a stub `queue`; the checks below need the plugin's
// own submit throat, so they mount it first.
const mounted = await mountPlugin(`http://127.0.0.1:${port}`)
const throatRuntime = mounted.runtime
const throatSubmissionsBefore = queueSubmissions
let throatRefusal
try {
  await throatRuntime.queue(badGraph, { workflowName: 'throat', source: 'smoke' })
  throatRefusal = null
} catch (error) {
  throatRefusal = error
}
assert('N-2 submit throat refuses a missing model (PREFLIGHT)',
  throatRefusal !== null && throatRefusal.code === 'PREFLIGHT' && throatRefusal.failure?.code === 'PREFLIGHT',
  JSON.stringify({ code: throatRefusal?.code, message: throatRefusal?.message }))
assert('N-2 throat refusal names the missing value and the available ones',
  JSON.stringify(throatRefusal?.failure?.nodeErrors ?? []).includes('missing.safetensors')
  && /sd_xl_base_1\.0\.safetensors/.test(throatRefusal?.failure?.message ?? ''),
  JSON.stringify(throatRefusal?.failure?.nodeErrors))
assert('N-1 throat refusal submits nothing', queueSubmissions === throatSubmissionsBefore,
  `queueSubmissions ${throatSubmissionsBefore} -> ${queueSubmissions}`)

// N-6: cross-path equivalence — the tool path and the throat path must produce
// the same code and the same missing-value set for the same graph.
const toolRefusal = await toolByName('comfyui_run', runtime).execute({ workflow: badGraph, run_label: 'throat-equiv-0001' }, exec)
const missingSet = (failure) => JSON.stringify((failure?.nodeErrors ?? []).map((entry) => [entry.value, entry.reason]))
assert('N-6 tool path and throat path refuse with the same code and missing values',
  toolRefusal.error?.code === throatRefusal?.code && missingSet(toolRefusal.error) === missingSet(throatRefusal?.failure),
  JSON.stringify({ tool: [toolRefusal.error?.code, missingSet(toolRefusal.error)], throat: [throatRefusal?.code, missingSet(throatRefusal?.failure)] }))

// --- N-4 + B1-C4: an unreadable object_info is a refusal, not a pass ---------
const deadMounted = await mountPlugin('http://127.0.0.1:1')
const deadSubmissions = queueSubmissions
let unavailable
try {
  await deadMounted.runtime.queue({ '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'x.safetensors' } } }, { workflowName: 'dead', source: 'smoke' })
  unavailable = null
} catch (error) {
  unavailable = error
}
assert('N-4 unreadable object_info refuses with PREFLIGHT_UNAVAILABLE',
  unavailable !== null && unavailable.code === 'PREFLIGHT_UNAVAILABLE', JSON.stringify(unavailable?.code ?? null))
assert('N-4 unavailable preflight submits nothing', queueSubmissions === deadSubmissions,
  `queueSubmissions ${deadSubmissions} -> ${queueSubmissions}`)
deadMounted.dispose()

// --- N-5/B-1.2: a parameter rewrite that produces a bad graph is caught ------
// Only the throat sees this: the tool-level preflight runs on the caller's
// graph, the parameter rewrite happens later, so this is the case that proves
// the check sits AFTER `applyWorkflowParameters`.
const mountedRuntime = throatRuntime
const savedResult = await mountedRuntime.saveWorkflow({
  name: 'smoke-param-rewrite',
  description: '参数改写后才引用未就位模型的图',
  workflow: {
    '3': { class_type: 'KSampler', inputs: { seed: 0, steps: 20, cfg: 6.5 } },
    '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_xl_base_1.0.safetensors' } },
  },
  parameters: [{ name: 'ckpt_name', label: 'ckpt', type: 'string', nodeId: '4', inputKey: 'ckpt_name' }],
})
const savedId = savedResult.workflow?.id ?? savedResult.id
assert('N-5 test fixture: the workflow is saved with an adjustable parameter', typeof savedId === 'string' && savedId !== '',
  JSON.stringify(savedResult))
const rewriteSubmissions = queueSubmissions
let rewritten
try {
  const saved = await mountedRuntime.getWorkflow(savedId)
  await mountedRuntime.queue(saved.workflow, {
    workflowName: 'smoke-param-rewrite',
    workflowId: savedId,
    source: 'smoke',
    parameters: [{ name: 'ckpt_name', label: 'ckpt', type: 'string', nodeId: '4', inputKey: 'ckpt_name' }],
    values: { ckpt_name: 'not-on-disk.safetensors' },
  })
  rewritten = null
} catch (error) {
  rewritten = error
}
assert('N-5 a parameter rewrite onto a missing model is refused after the rewrite',
  rewritten !== null && rewritten.code === 'PREFLIGHT'
  && JSON.stringify(rewritten.failure?.nodeErrors ?? []).includes('not-on-disk.safetensors'),
  JSON.stringify({ code: rewritten?.code, nodeErrors: rewritten?.failure?.nodeErrors }))
assert('N-5 the parameter-rewrite refusal submits nothing', queueSubmissions === rewriteSubmissions,
  `queueSubmissions ${rewriteSubmissions} -> ${queueSubmissions}`)

// --- N-3/S-3: the panel routes turn the refusal into a readable answer -------
const runRoute = mounted.routeOf('/comfyui/workflows/run')
const actionsRoute = mounted.routeOf('/comfyui/jobs/actions')
assert('N-3 test fixture: the panel routes are mounted on the web-server fiber',
  typeof runRoute === 'function' && typeof actionsRoute === 'function',
  JSON.stringify(mounted.routePaths()))
const panelSubmissions = queueSubmissions
const panelRun = await callRoute(runRoute, { id: savedId, parameters: { ckpt_name: 'not-on-disk.safetensors' } })
assert('N-3 panel run route answers 502 with the missing value and the available ones',
  panelRun.status === 502 && typeof panelRun.body?.error === 'string'
  && panelRun.body.error.includes('not-on-disk.safetensors')
  && panelRun.body.error.includes('sd_xl_base_1.0.safetensors'), JSON.stringify(panelRun))
assert('N-3 panel run route submits nothing', queueSubmissions === panelSubmissions,
  `queueSubmissions ${panelSubmissions} -> ${queueSubmissions}`)
const rerun = await callRoute(actionsRoute, { action: 'rerun', jobId: 'mock-prompt-1' })
assert('N-3 panel rerun route answers {ok:false,error} instead of crashing',
  rerun.status === 200 && rerun.body?.ok === false && String(rerun.body?.error ?? '').length > 0, JSON.stringify(rerun))
// The server is still serving: one more request on the same instance succeeds
// (the refusal threw, it did not take the route down).
const afterRefusals = await callRoute(runRoute, { id: savedId, parameters: { ckpt_name: 'not-on-disk.safetensors' } })
assert('N-3 the instance keeps serving after a refusal (no crash)',
  afterRefusals.status === 502 && panelRun.status === 502, JSON.stringify({ first: panelRun.status, second: afterRefusals.status }))
mounted.dispose()

// --- N-7/B-2: two concurrent runs must not share a runLabel ------------------
// The slow server makes the window between "draw the number" and "write the
// row" wide enough for the second run to enter it: that window is exactly the
// `await` the fix removed from the synchronous open/commit stretch. It answers
// /prompt and /history so the mounted plugin tool (used below) can run against
// the same instance.
const slowInfo = JSON.parse(JSON.stringify(OBJECT_INFO))
const submissionsById = new Map()
const slowServer = createServer((request, response) => {  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  if (url.pathname === '/object_info') {
    setTimeout(() => {
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify(slowInfo))
    }, 250)
    return
  }
  if (url.pathname === '/prompt') {
    let raw = ''
    request.on('data', (chunk) => { raw += chunk })
    request.on('end', () => {
      queueSubmissions += 1
      const id = `mock-prompt-${queueSubmissions}`
      submissionsById.set(id, JSON.parse(raw).prompt)
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ prompt_id: id }))
    })
    return
  }
  if (url.pathname.startsWith('/history/')) {
    const promptId = decodeURIComponent(url.pathname.slice('/history/'.length))
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify({ [promptId]: { status: { status_str: 'success', completed: true }, outputs: { '9': { images: [{ filename: 'out.png', subfolder: '', type: 'output' }] } } } }))
    return
  }
  if (url.pathname === '/view') {
    response.writeHead(200, { 'content-type': 'image/png' })
    response.end(MEDIA)
    return
  }
  response.writeHead(404, { 'content-type': 'application/json' })
  response.end(JSON.stringify({ error: `unexpected ${url.pathname}` }))
})
await new Promise((resolve) => slowServer.listen(0, '127.0.0.1', resolve))
const slowPort = slowServer.address().port
const concurrentDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-concurrent-'))
const slowedRuntime = runtimeFor(`http://127.0.0.1:${slowPort}`, concurrentDir)
// The mock runtime's `queue` is the submit throat, so it is the seam that can
// record which graph went with which promptId — the mock server's /prompt is
// never reached on this path.
const slowedQueue = slowedRuntime.queue
slowedRuntime.queue = async (workflow, meta) => {
  const promptId = await slowedQueue(workflow, meta)
  submissionsById.set(promptId, workflow)
  return promptId
}
const slowedRunTool = toolByName('comfyui_run', slowedRuntime)
const [concurrentA, concurrentB] = await Promise.all([
  slowedRunTool.execute({ template: 'txt2img', inputs: { '6': { text: 'concurrent a' } } }, exec),
  slowedRunTool.execute({ template: 'txt2img', inputs: { '6': { text: 'concurrent b' } } }, exec),
])
assert('N-7 concurrent runs get different runLabels',
  typeof concurrentA.runLabel === 'string' && typeof concurrentB.runLabel === 'string' && concurrentA.runLabel !== concurrentB.runLabel,
  `${concurrentA.runLabel} / ${concurrentB.runLabel}`)
const concurrentRows = readLedger(concurrentDir).records
assert('N-7 concurrent runs leave two ledger rows',
  concurrentRows.length === 2, JSON.stringify(concurrentRows.map((entry) => entry.runLabel)))
assert('N-7 each row carries its own promptId',
  concurrentRows.every((entry) => typeof entry.promptId === 'string' && entry.promptId.startsWith('mock-prompt-'))
  && concurrentRows[0].promptId !== concurrentRows[1].promptId,
  JSON.stringify(concurrentRows.map((entry) => entry.promptId)))
// The row's runLabel must belong to the submission that carried its own
// promptId. The mock runtime's `queue` is a stub (it never resolves seeds), so
// the seed cannot be compared here; the assertion keys on the graph identity
// each run passed in, which the mock records per promptId.
const seedMatchesOwnSubmission = concurrentRows.every((entry) => {
  const submitted = submissionsById.get(entry.promptId)
  const expectedText = concurrentA.runLabel === entry.runLabel ? 'concurrent a' : 'concurrent b'
  return submitted?.['6']?.inputs?.text === expectedText && submitted?.['3']?.inputs?.seed === entry.seed
})
assert('N-7 each row matches its own submission (prompt text + seed)',
  seedMatchesOwnSubmission,
  JSON.stringify({ rows: concurrentRows.map((entry) => ({ runLabel: entry.runLabel, promptId: entry.promptId, seed: entry.seed })), submittedKeys: [...submissionsById.keys()], texts: [...submissionsById.values()].map((workflow) => workflow?.['6']?.inputs?.text), expected: [concurrentA.runLabel, concurrentB.runLabel] }))

// --- N-8/B-2 static: the number and the row are drawn without an await -------
// A read of the ledger followed by a write of the row is only atomic while
// nothing can interleave; that holds if and only if no `await` sits between
// them, so this asserts the shape of `openRun` rather than trusting a comment.
const toolsSource = readFileSync(new URL('../src/tools.ts', import.meta.url), 'utf8')
const openRunBody = toolsSource.slice(toolsSource.indexOf('function openRun('), toolsSource.indexOf('/** A run result with no terminal state yet'))
assert('N-8 the reservation function contains no await and writes the row itself',
  openRunBody.includes('nextJobNumber(') && openRunBody.includes('upsertRun(') && !/\bawait\b/.test(openRunBody)
  && openRunBody.indexOf('nextJobNumber(') < openRunBody.indexOf('upsertRun('),
  `openRun body length ${openRunBody.length}`)
// Both slices are bounded by a marker that must actually be found: a missing
// marker would otherwise yield an empty or whole-file slice and pass silently.
function between(source, from, to) {
  const start = source.indexOf(from)
  const end = source.indexOf(to)
  if (start < 0 || end < 0 || end <= start) throw new Error(`marker not found: ${JSON.stringify({ from, to, start, end })}`)
  return source.slice(start, end)
}
const reservationBody = between(toolsSource, 'function preflightRun(', 'return { ok: true, runDir, runLabel, seed, seeds }')
const reserveToCommit = reservationBody.slice(reservationBody.lastIndexOf('const { runDir, runLabel } = openRun('))
const workflowPathBody = between(toolsSource, 'const { runDir, runLabel } = openRun(runtime, args.run_label, {', 'promptId = await runtime.queue(saved.workflow')
assert('N-8 nothing awaitable sits between drawing the number and committing the row',
  !/\bawait\b/.test(reserveToCommit) && reserveToCommit.includes('upsertRun') === false,
  JSON.stringify({ length: reserveToCommit.length }))
assert('N-8 the workflow path reserves and submits without an await in between',
  !/\bawait\b/.test(workflowPathBody),
  JSON.stringify({ workflowPathAwait: /\bawait\b/.test(workflowPathBody), length: workflowPathBody.length }))

// --- N-9/N-10/N-11/B-3: async without a jobs service leaves no queued row ----
const asyncDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-async-'))
const asyncRuntime = runtimeFor(`http://127.0.0.1:${port}`, asyncDir)
const asyncSubmissions = queueSubmissions
let asyncThrew
try {
  await toolByName('comfyui_run', asyncRuntime).execute({ template: 'txt2img', mode: 'async', run_label: 'asynq-a1' }, exec)
  asyncThrew = null
} catch (error) {
  asyncThrew = error
}
assert('N-9 async without a jobs service still throws the documented message',
  asyncThrew instanceof Error && /background jobs unavailable/.test(asyncThrew.message), String(asyncThrew?.message))
assert('N-9 async refusal is decided before the submit (nothing was sent)',
  queueSubmissions === asyncSubmissions, `queueSubmissions ${asyncSubmissions} -> ${queueSubmissions}`)
const asyncRowsAfter = readLedger(asyncDir).records
assert('N-9 no queued row is left behind', asyncRowsAfter.filter((entry) => entry.status === 'queued').length === 0,
  JSON.stringify(asyncRowsAfter))
assert('N-9 the refused run leaves no row at all', asyncRowsAfter.length === 0, JSON.stringify(asyncRowsAfter))
const afterAsync = await toolByName('comfyui_run', asyncRuntime).execute({ template: 'txt2img' }, exec)
// No label was given, so this run uses the default `comfyui` prefix: seeing
// `-0001` proves the refused async run consumed nothing from the ledger.
assert('N-9 the refused async run consumed no sequence number',
  afterAsync.runLabel === 'comfyui-0001', `${afterAsync.runLabel} (expected comfyui-0001, i.e. nothing was consumed)`)

// N-10: the second path is isomorphic. Its runtime serves the saved workflow
// so the run reaches the mode decision (the mock above has no workflow store).
const savedWorkflow = await mountedRuntime.getWorkflow(savedId)
const workflowAsyncRuntime = { ...asyncRuntime, getWorkflow: async (id) => (id === savedId ? savedWorkflow : undefined) }
let workflowAsyncThrew
try {
  await toolByName('comfyui_workflow', workflowAsyncRuntime).execute({
    action: 'run',
    id: savedId,
    mode: 'async',
    run_label: 'asynq2-0001',
  }, exec)
  workflowAsyncThrew = null
} catch (error) {
  workflowAsyncThrew = error
}
assert('N-10 comfyui_workflow run + async without jobs throws the same way',
  workflowAsyncThrew instanceof Error && /background jobs unavailable/.test(workflowAsyncThrew.message), String(workflowAsyncThrew?.message))
assert('N-10 comfyui_workflow async refusal consumed no sequence number and left no row',
  readLedger(asyncDir).records.filter((entry) => (entry.runLabel ?? '').startsWith('asynq2-')).length === 0
  && readLedger(asyncDir).records.filter((entry) => entry.status === 'queued').length === 0,
  JSON.stringify(readLedger(asyncDir).records.map((entry) => ({ runLabel: entry.runLabel, status: entry.status }))))

// N-11: a fetch addressed by that run's promptId hits the same row instead of
// opening a second one — the invariant that a hung `queued` row (promptId
// null) used to break.
const asyncRow = readLedger(asyncDir).records.find((entry) => entry.runLabel === afterAsync.runLabel)
assert('N-11 the completed run wrote a promptId to fetch by', typeof asyncRow?.promptId === 'string', JSON.stringify(asyncRow))
const fetchBefore = readLedger(asyncDir).records.length
const asyncFetch = await toolByName('comfyui_fetch_output', asyncRuntime).execute({ promptId: String(asyncRow?.promptId) }, exec)
const asyncRowsFinal = readLedger(asyncDir).records
assert('N-11 fetching by promptId does not add a ledger row',
  asyncRowsFinal.length === fetchBefore, JSON.stringify(asyncRowsFinal.map((entry) => entry.runLabel)))
assert('N-11 the fetch fills in the row opened by that run',
  asyncRowsFinal.find((entry) => entry.runLabel === afterAsync.runLabel)?.files?.[0]?.absPath === asyncFetch.files?.[0]?.absPath,
  JSON.stringify({ files: asyncRowsFinal.find((entry) => entry.runLabel === afterAsync.runLabel)?.files, fetched: asyncFetch.files?.[0]?.absPath }))

// --- N-12/N-13/B-4: no banned keyword on either side, PTC rendering intact ---
const ALLOWED = new Set(['type', 'oneOf', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const', 'description', 'title', 'default', 'examples'])
function bannedKeys(node, path) {
  const found = []
  if (typeof node !== 'object' || node === null || Array.isArray(node)) return found
  for (const key of Object.keys(node)) {
    if (!ALLOWED.has(key)) found.push(`${path}.${key}`)
  }
  for (const [key, child] of Object.entries(node.properties ?? {})) found.push(...bannedKeys(child, `${path}.properties.${key}`))
  if (node.items !== undefined) found.push(...bannedKeys(node.items, `${path}.items`))
  if (Array.isArray(node.oneOf)) node.oneOf.forEach((branch, index) => found.push(...bannedKeys(branch, `${path}.oneOf[${index}]`)))
  return found
}
const allTools = comfyUIToolDefinitions({ get: () => undefined }, runtime)
const sideViolations = allTools.flatMap((tool) => [
  ...bannedKeys(tool.parameters, 'parameters').map((entry) => `${tool.name}: ${entry}`),
  ...bannedKeys(tool.output?.schema, 'output.schema').map((entry) => `${tool.name}: ${entry}`),
])
assert('N-12 no keyword outside the whitelist on either side (including minimum/maximum)',
  sideViolations.length === 0, JSON.stringify(sideViolations))
const ptcDegraded = []
for (const tool of allTools) {
  try {
    const rendered = jsonSchemaToTs(tool.parameters)
    if (rendered === 'unknown' || /^unknown$/m.test(String(rendered))) ptcDegraded.push(tool.name)
  } catch (error) {
    ptcDegraded.push(`${tool.name}: threw ${error instanceof Error ? error.message : String(error)}`)
  }
}
assert('N-13 all six tools render to a non-degraded PTC input type', ptcDegraded.length === 0, JSON.stringify(ptcDegraded))

// --- N-14/F-1: an undeclared argument is refused, a declared one is not ------
const typo = await toolByName('comfyui_run', runtime).execute({ template: 'txt2img', timeout: 1000 }, exec)
assert('N-14 a misspelled argument is refused with UNKNOWN_ARGUMENT',
  typo.error?.code === 'UNKNOWN_ARGUMENT' && typo.error.message.includes('timeout')
  && typo.error.message.includes('timeout_ms'), JSON.stringify(typo.error))
const typoSubmissions = queueSubmissions
assert('N-14 the refused call never reached the server', queueSubmissions === typoSubmissions)
const legit = await toolByName('comfyui_run', runtime).execute({ template: 'txt2img', timeout_ms: 30_000, run_label: 'legit-0001' }, exec)
assert('N-14 a declared argument is not refused',
  legit.status === 'completed' && legit.error === null, JSON.stringify({ status: legit.status, error: legit.error }))
const workflowTypo = await toolByName('comfyui_workflow', runtime).execute({ action: 'list', limit: 3 }, exec)
assert('N-14 the second tool refuses its own misspelling too',
  workflowTypo.error?.code === 'UNKNOWN_ARGUMENT' || /UNKNOWN_ARGUMENT/.test(String(workflowTypo?.error?.message ?? '')),
  JSON.stringify(workflowTypo?.error ?? workflowTypo))

slowServer.close()
for (const dir of [concurrentDir, asyncDir, mounted.dataDir]) rmSync(dir, { recursive: true, force: true })
server.close()
for (const dir of [downloads, corruptDir, fetchDir]) rmSync(dir, { recursive: true, force: true })

for (const check of checks) console.log(`  ${check.ok ? 'OK  ' : 'FAIL'} ${check.label}${check.ok || check.detail === '' ? '' : ` — ${check.detail}`}`)
console.log(`\n结论：${checks.length - failures.length}/${checks.length} 项通过`)
if (failures.length > 0) {
  console.log(`FAIL ${failures.length} 项：`)
  for (const failure of failures) console.log(`  - ${failure}`)
}
process.exit(failures.length === 0 ? 0 : 1)
