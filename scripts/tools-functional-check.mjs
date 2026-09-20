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
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
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
// Counted so N-22 can state the `object_info` cost of each path as a measured
// number instead of repeating a claim about it.
let objectInfoCalls = 0

const server = createServer((request, response) => {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  const json = (body) => {
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(body))
  }
  if (url.pathname === '/system_stats') {
    return json({ system: { comfyui_version: '0.3.99-smoke' }, devices: [{ name: 'cuda:0 NVIDIA Smoke', type: 'cuda', vram_free: 12_345 }] })
  }
  if (url.pathname === '/object_info') {
    objectInfoCalls += 1
    return json(OBJECT_INFO)
  }
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
 *
 * `jobsPlan` decides what `ctx.get('jobs')` answers, which is the one seam the
 * BI-1 takeover cases need:
 *   undefined         -> no jobs service at all (the pre-submit refusal, N-9)
 *   'start-throws'    -> the service is there but `start()` throws (N-16/N-17)
 *   'vanish-on-read2' -> the service is there on the first read and gone on the
 *                        second — i.e. it unloads between the pre-submit check
 *                        and the takeover (N-18)
 */
async function mountPlugin(baseUrl, jobsPlan) {
  const dataDir = mkdtempSync(join(tmpdir(), 'dsh-comfyui-mount-'))
  const disposers = []
  const capturedRuntime = { value: undefined }
  const tools = []
  const routes = []
  const jobsReads = { count: 0 }
  // `start` never gets far enough to register a job: BI-1 is about the failure
  // of the takeover itself, not about what a started job would later do.
  const jobsService = { start() { throw new Error('smoke: jobs service refused to start the task') } }
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
      if (name === 'jobs') {
        jobsReads.count += 1
        if (jobsPlan === 'start-throws') return jobsService
        if (jobsPlan === 'vanish-on-read2') return jobsReads.count === 1 ? jobsService : undefined
        return undefined
      }
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
  return { runtime, toolFor, dataDir, runDir: join(dataDir, 'runs'), jobsReads, routeOf, routePaths: () => routes.map((route) => route.path), dispose: () => { for (const dispose of disposers) dispose() } }
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

// --- N-16/N-17/N-18/N-19/BI-1: a takeover that fails after the submit --------
// The window: the prompt has already been POSTed to `/prompt`, and then the
// background job cannot be started — either `start()` throws, or the jobs
// service is gone by the second read (the pre-submit check read it once
// already). The row the reservation opened must not stay `queued` with a null
// promptId: that is the shape that made a submitted prompt unreachable from
// `comfyui_fetch_output`, and it is exactly what this block pins down.
//
// Both mounts go through the plugin's own entry point and the real
// `runtime.queue`, so `queueSubmissions` counts requests that really reached
// the mock server — the invariant is not passed by "nothing was submitted".
const takeoverMounts = []
const takeoverCases = [
  { caseId: 'N-16', label: 'comfyui_run / start() throws', plan: 'start-throws', tool: 'comfyui_run', infoCalls: 2 },
  { caseId: 'N-17', label: 'comfyui_workflow run / start() throws', plan: 'start-throws', tool: 'comfyui_workflow', infoCalls: 1 },
  { caseId: 'N-18', label: 'comfyui_run / jobs service vanishes on the second read', plan: 'vanish-on-read2', tool: 'comfyui_run', infoCalls: 2 },
  { caseId: 'N-18', label: 'comfyui_workflow run / jobs service vanishes on the second read', plan: 'vanish-on-read2', tool: 'comfyui_workflow', infoCalls: 1 },
]
const objectInfoCounts = []
for (const entry of takeoverCases) {
  const takeoverMount = await mountPlugin(`http://127.0.0.1:${port}`, entry.plan)
  takeoverMounts.push(takeoverMount)
  const savedTakeover = await takeoverMount.runtime.saveWorkflow({
    name: 'smoke-takeover',
    description: 'BI-1 夹具：接管失败时台账必须落终态',
    workflow: {
      '3': { class_type: 'KSampler', inputs: { seed: 0, steps: 20, cfg: 6.5 } },
      '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_xl_base_1.0.safetensors' } },
      '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'dsh-comfyui' } },
    },
  })
  const takeoverWorkflowId = savedTakeover.workflow?.id ?? savedTakeover.id
  const label = `${entry.caseId} (${entry.label})`
  const args = entry.tool === 'comfyui_run'
    ? { template: 'txt2img', mode: 'async', run_label: `${entry.caseId.toLowerCase()}-0001` }
    : { action: 'run', id: takeoverWorkflowId, mode: 'async', run_label: `${entry.caseId.toLowerCase()}w-0001` }
  const submissionsBeforeTakeover = queueSubmissions
  const objectInfoBeforeTakeover = objectInfoCalls
  const rowsBeforeTakeover = readLedger(takeoverMount.runDir).records.length
  let takeoverThrew
  let takeoverReturned
  try {
    takeoverReturned = await takeoverMount.toolFor(entry.tool).execute(args, exec)
    takeoverThrew = null
  } catch (error) {
    takeoverThrew = error
  }
  objectInfoCounts.push({
    path: entry.tool,
    caseId: entry.caseId,
    objectInfoCalls: objectInfoCalls - objectInfoBeforeTakeover,
    expected: entry.infoCalls,
  })
  const rows = readLedger(takeoverMount.runDir).records
  const row = rows[0]
  const expectedPromptId = `mock-prompt-${queueSubmissions}`
  assert(`${label}: the caller learns the takeover failed instead of being handed a background job`,
    takeoverThrew instanceof Error && /background jobs unavailable/.test(takeoverThrew.message)
    && takeoverReturned === undefined,
    JSON.stringify({ threw: takeoverThrew === null ? null : takeoverThrew.message, returned: takeoverReturned ?? null }))
  assert(`${label}: the prompt really had been submitted before the takeover failed`,
    queueSubmissions - submissionsBeforeTakeover === 1,
    `submissions ${submissionsBeforeTakeover} -> ${queueSubmissions}`)
  assert(`${label}: the failure names the submitted promptId so it is not read as "nothing was sent"`,
    takeoverThrew instanceof Error && takeoverThrew.message.includes(expectedPromptId),
    String(takeoverThrew instanceof Error ? takeoverThrew.message : takeoverThrew))
  assert(`${label}: no queued row is left behind`,
    rows.filter((item) => item.status === 'queued').length === 0, JSON.stringify(rows))
  assert(`${label}: the reservation row is terminal and carries the real promptId`,
    rows.length === rowsBeforeTakeover + 1
    && (row?.status === 'failed' || row?.status === 'interrupted')
    && row.promptId === expectedPromptId,
    JSON.stringify({ rows, expectedPromptId }))
  assert(`${label}: the failure is coded JOBS_UNAVAILABLE with no node errors`,
    row?.error?.code === 'JOBS_UNAVAILABLE' && row.error.nodeErrors === null, JSON.stringify(row?.error))
  assert(`${label}: the run took exactly two jobs-service reads (pre-submit check + takeover)`,
    takeoverMount.jobsReads.count === 2, String(takeoverMount.jobsReads.count))
  // N-19: that promptId is the handle out. Fetching by it must fill the SAME
  // row in — a second row would mean the run and its output had drifted apart.
  const fetchRowsBefore = rows.length
  const takeoverFetch = await takeoverMount.toolFor('comfyui_fetch_output').execute({ promptId: expectedPromptId }, exec)
  const rowsAfterFetch = readLedger(takeoverMount.runDir).records
  assert(`${label}/N-19: fetching by that promptId fills the same row instead of opening another`,
    rowsAfterFetch.length === fetchRowsBefore
    && rowsAfterFetch.find((item) => item.runLabel === row?.runLabel)?.files?.[0]?.absPath === takeoverFetch?.files?.[0]?.absPath
    && typeof takeoverFetch?.files?.[0]?.absPath === 'string',
    JSON.stringify({ rowsBefore: fetchRowsBefore, rowsAfter: rowsAfterFetch.length, rowFiles: rowsAfterFetch.find((item) => item.runLabel === row?.runLabel)?.files ?? null, fetched: takeoverFetch?.files?.[0]?.absPath ?? null }))
}

// --- N-22/BI-2: the object_info cost of each path, measured ------------------
// `comfyui_run` reads the snapshot twice per successful submit (its own early
// check, then the submit throat); every other path goes through the throat
// alone. Both numbers are recorded rather than described.
const runInfoCounts = objectInfoCounts.filter((entry) => entry.path === 'comfyui_run')
const workflowInfoCounts = objectInfoCounts.filter((entry) => entry.path === 'comfyui_workflow')
assert('N-22 comfyui_run reads object_info twice per successful submit',
  runInfoCounts.length === 2 && runInfoCounts.every((entry) => entry.objectInfoCalls === 2),
  JSON.stringify(runInfoCounts))
assert('N-22 comfyui_workflow reaches object_info once per successful submit',
  workflowInfoCounts.length === 2 && workflowInfoCounts.every((entry) => entry.objectInfoCalls === 1),
  JSON.stringify(workflowInfoCounts))

// --- N-20/N-21/BI-1+BI-2 static: the shape the runtime cases rest on ---------
// N-20: both takeover segments must actually be wrapped — a passing runtime case
// on one path would otherwise hide the other path still throwing raw.
const takeoverWindows = [...toolsSource.matchAll(/if \(mode === 'async'\) \{\s*\n\s*let jobId: string\s*\n\s*try \{/g)]
  .map((match) => {
    const tail = toolsSource.slice(match.index, match.index + 4000)
    const end = tail.indexOf("Result = { kind: 'background'")
    return end < 0 ? tail : tail.slice(0, end)
  })
assert('N-20 both async takeover segments wrap start() in try/catch and close the row out',
  takeoverWindows.length === 2 && takeoverWindows.every((window) => window.includes('catch (error) {')
    && window.includes('upsertRun(runDir, { ...')
    && window.includes('jobsTakeoverFailure(')
    && window.includes("status: 'failed'")
    && window.includes('throw new Error(failure.message)')
    && !window.includes("status: 'queued'")),
  JSON.stringify({ windows: takeoverWindows.length, lengths: takeoverWindows.map((window) => window.length) }))

// N-21: the callsite count that BI-2 is judged on, counted the way the
// criterion words it (definition excluded), plus a zero-hit sweep for the dead
// export on both sides of the build.
function sourceFiles(root) {
  const out = []
  for (const name of readdirSync(new URL(root, import.meta.url), { withFileTypes: true })) {
    if (name.isDirectory()) out.push(...sourceFiles(`${root}/${name.name}`))
    else if (/\.(ts|js)$/.test(name.name)) out.push(new URL(`${root}/${name.name}`, import.meta.url))
  }
  return out
}
const preflightCallsites = []
for (const file of sourceFiles('../src')) {
  const lines = readFileSync(file, 'utf8').split('\n')
  lines.forEach((line, index) => {
    if (/preflightWorkflow\(/.test(line) && !/export function preflightWorkflow\(/.test(line)) {
      preflightCallsites.push(`${file.pathname.split('/src/')[1]}:${index + 1}`)
    }
  })
}
assert('N-21 preflightWorkflow( has exactly 2 callsites once the definition is excluded',
  preflightCallsites.length === 2
  && preflightCallsites.some((site) => site.startsWith('index.ts:'))
  && preflightCallsites.some((site) => site.startsWith('tools.ts:')),
  JSON.stringify(preflightCallsites))
const deadExportHits = []
for (const root of ['../src', '../lib']) {
  for (const file of sourceFiles(root)) {
    const text = readFileSync(file, 'utf8')
    const count = (text.match(/preflightApproved/g) ?? []).length
    if (count > 0) deadExportHits.push(`${root}/${file.pathname.split('/').pop()}:${count}`)
  }
}
assert('N-21 preflightApproved is gone from both src/ and lib/',
  deadExportHits.length === 0, JSON.stringify(deadExportHits))

// --- F-4/F-5 (rectify segment 1, P4): a failure with no node detail ----------
// The defect lives in `firstErrorText`'s CALLERS, not in the function alone, so
// F-4 is behavioural rather than a unit test on a module-private helper: the
// mock below answers a prompt whose history says "failed" while attaching no
// exception detail at all — the shape the real 69.7 s run left behind (the
// server reported `status_str: success`, the ledger row carried
// `nodeErrors: null`, and the tool still reported a failure). Under the old
// `firstErrorText` the only text that survived that shape was the flat
// `'unknown error'`, which is exactly how the real cause was destroyed at the
// reporting site. F-5 is the static half: all five call sites must carry their
// own fallback so none of them can pass a possibly-empty string through as the
// message.
let rectSubmissions = 0
/** What the mock's history says: success, or failed with/without detail. */
let rectHistoryMode = 'bare'
/** The graph bodies sent to `/prompt`, in submission order. */
const rectSubmitted = []
// The recipe cards need a few more class types than the templates do. They are
// declared bare here (a superset of the shared OBJECT_INFO, so no existing case
// changes meaning): preflight then has nothing to validate on them, which is
// what these cases want — they are about placeholder injection, not availability.
const RECT_OBJECT_INFO = {
  ...OBJECT_INFO,
  LoadImage: {},
  Canny: {},
  ControlNetLoader: {},
  SetUnionControlNetType: {},
  ControlNetApplyAdvanced: {},
}
const rectServer = createServer(async (request, response) => {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  const json = (body) => {
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(body))
  }
  if (url.pathname === '/system_stats') {
    return json({ system: { comfyui_version: '0.3.99-rectify' }, devices: [{ name: 'cuda:0 NVIDIA Rectify', type: 'cuda', vram_free: 4096 }] })
  }
  if (url.pathname === '/object_info') return json(RECT_OBJECT_INFO)
  if (url.pathname === '/queue') return json({ queue_running: [], queue_pending: [] })
  if (url.pathname === '/prompt') {
    let raw = ''
    for await (const chunk of request) raw += chunk
    rectSubmitted.push(JSON.parse(raw).prompt)
    rectSubmissions += 1
    return json({ prompt_id: `rect-prompt-${rectSubmissions}` })
  }
  if (url.pathname.startsWith('/history/')) {
    const promptId = decodeURIComponent(url.pathname.slice('/history/'.length))
    if (rectHistoryMode === 'success') {
      return json({
        [promptId]: {
          status: { status_str: 'success', completed: true },
          outputs: { '7': { images: [{ filename: 'rect.png', subfolder: '', type: 'output' }] } },
        },
      })
    }
    const messages = rectHistoryMode === 'detailed'
      ? [['execution_error', { node_id: '3', node_type: 'KSampler', exception_message: 'RECT-REAL-ERROR-TEXT' }]]
      : []
    return json({ [promptId]: { status: { status_str: 'error', completed: false, messages }, outputs: {} } })
  }
  response.writeHead(404, { 'content-type': 'application/json' })
  response.end(JSON.stringify({ error: `unexpected ${url.pathname}` }))
})
await new Promise((resolve) => rectServer.listen(0, '127.0.0.1', resolve))
const rectPort = rectServer.address().port
const rectMount = await mountPlugin(`http://127.0.0.1:${rectPort}`, undefined)
const rectRows = () => readLedger(rectMount.runDir).records

rectHistoryMode = 'bare'
const bareRun = await rectMount.toolFor('comfyui_run').execute({ template: 'txt2img', run_label: 'rect-p4-bare-0001' }, exec)
const bareRow = rectRows().find((entry) => entry.runLabel === 'rect-p4-bare-0001')
assert('F-4 a failure carrying no node detail reports the raised cause, not the placeholder alone',
  bareRun.status === 'error'
  && bareRun.error?.message !== 'unknown error'
  && /ComfyUI execution failed/.test(String(bareRun.error?.message ?? ''))
  && String(bareRun.error?.message ?? '').includes(`rect-prompt-${rectSubmissions}`),
  JSON.stringify({ status: bareRun.status, error: bareRun.error }))
assert('F-4 the ledger row carries the same real message instead of the placeholder',
  bareRow?.error?.message === bareRun.error?.message
  && /ComfyUI execution failed/.test(String(bareRow?.error?.message ?? '')),
  JSON.stringify(bareRow?.error ?? null))

rectHistoryMode = 'detailed'
const detailedRun = await rectMount.toolFor('comfyui_run').execute({ template: 'txt2img', run_label: 'rect-p4-detailed-0001' }, exec)
const detailedRow = rectRows().find((entry) => entry.runLabel === 'rect-p4-detailed-0001')
assert('F-6 a run that failed WITH node detail still reports that detail (unchanged by the fix)',
  /RECT-REAL-ERROR-TEXT/.test(String(detailedRun.error?.message ?? ''))
  && /RECT-REAL-ERROR-TEXT/.test(String(detailedRow?.error?.message ?? '')),
  JSON.stringify({ result: detailedRun.error ?? null, ledger: detailedRow?.error ?? null }))

const firstErrorCallLines = toolsSource.split('\n')
  .map((line, index) => ({ line, number: index + 1 }))
  .filter((entry) => /firstErrorText\(/.test(entry.line))
  .filter((entry) => !/^\s*(\*|\/\/)/.test(entry.line))
  .filter((entry) => !/function firstErrorText\(/.test(entry.line))
assert('F-5 all remaining firstErrorText callsites carry a fallback (no bare call survives)',
  firstErrorCallLines.length === 3 && firstErrorCallLines.every((entry) => entry.line.includes('||')),
  JSON.stringify(firstErrorCallLines.map((entry) => `tools.ts:${entry.number} ${entry.line.trim()}`)))
assert('F-5 the empty state of firstErrorText is the empty string, never a placeholder',
  /function firstErrorText\([\s\S]{0,900}?if \(first === undefined\) return ''/.test(toolsSource),
  JSON.stringify(toolsSource.slice(toolsSource.indexOf('function firstErrorText(')).split('\n').slice(0, 4)))
// The two former bare callsites are not "fixed" by adding a fallback there:
// they were rebuilding the failure out of `nodeErrors` alone and dropping the
// message `waitAndRecord` had already put on the ledger row. The failure is now
// handed over whole, which is what makes the tool result and its row agree.
assert('F-4 the failure waitAndRecord recorded is handed to the caller whole (sync + async)',
  (toolsSource.match(/error: state\.failure,/g) ?? []).length === 2
  && (toolsSource.match(/error: terminal\.failure \?\?/g) ?? []).length === 2,
  JSON.stringify({
    async: (toolsSource.match(/error: state\.failure,/g) ?? []).length,
    sync: (toolsSource.match(/error: terminal\.failure \?\?/g) ?? []).length,
  }))

// --- F-1/F-2/F-3 (rectify segment 2, P2): the caller's graph is never written
// The host freezes a tool call's arguments (`dsh-tools`: `snapshotJsonValue` →
// `deepFreeze`), and the run writes into the graph before submitting it (the
// `inputs` merge, seed resolution, later the placeholder injection). These
// cases hand over a graph frozen the same way and check three things at once:
// the call does not throw, the submitted graph carries the resolved values, and
// the caller's own object is untouched afterwards.
const RECIPES_ROOT = join(homedir(), '.agents', 'skills', 'Comfyui-use', 'recipes')
/** The recipe card's own file — never retyped here, so the fixture cannot drift. */
function readRecipeWorkflow(recipe) {
  return JSON.parse(readFileSync(join(RECIPES_ROOT, recipe, `${recipe}.workflow.json`), 'utf8'))
}
/** Recursive `Object.freeze`, i.e. what `deepFreeze` does to the arguments. */
function deepFreeze(value) {
  if (value !== null && typeof value === 'object') {
    for (const key of Object.keys(value)) deepFreeze(value[key])
    Object.freeze(value)
  }
  return value
}
/** Whether every object in the tree is still frozen (the clone was the writer). */
function stillFrozen(value) {
  if (value === null || typeof value !== 'object') return true
  return Object.isFrozen(value) && Object.values(value).every(stillFrozen)
}
/**
 * The poster card as a caller who is NOT using the params channel hands it
 * over: read from the recipe file (structure, wiring and node ids all come from
 * the card), then fully materialised — every placeholder resolved — so these
 * three cases exercise the clone alone. A NUMBER in a seed slot is required:
 * without one the old code never wrote, and the freeze bug would not reproduce.
 * (The raw, placeholder-carrying card is covered by F-7..F-22 instead.)
 */
function posterCardWithNumericSeed() {
  const graph = readRecipeWorkflow('poster-sdxl-base')
  graph['1'].inputs.ckpt_name = 'sd_xl_base_1.0.safetensors'
  graph['2'].inputs.text = 'a postcard subject'
  graph['3'].inputs.text = 'text, watermark'
  graph['4'].inputs.width = 1024
  graph['4'].inputs.height = 1536
  graph['5'].inputs.steps = 24
  graph['5'].inputs.cfg = 6.5
  graph['5'].inputs.sampler_name = 'euler'
  graph['5'].inputs.scheduler = 'normal'
  graph['5'].inputs.seed = 0
  graph['7'].inputs.filename_prefix = 'rect-p2'
  return graph
}

rectHistoryMode = 'success'
rectSubmitted.length = 0

const frozenCard = deepFreeze(posterCardWithNumericSeed())
const frozenBefore = JSON.stringify(frozenCard)
let frozenRun
try {
  frozenRun = await rectMount.toolFor('comfyui_run').execute({ workflow: frozenCard, run_label: 'rect-p2-frozen-0001' }, exec)
} catch (error) {
  frozenRun = { threw: error }
}
assert('F-1 a deeply frozen inline graph with a numeric seed is accepted (no TypeError)',
  frozenRun.threw === undefined && frozenRun.status === 'completed',
  JSON.stringify({ threw: String(frozenRun.threw ?? ''), status: frozenRun.status, error: frozenRun.error ?? null }))
assert('F-1 the submitted graph is the run\'s own object, with the seed written into it',
  rectSubmitted.length === 1 && rectSubmitted[0]['5'].inputs.seed === 0,
  JSON.stringify(rectSubmitted[0]?.['5']?.inputs ?? null))
assert('F-1 the caller\'s frozen graph is byte-identical and still frozen after the call',
  JSON.stringify(frozenCard) === frozenBefore && stillFrozen(frozenCard),
  JSON.stringify({ same: JSON.stringify(frozenCard) === frozenBefore, frozen: stillFrozen(frozenCard) }))

const frozenOverridden = deepFreeze(posterCardWithNumericSeed())
const overriddenBefore = JSON.stringify(frozenOverridden)
let overriddenRun
try {
  overriddenRun = await rectMount.toolFor('comfyui_run').execute({
    workflow: frozenOverridden,
    inputs: { '4': { width: 768 } },
    run_label: 'rect-p2-frozen-0002',
  }, exec)
} catch (error) {
  overriddenRun = { threw: error }
}
assert('F-2 a frozen graph plus an `inputs` override neither throws nor leaks the merge',
  overriddenRun.threw === undefined && overriddenRun.status === 'completed'
  && rectSubmitted[1]?.['4']?.inputs?.width === 768
  && rectSubmitted[1]?.['4']?.inputs?.height === 1536,
  JSON.stringify({ threw: String(overriddenRun.threw ?? ''), node4: rectSubmitted[1]?.['4']?.inputs ?? null }))
assert('F-2 the frozen override target is unchanged in the caller\'s graph',
  JSON.stringify(frozenOverridden) === overriddenBefore && frozenOverridden['4'].inputs.width === 1024,
  JSON.stringify({ same: JSON.stringify(frozenOverridden) === overriddenBefore, width: frozenOverridden['4'].inputs.width }))

const frozenSeeded = deepFreeze(posterCardWithNumericSeed())
const seededBefore = JSON.stringify(frozenSeeded)
let seededRun
try {
  seededRun = await rectMount.toolFor('comfyui_run').execute({ workflow: frozenSeeded, seed: 12345, run_label: 'rect-p2-frozen-0003' }, exec)
} catch (error) {
  seededRun = { threw: error }
}
const seededRow = rectRows().find((entry) => entry.runLabel === 'rect-p2-frozen-0003')
assert('F-3 a frozen graph plus a top-level seed neither throws nor writes back',
  seededRun.threw === undefined && rectSubmitted[2]?.['5']?.inputs?.seed === 12345
  && seededRun.seed === 12345 && seededRow?.seed === 12345
  && JSON.stringify(frozenSeeded) === seededBefore,
  JSON.stringify({
    threw: String(seededRun.threw ?? ''),
    submitted: rectSubmitted[2]?.['5']?.inputs?.seed ?? null,
    seed: seededRun.seed ?? null,
    row: seededRow?.seed ?? null,
    same: JSON.stringify(frozenSeeded) === seededBefore,
  }))

// --- F-7..F-22 (rectify segment 3, G-1): recipe-card placeholders ------------
// The cards are read from their own files, unedited: `<name>` / `<name: default>`
// values go in and a runnable graph comes out. What the cases pin down is that
// the fill happens (a) only where a placeholder is, (b) with the TYPE the card
// promised, (c) before `inputs` and before seed resolution, and (d) never on the
// caller's object.
const POSTER_PARAMS = {
  ckpt_name: 'sd_xl_base_1.0.safetensors',
  positive: 'a',
  negative: 'b',
  width: 512,
  height: 512,
  steps: 10,
  cfg: 7.5,
  sampler_name: 'euler',
  scheduler: 'normal',
  filename_prefix: 't',
  seed: 7,
}
/** The names the poster card declares with no default: passing them is mandatory. */
const POSTER_REQUIRED = { positive: 'a', negative: 'b', filename_prefix: 't', seed: 11 }

rectSubmitted.length = 0
async function rectRun(args, label) {
  const index = rectSubmitted.length
  try {
    const result = await rectMount.toolFor('comfyui_run').execute({ ...args, run_label: `rect-g1-${label}` }, exec)
    return { result, body: rectSubmitted[index], threw: null }
  } catch (error) {
    return { result: null, body: rectSubmitted[index], threw: error }
  }
}
/** A refusal that happened before `openRun` opened a row (the documented shape). */
const noRow = (label) => rectRows().find((entry) => entry.runLabel === `rect-g1-${label}`) === undefined
/** The placeholder positions of a card, i.e. the only slots a fill may touch. */
function placeholderPositions(card) {
  const positions = new Set()
  for (const [nodeId, node] of Object.entries(card)) {
    for (const [key, value] of Object.entries(node.inputs)) {
      if (typeof value === 'string' && value.includes('<') && value.includes('>')) positions.add(`${nodeId}.${key}`)
    }
  }
  return positions
}
/** Every non-placeholder slot (and the node set) that changed during the fill. */
function fidelityDiff(card, submitted) {
  const positions = placeholderPositions(card)
  const diff = []
  for (const [nodeId, node] of Object.entries(card)) {
    const got = submitted?.[nodeId]
    if (got === undefined || got.class_type !== node.class_type) {
      diff.push(`${nodeId}.class_type`)
      continue
    }
    for (const [key, value] of Object.entries(node.inputs)) {
      if (positions.has(`${nodeId}.${key}`)) continue
      if (JSON.stringify(got.inputs[key]) !== JSON.stringify(value)) diff.push(`${nodeId}.${key}`)
    }
    for (const key of Object.keys(got.inputs)) {
      if (!(key in node.inputs)) diff.push(`${nodeId}.${key}:added`)
    }
  }
  if (Object.keys(submitted ?? {}).length !== Object.keys(card).length) diff.push('node-set')
  return diff
}

const posterCard = readRecipeWorkflow('poster-sdxl-base')

const f7 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: POSTER_PARAMS }, 'f7-0001')
assert('F-7 a card submitted verbatim is filled from `params`, with the card\'s own types',
  f7.threw === null && f7.result?.status === 'completed'
  && f7.body?.['4']?.inputs?.width === 512 && typeof f7.body['4'].inputs.width === 'number'
  && f7.body?.['5']?.inputs?.seed === 7 && typeof f7.body['5'].inputs.seed === 'number'
  && f7.body?.['5']?.inputs?.steps === 10 && typeof f7.body['5'].inputs.steps === 'number'
  && f7.body?.['2']?.inputs?.text === 'a' && f7.body?.['7']?.inputs?.filename_prefix === 't',
  JSON.stringify({ threw: String(f7.threw ?? ''), status: f7.result?.status ?? null, error: f7.result?.error ?? null, body: f7.body ?? null }))

const f8 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: POSTER_PARAMS }, 'f8-0001')
assert('F-8 a string default takes a same-type string unchanged',
  f8.threw === null && f8.body?.['1']?.inputs?.ckpt_name === 'sd_xl_base_1.0.safetensors',
  JSON.stringify({ threw: String(f8.threw ?? ''), ckpt: f8.body?.['1']?.inputs?.ckpt_name ?? null }))

const f9 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: { ...POSTER_REQUIRED, sampler_name: 3 } }, 'f9-0001')
assert('F-9 E-2: a number passed to a string default is refused (no silent stringify)',
  f9.threw !== null && /参数类型冲突/.test(f9.threw.message) && f9.threw.message.includes('sampler_name')
  && /推断为 string/.test(f9.threw.message) && /为 number/.test(f9.threw.message)
  && noRow('f9-0001') && f9.body === undefined,
  JSON.stringify({ message: String(f9.threw?.message ?? ''), submitted: f9.body !== undefined }))

const f10 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: { ...POSTER_REQUIRED, cfg: '7.5' } }, 'f10-0001')
assert('F-10 a numeric string widens into a numeric default',
  f10.threw === null && f10.body?.['5']?.inputs?.cfg === 7.5 && typeof f10.body['5'].inputs.cfg === 'number',
  JSON.stringify({ threw: String(f10.threw ?? ''), cfg: f10.body?.['5']?.inputs?.cfg ?? null }))

const f11 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: { ...POSTER_REQUIRED, cfg: 'abc' } }, 'f11-0001')
assert('F-11 E-2: a non-numeric string for a numeric default is refused',
  f11.threw !== null && /参数类型冲突/.test(f11.threw.message) && f11.threw.message.includes('cfg') && noRow('f11-0001'),
  JSON.stringify(String(f11.threw?.message ?? '')))

const f12 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: { ...POSTER_REQUIRED, steps: 30 } }, 'f12-0001')
assert('F-12 a placeholder with a default falls back to it when nothing is passed',
  f12.threw === null && f12.result?.status === 'completed'
  && f12.body?.['5']?.inputs?.steps === 30
  && f12.body?.['4']?.inputs?.width === 1024 && f12.body?.['4']?.inputs?.height === 1536
  && f12.body?.['5']?.inputs?.sampler_name === 'euler' && f12.body?.['5']?.inputs?.cfg === 6.5,
  JSON.stringify({ threw: String(f12.threw ?? ''), node4: f12.body?.['4']?.inputs ?? null, node5: f12.body?.['5']?.inputs ?? null }))

const f13 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base') }, 'f13-0001')
assert('F-13 E-3: a default-less placeholder with no value is refused, naming every one of them',
  f13.threw !== null && /缺少必填参数/.test(f13.threw.message)
  && ['positive', 'negative', 'seed', 'filename_prefix'].every((name) => f13.threw.message.includes(name))
  && /node "2"\.text/.test(f13.threw.message) && noRow('f13-0001') && f13.body === undefined,
  JSON.stringify({ message: String(f13.threw?.message ?? ''), submitted: f13.body !== undefined }))

const inlineTextGraph = (text) => ({
  '6': { class_type: 'CLIPTextEncode', inputs: { text, clip: ['4', 1] } },
  '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_xl_base_1.0.safetensors' } },
})
const f14 = await rectRun({ workflow: inlineTextGraph('a <subject> b'), params: { subject: 'cat' } }, 'f14-0001')
assert('F-14 an embedded <name> is replaced as text inside the surrounding string',
  f14.threw === null && f14.body?.['6']?.inputs?.text === 'a cat b',
  JSON.stringify({ threw: String(f14.threw ?? ''), text: f14.body?.['6']?.inputs?.text ?? null }))

const f15 = await rectRun({ workflow: inlineTextGraph('<a: b> c'), params: { a: 'x' } }, 'f15-0001')
assert('F-15 E-1: a malformed placeholder is refused with node, input and a value snippet',
  f15.threw !== null && /占位符语法非法/.test(f15.threw.message) && f15.threw.message.includes('node "6".text')
  && f15.threw.message.includes('<a: b> c') && noRow('f15-0001') && f15.body === undefined,
  JSON.stringify({ message: String(f15.threw?.message ?? ''), submitted: f15.body !== undefined }))

const f16 = await rectRun({ workflow: inlineTextGraph('x <pos: y> z'), params: { pos: 'p' } }, 'f16-0001')
assert('F-16 E-1: a default-carrying placeholder inside a string is refused',
  f16.threw !== null && /占位符语法非法/.test(f16.threw.message) && noRow('f16-0001'),
  JSON.stringify(String(f16.threw?.message ?? '')))

const f17 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: POSTER_PARAMS }, 'f17-0001')
const f17Diff = fidelityDiff(posterCard, f17.body)
assert('F-17 the fill touches only the placeholder positions (class types, links, node set untouched)',
  f17Diff.length === 0 && f17.body?.['4']?.inputs?.batch_size === 1 && f17.body?.['5']?.inputs?.denoise === 1.0
  && JSON.stringify(f17.body?.['2']?.inputs?.clip) === JSON.stringify(['1', 1])
  && JSON.stringify(f17.body?.['5']?.inputs?.latent_image) === JSON.stringify(['4', 0]),
  JSON.stringify({ diff: f17Diff, batch: f17.body?.['4']?.inputs?.batch_size ?? null, clip: f17.body?.['2']?.inputs?.clip ?? null }))

const cnCard = readRecipeWorkflow('controlnet-guide')
const CN_PARAMS = {
  ckpt_name: 'sd_xl_base_1.0.safetensors', positive: 'p', negative: 'n',
  width: 896, height: 1152, steps: 20, cfg: 6.0, sampler_name: 'euler', scheduler: 'normal',
  filename_prefix: 'cn', seed: 3,
  cn_ref_image: 'm1_s1.png', cn_model: 'diffusion_pytorch_model_promax.safetensors',
  cn_type: 'canny/lineart/anime_lineart/mlsd', cn_strength: 0.7, cn_start_percent: 0.0, cn_end_percent: 1.0,
}
const f18 = await rectRun({ workflow: readRecipeWorkflow('controlnet-guide'), params: CN_PARAMS }, 'f18-0001')
assert('F-18 the six cn_* placeholders go through the same mechanism, with numeric types kept',
  f18.threw === null && f18.result?.status === 'completed'
  && f18.body?.['8']?.inputs?.image === 'm1_s1.png'
  && f18.body?.['10']?.inputs?.control_net_name === 'diffusion_pytorch_model_promax.safetensors'
  && f18.body?.['11']?.inputs?.type === 'canny/lineart/anime_lineart/mlsd'
  && f18.body?.['12']?.inputs?.strength === 0.7 && typeof f18.body['12'].inputs.strength === 'number'
  && f18.body?.['12']?.inputs?.start_percent === 0 && typeof f18.body['12'].inputs.start_percent === 'number'
  && f18.body?.['12']?.inputs?.end_percent === 1 && typeof f18.body['12'].inputs.end_percent === 'number'
  && fidelityDiff(cnCard, f18.body).length === 0,
  JSON.stringify({ threw: String(f18.threw ?? ''), diff: fidelityDiff(cnCard, f18.body), node12: f18.body?.['12']?.inputs ?? null }))

const f19 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: { ...POSTER_PARAMS, seed: 99 }, seed: 5 }, 'f19-0001')
assert('F-19 a top-level seed still wins over the one injected from `params`',
  f19.threw === null && f19.body?.['5']?.inputs?.seed === 5 && f19.result?.seed === 5,
  JSON.stringify({ submitted: f19.body?.['5']?.inputs?.seed ?? null, result: f19.result?.seed ?? null }))

const f20 = await rectRun({ workflow: readRecipeWorkflow('poster-sdxl-base'), params: { ...POSTER_PARAMS, seed: 99 } }, 'f20-0001')
const f20Row = rectRows().find((entry) => entry.runLabel === 'rect-g1-f20-0001')
assert('F-20 an injected numeric seed is adopted by seed resolution and recorded',
  f20.threw === null && f20.body?.['5']?.inputs?.seed === 99
  && f20.result?.seeds?.['5.seed'] === 99 && f20Row?.seeds?.['5.seed'] === 99,
  JSON.stringify({ submitted: f20.body?.['5']?.inputs?.seed ?? null, result: f20.result?.seeds ?? null, row: f20Row?.seeds ?? null }))

const f21 = await rectRun({ template: 'txt2img', params: { width: 512 } }, 'f21-0001')
const f21Row = rectRows().find((entry) => entry.runLabel === 'rect-g1-f21-0001')
assert('F-21 `params` on the template path injects nothing and is not an error',
  f21.threw === null && f21.result?.status === 'completed' && f21.body?.['5']?.inputs?.width === 1024,
  JSON.stringify({ threw: String(f21.threw ?? ''), width: f21.body?.['5']?.inputs?.width ?? null, error: f21.result?.error ?? null }))
assert('F-21 unused names are recorded as params_ignored on the run',
  JSON.stringify(f21Row?.params?.params_ignored) === JSON.stringify(['width']),
  JSON.stringify(f21Row?.params ?? null))

const f22array = await rectRun({ template: 'txt2img', params: [1, 2] }, 'f22a-0001')
const f22scalar = await rectRun({ template: 'txt2img', params: 'x' }, 'f22b-0001')
assert('F-22 E-5: `params` that is not an object is refused before anything is submitted',
  f22array.threw?.message === 'comfyui_run: params must be an object keyed by placeholder name'
  && f22scalar.threw?.message === 'comfyui_run: params must be an object keyed by placeholder name'
  && noRow('f22a-0001') && noRow('f22b-0001'),
  JSON.stringify({ array: String(f22array.threw?.message ?? ''), scalar: String(f22scalar.threw?.message ?? '') }))

rectMount.dispose()
rectServer.close()
rmSync(rectMount.dataDir, { recursive: true, force: true })

slowServer.close()
for (const dir of [concurrentDir, asyncDir, mounted.dataDir]) rmSync(dir, { recursive: true, force: true })
for (const takeMount of takeoverMounts) {
  takeMount.dispose()
  rmSync(takeMount.dataDir, { recursive: true, force: true })
}
server.close()
for (const dir of [downloads, corruptDir, fetchDir]) rmSync(dir, { recursive: true, force: true })

for (const check of checks) console.log(`  ${check.ok ? 'OK  ' : 'FAIL'} ${check.label}${check.ok || check.detail === '' ? '' : ` — ${check.detail}`}`)
console.log(`\n结论：${checks.length - failures.length}/${checks.length} 项通过`)
if (failures.length > 0) {
  console.log(`FAIL ${failures.length} 项：`)
  for (const failure of failures) console.log(`  - ${failure}`)
}
process.exit(failures.length === 0 ? 0 : 1)
