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

const { ComfyUIClient } = await import('../lib/comfyui.js')
const { comfyUIToolDefinitions } = await import('../lib/tools.js')
const { readLedger } = await import('../lib/ledger.js')

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

server.close()
for (const dir of [downloads, corruptDir, fetchDir]) rmSync(dir, { recursive: true, force: true })

for (const check of checks) console.log(`  ${check.ok ? 'OK  ' : 'FAIL'} ${check.label}${check.ok || check.detail === '' ? '' : ` — ${check.detail}`}`)
console.log(`\n结论：${checks.length - failures.length}/${checks.length} 项通过`)
if (failures.length > 0) {
  console.log(`FAIL ${failures.length} 项：`)
  for (const failure of failures) console.log(`  - ${failure}`)
}
process.exit(failures.length === 0 ? 0 : 1)
