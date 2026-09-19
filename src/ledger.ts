/**
 * Local run ledger (`runs.json`) and the output-file layout shared by
 * `comfyui_run` and `comfyui_fetch_output`.
 *
 * ComfyUI's own `/history` is in-memory and dropped on restart or a "clear
 * history" click, so it cannot answer "what did this run actually use?" after
 * the fact. This ledger is the durable answer, and it is deliberately minimal:
 *
 * - **Identity keyed by `runLabel` when present, else `promptId`.** `runLabel`
 *   is the governance key (`<批次>-<lane>-<job>`) a caller can predict and
 *   address; `promptId` is the server-generated UUID, which only exists after
 *   a successful submit. A single run therefore starts as a `queued` record
 *   keyed by `runLabel` and is *overwritten in place* (never appended) once
 *   the terminal state is known — one row per run, not a queued row plus a
 *   result row. `comfyui_fetch_output` later fills the same row in by
 *   `promptId` (or by `promptId = view:<filename>` for a directly addressed
 *   file), so the downloaded paths land on the run that produced them.
 * - **The resolved seed is recorded**, so a replayed run reads the seed it
 *   actually used instead of re-randomizing.
 * - **Corruption is survivable.** Bookkeeping must never block generation: an
 *   unreadable/absent/hand-corrupted file degrades to an empty table plus a
 *   note, and the caller keeps going.
 *
 * Output files are never overwritten: `uniqueOutputPath` bumps the stem
 * (`image.png` → `image.01.png` → `image.02.png`), so two runs sharing a
 * `filename_prefix` both stay on disk.
 *
 * All file access is synchronous on purpose (matching the surrounding plugin
 * data helpers): ledger writes happen on the tool call path, where a pending
 * promise would let two concurrent writes interleave on the same file.
 * Encoding: UTF-8 without BOM, LF, zero third-party dependencies.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { extname, join } from 'node:path'

/** Name of the ledger file, suffixed onto the run directory. */
export const LEDGER_FILE = 'runs.json'

/** One file this run produced or downloaded. */
export interface LedgerFile {
  filename: string
  subfolder: string
  type: string
  /** Absolute path on this machine, once the file has been fetched. */
  absPath?: string
  size?: number
}

/** One run, keyed by `runLabel` when it has one and by `promptId` otherwise. */
export interface LedgerRecord {
  /** Governance key (`<批次>-<lane>-<job>`) — the preferred identity. */
  runLabel?: string
  /** Server-generated prompt UUID — the fallback identity. */
  promptId?: string
  /** Seed actually written into the submitted workflow (never -1, never null). */
  seed?: number
  /** All seed-typed inputs of the submitted workflow, by `nodeId.inputKey`. */
  seeds?: Record<string, number>
  /** The resolved inputs this run used, for replay. */
  params?: Record<string, unknown>
  /** Workflow name/template the run came from (display + audit). */
  workflowName?: string | null
  ts: string
  /** `queued` at submit time, overwritten by the terminal status in place. */
  status: 'queued' | 'completed' | 'failed' | 'interrupted'
  durationMs?: number
  media?: LedgerFile[]
  /** Files fetched to disk by `comfyui_fetch_output`. */
  files?: LedgerFile[]
  error?: { code?: string; message?: string; nodeErrors?: unknown }
}

/** Reading result: a corrupt ledger yields an empty table plus a note. */
export interface LedgerRead {
  records: LedgerRecord[]
  /** Non-null when the file existed but could not be read/parsed. */
  note: string | null
}

/** The absolute path of the ledger file for one run directory. */
export function ledgerPath(dir: string): string {
  return join(dir, LEDGER_FILE)
}

function isRecord(value: unknown): value is LedgerRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Read the ledger. An absent file is normal (first run) and returns no note;
 * a corrupt one returns the note so the caller can surface it without the
 * bookkeeping failure becoming a run failure.
 */
export function readLedger(dir: string): LedgerRead {
  let raw: string
  try {
    raw = readFileSync(ledgerPath(dir), 'utf8')
  } catch (error) {
    if ((error as { code?: string } | null)?.code === 'ENOENT') return { records: [], note: null }
    return { records: [], note: `runs.json 读取失败(${messageOf(error)})，按空表处理` }
  }
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return { records: [], note: 'runs.json 结构不是数组，按空表处理' }
    return { records: parsed.filter(isRecord), note: null }
  } catch (error) {
    return { records: [], note: `runs.json 解析失败(${messageOf(error)})，按空表处理` }
  }
}

/** Write the ledger back, creating the directory when needed. */
export function writeLedger(dir: string, records: LedgerRecord[]): string {
  mkdirSync(dir, { recursive: true })
  const file = ledgerPath(dir)
  writeFileSync(file, `${JSON.stringify(records, null, 2)}\n`, 'utf8')
  return file
}

function mergeRecord(previous: LedgerRecord, next: LedgerRecord): LedgerRecord {
  const merged: LedgerRecord = { ...previous, ...next }
  // `files`/`media` are whole-table fields: a later write that omits them
  // (a status-only overwrite) must not erase the paths already recorded.
  if (next.files === undefined && previous.files !== undefined) merged.files = previous.files
  if (next.media === undefined && previous.media !== undefined) merged.media = previous.media
  // The identity of the matching row is kept even when the update carries only
  // the other key (queued by runLabel, completed by promptId).
  if (merged.runLabel === undefined && previous.runLabel !== undefined) merged.runLabel = previous.runLabel
  if (merged.promptId === undefined && previous.promptId !== undefined) merged.promptId = previous.promptId
  return merged
}

/**
 * Insert or overwrite one run, keyed by `runLabel` first and `promptId`
 * second. This is what makes `queued → terminal` an overwrite of one row.
 * Returns the file path, the table size, and the read note (if any).
 */
export function upsertRun(dir: string, record: LedgerRecord): { file: string; records: number; note: string | null } {
  const { records, note } = readLedger(dir)
  const keyField: 'runLabel' | 'promptId' | undefined =
    typeof record.runLabel === 'string' && record.runLabel !== '' ? 'runLabel'
      : typeof record.promptId === 'string' && record.promptId !== '' ? 'promptId'
        : undefined
  // A record with neither identity would match every other identless row;
  // refusing the write keeps the table honest instead of merging unrelated runs.
  if (keyField === undefined) return { file: ledgerPath(dir), records: records.length, note }
  const key: unknown = record[keyField]
  const index = records.findIndex((entry) => entry[keyField] === key)
  if (index >= 0) records[index] = mergeRecord(records[index]!, record)
  else records.push({ ...record })
  const file = writeLedger(dir, records)
  return { file, records: records.length, note }
}

/** One run's ledger row, by `runLabel` first and `promptId` second. */
export function findRun(dir: string, key: string): LedgerRecord | undefined {
  const { records } = readLedger(dir)
  return records.find((entry) => entry.runLabel === key || entry.promptId === key)
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * One past the highest `<prefix>-<NNNN>` seen in the ledger. Both identity
 * fields are scanned, so the counter keeps climbing across a run that was
 * only ever recorded by `promptId`.
 */
export function nextJobNumber(dir: string, prefix: string): number {
  const { records } = readLedger(dir)
  const pattern = new RegExp(`^${escapeRegExp(prefix)}-(\\d+)$`)
  let max = 0
  for (const record of records) {
    for (const field of ['runLabel', 'promptId'] as const) {
      const value = record[field]
      if (typeof value !== 'string') continue
      const match = value.match(pattern)
      if (match === null) continue
      const parsed = Number(match[1])
      if (Number.isFinite(parsed)) max = Math.max(max, parsed)
      break
    }
  }
  return max + 1
}

/** The default governance prefix when the caller names no run label. */
export const DEFAULT_RUN_PREFIX = 'comfyui'

/** The governance prefix: `COMFYUI_RUN_PREFIX` when set, else `comfyui`. */
export function resolveRunPrefix(): string {
  const fromEnv = process.env['COMFYUI_RUN_PREFIX']
  return typeof fromEnv === 'string' && fromEnv.trim() !== '' ? fromEnv.trim() : DEFAULT_RUN_PREFIX
}

/** Compose `<prefix>-<NNNN>` from a prefix and a job number. */
export function formatRunLabel(prefix: string, job: number): string {
  return `${prefix}-${String(job).padStart(4, '0')}`
}

/**
 * A download path that never overwrites an existing file: the stem is bumped
 * with a 2-digit sequence (`image.png` → `image.01.png` → `image.02.png`).
 * `relPath` is resolved against `dir` unchanged (sub-directories included).
 */
export function uniqueOutputPath(dir: string, relPath: string): string {
  const direct = join(dir, relPath)
  if (!existsSync(direct)) return direct
  const ext = extname(relPath)
  const stem = ext === '' ? relPath : relPath.slice(0, -ext.length)
  for (let index = 1; ; index += 1) {
    const candidate = join(dir, `${stem}.${String(index).padStart(2, '0')}${ext}`)
    if (!existsSync(candidate)) return candidate
  }
}

/** Strip a server-reported media path down to a path that cannot escape `dir`. */
export function safeRelativePath(relPath: string): string {
  const normalized = relPath.replace(/\\/g, '/').replace(/^\/+/, '')
  const parts = normalized.split('/').filter((part) => part !== '')
  if (parts.length === 0 || parts.some((part) => part === '..' || part === '.' || part.includes(':'))) {
    throw new Error(`非法输出相对路径: ${relPath}`)
  }
  return parts.join('/')
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
