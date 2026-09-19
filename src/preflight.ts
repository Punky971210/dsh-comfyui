/**
 * Submit-time preflight against the server's node definitions.
 *
 * `comfyui_run` used to hand the workflow straight to `/prompt` and let
 * ComfyUI reject it. That rejection arrives after the prompt is queued, which
 * costs a round trip and — for a workflow naming a model that is not on disk —
 * can make ComfyUI try to *fetch* the model. This module answers the question
 * before anything is submitted: are the class types registered, and are the
 * model/option values the loaders ask for actually present in `object_info`?
 *
 * It is read-only by construction: the only input is an `object_info` snapshot
 * the caller already fetched. Nothing here downloads, installs, or mutates a
 * server-side file — a model that is not on disk is reported, never obtained.
 */
import type { ComfyUIHistoryEntry } from './comfyui.js'

/** Node-input spec shapes as `object_info` reports them. */
interface NodeDefinition {
  input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> }
}

/** A workflow node as the tools accept it. */
type WorkflowNode = { class_type: string; inputs: Record<string, unknown> }

/** One reason the workflow was refused, addressed to a node. */
export interface PreflightProblem {
  nodeId: string
  classType: string
  inputKey?: string
  /** The value that is not available on the server (model name / option). */
  value?: string
  code: 'UNKNOWN_CLASS' | 'MISSING_VALUE'
  message: string
}

/** The preflight verdict. */
export interface PreflightResult {
  ok: boolean
  problems: PreflightProblem[]
  /** Class types that could not be checked because no definition was found. */
  unknownClasses: string[]
}

/** The outbound error shape carried on a refused run. */
export interface RunFailure {
  code: string
  message: string
  /** Missing items / node-level detail, aligned with the glue `nodeErrors` field. */
  nodeErrors: Array<{ node: string; classType: string; input: string; value: string; reason: string }> | null
}

const MAX_PROBLEMS = 20
const MAX_REPORTED_VALUES = 8

function isNodeDefinition(value: unknown): value is NodeDefinition {
  return typeof value === 'object' && value !== null
}

/** The declared option list of one input, or undefined when it is not a list. */
function declaredOptions(definition: NodeDefinition | undefined, inputKey: string): string[] | undefined {
  if (definition === undefined) return undefined
  const spec = definition.input?.required?.[inputKey] ?? definition.input?.optional?.[inputKey]
  if (!Array.isArray(spec)) return undefined
  const first = spec[0]
  if (Array.isArray(first)) {
    return first.filter((value): value is string => typeof value === 'string')
  }
  if (first === 'COMFY_DYNAMICCOMBO_V3') {
    const meta = spec[1]
    const options = meta !== null && typeof meta === 'object' ? (meta as { options?: unknown }).options : undefined
    if (!Array.isArray(options)) return undefined
    return options
      .map((option) => (option !== null && typeof option === 'object' ? (option as { key?: unknown }).key : undefined))
      .filter((value): value is string => typeof value === 'string')
  }
  if (first === 'COMBO') {
    const meta = spec[1]
    const options = meta !== null && typeof meta === 'object' ? (meta as { options?: unknown }).options : undefined
    if (!Array.isArray(options)) return undefined
    return options.filter((value): value is string => typeof value === 'string')
  }
  return undefined
}

/**
 * Check a workflow against one `object_info` snapshot.
 *
 * Only two things are checked, both of which ComfyUI would reject later anyway:
 * the class type must be registered, and a loader's model name must be one of
 * the values the server reports for that input. Inputs whose declared spec is
 * not a value list (free strings, numbers, links) are skipped — guessing at
 * them would refuse valid workflows.
 */
export function preflightWorkflow(
  workflow: Record<string, WorkflowNode>,
  objectInfo: Record<string, unknown>,
): PreflightResult {
  const problems: PreflightProblem[] = []
  const unknownClasses = new Set<string>()
  const definitionOf = (classType: string): NodeDefinition | undefined => {
    const raw = objectInfo[classType]
    return isNodeDefinition(raw) ? raw : undefined
  }

  for (const [nodeId, node] of Object.entries(workflow)) {
    if (problems.length >= MAX_PROBLEMS) break
    const classType = node?.class_type
    if (typeof classType !== 'string' || classType === '') continue
    const definition = definitionOf(classType)
    if (definition === undefined) {
      unknownClasses.add(classType)
      problems.push({
        nodeId,
        classType,
        code: 'UNKNOWN_CLASS',
        message: `节点类型 "${classType}" 未在服务端注册（object_info 无此 class_type）`,
      })
      continue
    }
    for (const [inputKey, raw] of Object.entries(node.inputs ?? {})) {
      if (problems.length >= MAX_PROBLEMS) break
      // Links ([nodeId, slot]) and numbers carry no model name to check.
      if (typeof raw !== 'string' || raw === '') continue
      const options = declaredOptions(definition, inputKey)
      if (options === undefined || options.length === 0) continue
      if (options.includes(raw)) continue
      const shown = options.slice(0, MAX_REPORTED_VALUES).join(', ')
      const more = options.length > MAX_REPORTED_VALUES ? ` …（共 ${options.length} 项）` : ''
      problems.push({
        nodeId,
        classType,
        inputKey,
        value: raw,
        code: 'MISSING_VALUE',
        // The refusal is the whole point of the red line: the value is missing
        // on disk, and listing the available ones is what lets the caller fix
        // it without a download ever being attempted.
        message: `节点 ${nodeId}(${classType}).${inputKey} = "${raw}" 不在服务端可用值内（可选: ${shown}${more}）。请先让用户把该模型放到 ComfyUI 对应目录，本工具不会自动下载。`,
      })
    }
  }

  return { ok: problems.length === 0, problems, unknownClasses: [...unknownClasses] }
}

/**
 * Whether the server can even answer the preflight. An unavailable
 * `object_info` is NOT "assume ok": a run submitted while the definitions
 * cannot be read is a run that skipped every check, so it is refused with its
 * own code (distinct from a model that is missing).
 */
export function preflightUnavailable(cause: string): RunFailure {
  return {
    code: 'PREFLIGHT_UNAVAILABLE',
    message: `提交前预检无法完成：读取服务端节点定义失败（${cause}）。未提交任何任务，也未触发任何模型下载。`,
    nodeErrors: null,
  }
}

/** Build the structured refusal for a workflow that failed the preflight. */
export function preflightFailure(result: PreflightResult, runDir: string): RunFailure {
  const nodeErrors = result.problems.map((problem) => ({
    node: problem.nodeId,
    classType: problem.classType,
    input: problem.inputKey ?? '',
    value: problem.value ?? '',
    reason: problem.code,
  }))
  const head = result.problems.slice(0, 3).map((problem) => `${problem.nodeId}.${problem.classType}${problem.inputKey !== undefined ? `.${problem.inputKey}` : ''}`)
  return {
    code: 'PREFLIGHT',
    message: `提交前预检未通过（${result.problems.length} 项）: ${head.join('; ')}${result.problems.length > head.length ? ' …' : ''}。模型/节点未就位时不会提交、也不会自动下载；补齐后重试。台账目录: ${runDir}`,
    nodeErrors,
  }
}

/**
 * Extract ComfyUI's per-node execution errors from a history entry.
 *
 * The server reports failures as `status.messages` entries whose first element
 * is the event name and whose second is an object; `execution_error` carries
 * the node type/id and the exception text. Flattening those into a list is what
 * lets a caller see *which* node failed instead of one opaque string.
 */
export function extractNodeErrors(entry: ComfyUIHistoryEntry | undefined): RunFailure['nodeErrors'] {
  const messages = entry?.status?.messages
  if (!Array.isArray(messages) || messages.length === 0) return null
  const errors: NonNullable<RunFailure['nodeErrors']> = []
  for (const message of messages) {
    if (!Array.isArray(message) || typeof message[0] !== 'string') continue
    const payload = message[1]
    if (typeof payload !== 'object' || payload === null) continue
    const record = payload as Record<string, unknown>
    const text = typeof record.exception_message === 'string'
      ? record.exception_message
      : typeof record.exception_type === 'string' ? record.exception_type : undefined
    if (text === undefined) continue
    errors.push({
      node: typeof record.node_id === 'string' || typeof record.node_id === 'number' ? String(record.node_id) : '',
      classType: typeof record.node_type === 'string' ? record.node_type : '',
      input: typeof record.input_name === 'string' ? record.input_name : '',
      value: typeof record.input_value === 'string' ? record.input_value.slice(0, 120) : '',
      reason: text.slice(0, 300),
    })
  }
  return errors.length > 0 ? errors : null
}


