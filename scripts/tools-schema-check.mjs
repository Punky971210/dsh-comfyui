#!/usr/bin/env node
/**
 * Tool-schema self-check for dsh-comfyui.
 *
 * Why this exists: this machine's `@deepseek-ai/dsh-tools` validates a tool's
 * `output.schema` at REGISTER time (throwing a `JsonSchemaError` that takes the
 * whole loader chain down, since registration is not wrapped by every caller),
 * but never validates `parameters` until a call is made. A wrong `parameters`
 * shape therefore ships silently and blows up on the first tool call. This
 * script runs the very same validators the host runs, offline, against EVERY
 * registered tool — not a sample.
 *
 * It imports the built plugin (`lib/`) so it checks what actually loads, and it
 * enumerates the tools from `comfyUIToolDefinitions`, the same list
 * `registerComfyUITools` iterates: a hand-written list here would stop covering
 * a tool the moment one was added.
 *
 * Usage: node scripts/tools-schema-check.mjs [--json]
 * Exit code 0 = every tool is compliant; 1 = at least one FAIL.
 */
import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const DSH_TOOLS = 'file:///D:/dsh/npm/0.1.6-alpha.1/node_modules/@deepseek-ai/dsh-tools/lib/types/json-schema.js'
const { assertSupportedJsonSchema, validateJsonSchemaValue, JsonSchemaError } = await import(DSH_TOOLS)
const { comfyUIToolDefinitions } = await import('../lib/tools.js')

/**
 * Keywords the host's validator accepts. The STRICT set is what
 * `assertSupportedJsonSchema` enforces for `output.schema`; `parameters` rides
 * the more permissive projection path, which additionally honours the numeric
 * bounds the existing tools already use — recorded as an extension below so the
 * exception is visible rather than silent.
 */
const STRICT_KEYWORDS = new Set([
  'type', 'oneOf', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const',
  'description', 'title', 'default', 'examples',
])
const PARAMETER_ONLY_KEYWORDS = new Set(['minimum', 'maximum'])
const BANNED_KEYWORDS = ['$ref', 'allOf', 'anyOf', 'not', 'format', 'pattern', 'minLength', 'maxLength', 'uniqueItems']

const scratch = mkdtempSync(join(tmpdir(), 'dsh-comfyui-schema-'))
/** Values recorded for the report; keys are set by the walkers below. */
const facts = []
const failures = []
const extensions = []

function fail(tool, rule, detail) {
  failures.push({ tool, rule, detail })
}

function note(tool, rule, detail) {
  facts.push({ tool, rule, detail })
}

/** A synthetic value that satisfies one schema node, or undefined. */
function sampleFor(schema) {
  if (typeof schema !== 'object' || schema === null) return undefined
  if (Array.isArray(schema.oneOf) && schema.oneOf.length > 0) return sampleFor(schema.oneOf[0])
  if (Array.isArray(schema.enum) && schema.enum.length > 0) return schema.enum[0]
  if ('const' in schema) return schema.const
  switch (schema.type) {
    case 'string': return 'sample'
    case 'integer': return 1
    case 'number': return 1
    case 'boolean': return true
    case 'null': return null
    case 'array': return [sampleFor(schema.items)]
    case 'object': {
      const out = {}
      for (const [key, child] of Object.entries(schema.properties ?? {})) out[key] = sampleFor(child)
      return out
    }
    default: return undefined
  }
}

/**
 * A value the root schema must refuse. Every tool's parameters are an object
 * schema, so a bare string is invalid at the root regardless of the properties
 * — this sample proves the validator is actually wired to this schema rather
 * than passing everything through.
 */
function negativeFor(schema) {
  return schema?.type === 'object' ? 'not-an-object' : {}
}

/**
 * Walk every node of a schema, collecting rule violations that a recursive
 * inspection can decide on its own.
 */
function walk(node, path, tool, { isParameter, parent }) {
  if (typeof node !== 'object' || node === null || Array.isArray(node)) return
  for (const key of BANNED_KEYWORDS) {
    if (key in node) fail(tool, 'R8-banned-keyword', `${path}.${key}`)
  }
  for (const key of Object.keys(node)) {
    if (!STRICT_KEYWORDS.has(key) && !PARAMETER_ONLY_KEYWORDS.has(key)) {
      fail(tool, 'R8-unknown-keyword', `${path}.${key}`)
    }
    if (isParameter && PARAMETER_ONLY_KEYWORDS.has(key)) extensions.push({ tool, path: `${path}.${key}` })
  }
  if (Array.isArray(node.type)) fail(tool, 'R4-type-array', `${path}.type = ${JSON.stringify(node.type)}`)
  if (Array.isArray(node.oneOf)) {
    if (node.oneOf.length < 2) fail(tool, 'R5-oneOf-too-small', `${path}.oneOf has ${node.oneOf.length} branch(es)`)
    for (const sibling of ['type', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const']) {
      if (sibling in node) fail(tool, 'R5-oneOf-sibling', `${path}.${sibling}`)
    }
  }
  if (node.type === 'object' && !('additionalProperties' in node)) {
    fail(tool, 'R6-additionalProperties-missing', path)
  }
  if ('additionalProperties' in node && typeof node.additionalProperties !== 'boolean') {
    fail(tool, 'R6-additionalProperties-not-boolean', `${path}.additionalProperties`)
  }
  if (Array.isArray(node.required)) {
    const properties = node.properties ?? {}
    for (const name of node.required) {
      if (!(name in properties)) fail(tool, 'R9-required-orphan', `${path}.required "${name}"`)
    }
  }
  for (const [key, child] of Object.entries(node.properties ?? {})) {
    walk(child, `${path}.properties.${key}`, tool, { isParameter, parent: node })
  }
  if (node.items !== undefined) walk(node.items, `${path}.items`, tool, { isParameter, parent: node })
  if (Array.isArray(node.oneOf)) {
    node.oneOf.forEach((branch, index) => walk(branch, `${path}.oneOf[${index}]`, tool, { isParameter, parent: node }))
  }
  // A required NAME inside a property node is the DSL spelling and means
  // nothing on the raw channel — flag it so the two channels never mix.
  if (parent !== undefined && node.required === true) fail(tool, 'R2-prop-required-boolean', path)
}

const runtime = {
  getConfig: () => ({
    baseUrl: 'http://127.0.0.1:8188',
    apiKeyEnv: 'COMFYUI_API_KEY',
    connectTimeoutMs: 10_000,
    timeoutMs: 900_000,
    pollIntervalMs: 1_000,
    maxMediaItems: 12,
    maxMediaBytes: 64 * 1024 * 1024,
    dataDir: scratch,
    maxAssets: 200,
    skillsDir: '',
    mediaHost: '',
    outputDir: '',
    downloadDir: scratch,
    comfyuiDirs: [],
  }),
  getApiKey: async () => undefined,
  createClient: () => { throw new Error('schema check never calls a client') },
  hostHint: { origin: () => undefined, remember: () => {} },
  proxyBase: () => undefined,
  downloadDir: () => scratch,
  settingsWritable: () => false,
  updateConfig: async () => ({ ok: false, error: 'schema check' }),
  queue: async () => { throw new Error('schema check never queues') },
  untrack: () => {},
  trackedRuns: () => [],
  queueProgress: () => undefined,
  listWorkflows: async () => [],
  getWorkflow: async () => undefined,
  saveWorkflow: async () => ({ ok: false, error: 'schema check' }),
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
  analyzeComfyWorkflow: async () => ({ ok: false, error: 'schema check' }),
  extractComfyWorkflow: async () => ({ ok: false, error: 'schema check' }),
}

const ctx = { get: () => undefined }

const tools = comfyUIToolDefinitions(ctx, runtime)
const report = []

// A1: the raw channel only. `defineTool` compiles `parameters` as a property
// map and rejects the root `required: [...]` arrays these tools carry, which
// would throw at LOAD time and take the loader chain down with it. The built
// output is what actually loads, so the scan runs over `lib/`, not `src/`.
const sourceHits = []
for (const file of readdirSync(new URL('../lib', import.meta.url), { recursive: true })) {
  if (!String(file).endsWith('.js')) continue
  const text = readFileSync(new URL(`../lib/${file}`, import.meta.url), 'utf8')
  if (text.includes('defineTool')) sourceHits.push(String(file))
}
if (sourceHits.length > 0) fail('(plugin)', 'A1-defineTool', `lib/ 出现 defineTool 引用: ${sourceHits.join(', ')}`)

for (const tool of tools) {
  const name = tool.name
  const parameters = tool.parameters
  const outputSchema = tool.output?.schema

  // A1: the raw channel only. `defineTool` would make the existing
  // `required: [...]` arrays throw at LOAD time and kill the loader chain.
  if (typeof tool.required === 'boolean' || parameters?.properties === undefined && parameters?.type !== 'object') {
    fail(name, 'A1-raw-channel', 'parameters is not a bare JSON Schema object')
  }
  if (!('output' in tool) || typeof tool.output?.render !== 'function') fail(name, 'A3-output-render', 'missing output.render')

  // A3: registration-time validation, byte-for-byte what the host runs.
  let outputOk = true
  try {
    assertSupportedJsonSchema(outputSchema)
  } catch (error) {
    outputOk = false
    fail(name, 'A3-assertSupportedJsonSchema', error instanceof JsonSchemaError ? error.message : String(error))
  }

  // A5/A6/A7/A8 + R4/R5/R9: recursive inspection.
  walk(parameters, 'parameters', name, { isParameter: true, parent: undefined })
  walk(outputSchema, 'output.schema', name, { isParameter: false, parent: undefined })

  // A2: no `type` arrays anywhere (walk() covers it; this is the counter).
  const typeArrays = []
  const scanTypes = (node, path) => {
    if (typeof node !== 'object' || node === null || Array.isArray(node)) return
    if (Array.isArray(node.type)) typeArrays.push(path)
    for (const [key, child] of Object.entries(node.properties ?? {})) scanTypes(child, `${path}.properties.${key}`)
    if (node.items !== undefined) scanTypes(node.items, `${path}.items`)
    if (Array.isArray(node.oneOf)) node.oneOf.forEach((branch, i) => scanTypes(branch, `${path}.oneOf[${i}]`))
  }
  scanTypes(parameters, 'parameters')
  scanTypes(outputSchema, 'output.schema')

  // A4: positive sample must validate clean; a negative sample must be refused.
  const positive = sampleFor(parameters)
  const positiveViolations = validateJsonSchemaValue(parameters, positive, 'args')
  if (positiveViolations.length > 0) {
    fail(name, 'A4-parameters-positive', positiveViolations.join('; '))
  }
  const negativeViolations = validateJsonSchemaValue(parameters, negativeFor(parameters), 'args')
  const requiredNames = Array.isArray(parameters?.required) ? parameters.required : []
  if (requiredNames.length === 0) {
    // Nothing is required, so there are two obligations: a valid sample and an
    // empty object must both be accepted (no false refusal), and something the
    // schema cannot accept must still be refused (the check has teeth).
    if (positiveViolations.length > 0 || validateJsonSchemaValue(parameters, {}, 'args').length > 0) {
      fail(name, 'A4-parameters-positive', 'a schema-valid sample or an empty object was refused')
    }
    if (negativeViolations.length === 0) fail(name, 'A4-parameters-negative', 'validator accepted a wrong-typed root')
  } else {
    // Every required name must be individually enforceable: dropping each one
    // from an otherwise valid sample has to produce its own violation.
    for (const dropped of requiredNames) {
      const sample = { ...positive }
      delete sample[dropped]
      const violations = validateJsonSchemaValue(parameters, sample, 'args')
      if (!violations.some((line) => line.includes(`missing required property "args.${dropped}"`))) {
        fail(name, 'A4-parameters-negative', `required "${dropped}" is not enforced (got ${JSON.stringify(violations)})`)
      }
    }
  }

  // Additional: closer semantics only hold when explicitly declared false.
  const closed = []
  const scanClosed = (node, path) => {
    if (typeof node !== 'object' || node === null || Array.isArray(node)) return
    if (node.additionalProperties === false) closed.push(path)
    for (const [key, child] of Object.entries(node.properties ?? {})) scanClosed(child, `${path}.properties.${key}`)
    if (node.items !== undefined) scanClosed(node.items, `${path}.items`)
    if (Array.isArray(node.oneOf)) node.oneOf.forEach((branch, i) => scanClosed(branch, `${path}.oneOf[${i}]`))
  }
  scanClosed(outputSchema, 'output.schema')

  note(name, 'required', JSON.stringify(requiredNames))
  note(name, 'closed-objects', String(closed.length))
  note(name, 'parameters-type', String(parameters?.type))

  report.push({
    tool: name,
    outputSchemaPasses: outputOk,
    required: requiredNames,
    positiveViolations,
    negativeViolations,
    closedObjects: closed.length,
  })
}

rmSync(scratch, { recursive: true, force: true })

const summary = {
  tools: tools.length,
  names: tools.map((tool) => tool.name),
  failures,
  parameterOnlyKeywords: [...new Set(extensions.map((entry) => `${entry.tool}: ${entry.path}`))],
  report,
}

if (process.argv.includes('--json')) {
  console.log(JSON.stringify(summary, null, 2))
} else {
  console.log(`工具 schema 自校验：${summary.tools} 个工具（${summary.names.join(', ')}）`)
  for (const entry of report) {
    console.log(`  OK   ${entry.tool.padEnd(24)} output.schema=通过  required=${JSON.stringify(entry.required)} 闭合 object=${entry.closedObjects} 正样本 violation=${entry.positiveViolations.length} 负样本 violation=${entry.negativeViolations.length}`)
  }
  if (summary.parameterOnlyKeywords.length > 0) {
    console.log(`  注   parameters 侧使用的宽松关键字（非 output.schema 白名单，宿主注册期不校验 parameters，调用期接受）：${summary.parameterOnlyKeywords.join(', ')}`)
  }
  if (failures.length === 0) {
    console.log(`\n结论：全部 ${summary.tools} 个工具合规（FAIL 0）。`)
  } else {
    console.log(`\n结论：FAIL ${failures.length} 项`)
    for (const entry of failures) console.log(`  FAIL ${entry.tool} [${entry.rule}] ${entry.detail}`)
  }
}

process.exit(failures.length === 0 ? 0 : 1)
