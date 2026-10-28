// ast-context-cache search-first nudge for OpenCode.
// Written by `ast-mcp install --target opencode --component hooks`; `ast-mcp uninstall` removes it.
// Blocks a session's first grep, glob, or whole-file read of a code file once, until an
// ast-context-cache tool has been used, and names the tool to use instead. Every later call goes
// through. Turn it off with the feature_search_nudge flag (dashboard Settings -> Features) or
// AST_FEATURE_SEARCH_NUDGE=false. Any error lets the call through.
const HEALTH_URL = __HEALTH_URL__
const MESSAGES = __MESSAGES__
const CODE_EXTENSIONS = __CODE_EXTENSIONS__
const AST_TOOLS = __AST_TOOLS__
const HEALTH_TIMEOUT_MS = 1500

// Sessions already nudged or already using ast-context-cache.
const done = new Set()

function isAstTool(name) {
  const n = String(name || "").toLowerCase()
  if (n.includes("ast-context-cache") || n.includes("ast_context_cache")) return true
  return AST_TOOLS.some((t) => n === t || n.endsWith("_" + t) || n.endsWith("-" + t) || n.endsWith("." + t))
}

function nudgeKind(tool, args) {
  switch (String(tool || "").toLowerCase()) {
    case "grep":
      return "grep"
    case "glob":
      return "glob"
    case "read": {
      if (args.offset != null || args.limit != null) return ""
      const path = String(args.filePath || args.file_path || args.path || "")
      const base = path.slice(path.lastIndexOf("/") + 1)
      const dot = base.lastIndexOf(".")
      return dot > 0 && CODE_EXTENSIONS.includes(base.slice(dot).toLowerCase()) ? "read" : ""
    }
  }
  return ""
}

async function nudgeEnabled() {
  const env = String(process.env.AST_FEATURE_SEARCH_NUDGE || "").trim().toLowerCase()
  if (["0", "f", "false"].includes(env)) return false
  try {
    const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) })
    if (!res.ok) return false
    const body = await res.json()
    return body.search_nudge === true
  } catch {
    return false
  }
}

export const AstContextCacheSearchNudge = async () => ({
  "tool.execute.before": async (input, output) => {
    let kind = ""
    try {
      const session = input && input.sessionID
      if (!session || done.has(session)) return
      if (isAstTool(input.tool)) {
        done.add(session)
        return
      }
      kind = nudgeKind(input.tool, (output && output.args) || {})
      if (!kind || !(await nudgeEnabled()) || done.has(session)) return
      done.add(session)
    } catch {
      return
    }
    throw new Error(MESSAGES[kind])
  },
})
