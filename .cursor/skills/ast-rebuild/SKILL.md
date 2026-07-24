---
name: ast-context-cache-rebuild
description: Rebuild and restart ast-mcp after source changes. Use when the user says rebuild ast-mcp, restart ast-mcp, or after editing this repository's server code.
---

# Rebuild & Restart ast-context-cache

Builds the server binary from the **current git repo root** and restarts the process.

- MCP: `http://localhost:7821/mcp`
- Dashboard: `http://localhost:7830`

## Steps

### 1. Build

From the repository root (where `Makefile` lives):

```bash
make build
```

`make build` rebuilds the React dashboard (`ui-build`) and copies `VERSION` → `internal/version/VERSION`. After `git pull`, prefer `make build` over bare `go build` so the SPA assets stay in sync.

If the build fails, stop and show errors. Do not kill the running server until build succeeds.

### 2. Stop the running server

Prefer the shell helper if installed:

```bash
ast-mcp stop
```

Otherwise, from repo root:

```bash
pkill -f './ast-mcp' || pkill -f 'ast-mcp' || true
sleep 1
pgrep -f 'ast-mcp' || echo "stopped"
```

### 3. Start

```bash
ast-mcp start
```

Or foreground from repo root (`block_until_ms: 0` if running in agent shell):

```bash
./ast-mcp
```

### 4. Verify

```bash
curl -s http://localhost:7821/health
```

Expected healthy JSON with `"service":"ast-context-cache"`.

### 5. Report

Confirm build and health check result to the user.
