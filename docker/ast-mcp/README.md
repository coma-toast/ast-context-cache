# ast-mcp Docker (process keep-alive)

Run the **ast-mcp binary** in a container with `restart: unless-stopped`.

This is **not** the Docker Model Runner (DMR) embedding backend. DMR notes live in the parent [`docker/README.md`](../README.md).

## Why prebuilt binary?

`ast-mcp` is CGO-linked (tokenizers + optional ONNX Runtime). A reliable multi-stage in-container Go build needs matching ORT headers/libs per arch. This image **COPYs a Linux `ast-mcp`** you built on the host or in CI.

## Quick start

1. Produce a **Linux** binary and place it next to the Dockerfile:

   ```bash
   # On Linux (or a Linux CI builder):
   make build
   cp ast-mcp docker/ast-mcp/ast-mcp
   ```

2. Optional ONNX: place `libonnxruntime.so*` under `docker/ast-mcp/lib/`, **or** skip ONNX and set `EMBED_BACKEND` to `ollama` / `http` / `openai` / `docker` in `compose.yml`.

3. Start:

   ```bash
   docker compose -f docker/ast-mcp/compose.yml up -d --build
   ```

4. Check: MCP `http://localhost:7821/health`, dashboard `http://localhost:7830`.

Data persists in the `astcache` volume (`HOME=/data` → `/data/.astcache`).

## Mount-only alternative

If you already have a Linux binary and ORT on the host:

```yaml
# example override — mount instead of COPY
volumes:
  - /path/to/linux/ast-mcp:/usr/local/bin/ast-mcp:ro
  - /path/to/libonnxruntime.so:/usr/local/lib/libonnxruntime.so:ro
  - astcache:/data/.astcache
```

## Shell keep-alive (no Docker)

On the host: `ast-mcp supervise` (after `make install`). See [skills/operator/SKILL.md](../../skills/operator/SKILL.md).
