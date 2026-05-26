---
name: ast-context-cache-operator
description: Configure ast-mcp embedding backends, dashboard settings (7830), log indexing, ignore globs, pinning, or health checks—not routine MCP code search.
---

# ast-context-cache Operator

## When to Use

When configuring or operating the ast-mcp server (not day-to-day MCP search). Use when the user asks about:
- Embedding backends (ONNX, Ollama, HTTP, OpenAI/LiteLLM)
- Dashboard settings (port 7830)
- Log file indexing or retention
- Watcher ignore globs
- Health checks or server restart after config changes

## Embedding backends

Vectors must stay **768-dimensional** L2-normalized to match the index. Changing model or dimension requires re-index or clearing vectors.

| `EMBED_BACKEND` | Use case | Main env vars |
|-----------------|----------|---------------|
| `onnx` (default) | Local; `make setup` downloads model + tokenizer | `MODEL_DIR` |
| `ollama` | Ollama with 768-d model (e.g. `nomic-embed-text`) | `OLLAMA_HOST`, `OLLAMA_EMBED_MODEL` |
| `http` | Custom `POST` JSON embed service | `EMBED_HTTP_URL`, `EMBED_HTTP_BEARER` |
| `openai` / `litellm` | OpenAI-compatible `POST /v1/embeddings` | `EMBED_OPENAI_BASE_URL`, `EMBED_OPENAI_MODEL`, `EMBED_OPENAI_API_KEY`, `EMBED_OPENAI_DIMENSIONS` |

- **Process env** must export these vars for whatever starts `ast-mcp` (terminal, `ast-mcp start`, systemd).
- **Dashboard → Settings:** same keys saved to `~/.astcache/usage.db`; **non-empty env always overrides** SQLite.
- **Restart ast-mcp** after changing embedding settings.
- Confirm: `GET http://localhost:7821/health` and `GET /embed/health` (`embed_mode`, `embed_model`, `backend`).

See [README — Embedding backends](../../README.md#embedding-backends).

## Dashboard (port 7830)

- **Index health:** embed queue depth, active workers, pinned projects.
- **Settings → Pin:** priority embedding, no idle watcher stop, warmer vector cache unload.
- **Settings → Embedding backend:** persist env-equivalent keys (env wins).
- **Ignore globs:** skip high-churn paths that would otherwise index as code (applied after `IsCodeFile`).
- **Log indexing:** plain `.log` / `.txt` not indexed by default; enable for FTS-only (no embeddings).
- **Log retention:** optional deletion of `.log` only under configured absolute roots.

## Shell helper

After `make install`:

```bash
ast-mcp start|stop|restart|status|health|log|build|dash
```

## Rebuild after source changes

Use skill `ast-context-cache-rebuild` (`.cursor/skills/ast-rebuild/`) or `make build && ast-mcp restart`.

Canonical copy: [skills/operator/SKILL.md](../../skills/operator/SKILL.md)
