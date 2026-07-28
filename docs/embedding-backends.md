# Embedding backends

The vector store is built for **768-dimensional** L2-normalized embeddings. The default is local **ONNX** (no extra services). Alternatives use environment variables and do not require downloading `model.onnx` for the main process (unless you switch back to `onnx`).

## Backend table

| `EMBED_BACKEND` | When to use | Main env vars |
|-----------------|------------|---------------|
| `onnx` (default) | Full local path: `make setup` pulls HuggingFace ONNX + tokenizer | `MODEL_DIR` to override model directory |
| `ollama` | Local or Docker [Ollama](https://ollama.com) with a **768-d** model; default `nomic-embed-text` | `OLLAMA_HOST` (e.g. `http://127.0.0.1:11434`), `OLLAMA_EMBED_MODEL` |
| `http` | Any service that matches the built-in JSON: `POST` body `{"texts":["..."]}` → `{"embeddings":[[float32,...]]}` (same as `http://localhost:7821/embed` on the ONNX server) | `EMBED_HTTP_URL` (default `http://127.0.0.1:8080/embed`), `EMBED_HTTP_BEARER` |
| `openai` (alias: `litellm`) | [LiteLLM](https://docs.litellm.ai/docs/), OpenAI, or any **OpenAI-compatible** `POST /v1/embeddings` gateway; vectors must be **768-d** (native model or `dimensions` in JSON) | `EMBED_OPENAI_BASE_URL` (default `https://api.openai.com/v1`), **`EMBED_OPENAI_MODEL`** (required), `EMBED_OPENAI_API_KEY`, `EMBED_OPENAI_DIMENSIONS` (optional: unset sends `768` for v3 shortening; `0` omits the field) |
| `docker` | [Docker Model Runner](https://docs.docker.com/ai/model-runner/) embeddings (port **12434**); no local ONNX in ast-mcp | `EMBED_DOCKER_URL` (default `http://127.0.0.1:12434`), `EMBED_DOCKER_MODEL` (default `ai/qwen3-embedding`), `EMBED_DOCKER_DIMENSIONS` (default `768`) |

## Docker Model Runner quick start

See [`docker/README.md`](../docker/README.md). Quick start:

```bash
docker desktop enable model-runner --tcp 12434   # if needed
docker model pull ai/qwen3-embedding
EMBED_BACKEND=docker make run
```

Re-index projects after switching embed backends.

## Process environment vs Dashboard Settings

**Process environment:** Whatever starts `ast-mcp` (foreground terminal, the `ast-mcp` shell function from `make install`, systemd, or another supervisor) must have the embedding variables from the table above exported for non-default backends—for example set `EMBED_BACKEND=docker` and `EMBED_DOCKER_PROVIDER=ollama`, or `EMBED_BACKEND=openai`, `EMBED_OPENAI_BASE_URL`, `EMBED_OPENAI_API_KEY`, and `EMBED_OPENAI_MODEL`, in the same environment as the process that execs `./ast-mcp`.

**Dashboard (easier):** On **Settings** (port 7830), use **Embedding backend** to save the same keys into local SQLite (`~/.astcache/usage.db`). **Non-empty environment variables always override** the saved values. **Restart ast-mcp** after changing embedding settings so `NewForMain` runs again.

## Health endpoints

`GET /health` and `GET /embed/health` include `embed_mode`, `embed_model`, and `backend` so you can confirm which path is active.

## See also

- [Quick Start](../README.md#quick-start) — installing and running ast-mcp
- [`skills/operator/SKILL.md`](../skills/operator/SKILL.md) — operator runbooks (dashboard, WAL, keep-alive, failure modes)
