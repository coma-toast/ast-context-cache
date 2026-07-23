# Docker Model Runner (embeddings)

> For running **ast-mcp itself** in Docker (`restart: unless-stopped`), see [`ast-mcp/README.md`](ast-mcp/README.md). This file is only about the DMR embedding backend.

The `EMBED_BACKEND=docker` path uses [Docker Model Runner](https://docs.docker.com/ai/model-runner/) (DMR), not Ollama or TEI containers.

## Setup

1. Enable **Docker Model Runner** in Docker Desktop (Settings → AI) or Docker Engine.
2. Enable host TCP on port **12434** (Docker Desktop CLI example):

   ```bash
   docker desktop enable model-runner --tcp 12434
   ```

3. Pull an embedding model (768-d output required for the ast-context-cache index):

   ```bash
   docker model pull ai/qwen3-embedding
   ```

4. Start ast-mcp with the docker backend:

   ```bash
   export EMBED_BACKEND=docker
   # optional overrides:
   # export EMBED_DOCKER_URL=http://127.0.0.1:12434
   # export EMBED_DOCKER_MODEL=ai/qwen3-embedding
   # export EMBED_DOCKER_DIMENSIONS=768
   make run
   ```

   Or set the same keys on the dashboard **Settings → Embedding backend** and restart `ast-mcp`.

## API

DMR exposes OpenAI-compatible embeddings at:

`http://localhost:12434/engines/v1/embeddings`

ast-mcp normalizes `EMBED_DOCKER_URL` to include `/engines/v1` when omitted.

## Re-index

Switching embedding backends or models changes the vector space. **Re-index** indexed projects after changing embed settings.

## Docs

- [DMR API reference](https://docs.docker.com/ai/model-runner/api-reference/)
- [ai/qwen3-embedding on Docker Hub](https://hub.docker.com/r/ai/qwen3-embedding)
