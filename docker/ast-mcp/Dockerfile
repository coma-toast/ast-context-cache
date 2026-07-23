# ast-mcp container image (prebuilt binary).
#
# CGO + ONNX Runtime make a full multi-stage build fragile across hosts.
# Prefer: build a Linux ast-mcp on the host/CI, copy it next to this Dockerfile.
#
#   make build                          # on Linux
#   cp ast-mcp docker/ast-mcp/ast-mcp
#   docker compose -f docker/ast-mcp/compose.yml up -d --build
#
# For embeddings without shipping ONNX into the image, set EMBED_BACKEND to
# ollama, http, openai, or docker (see README.md in this directory).

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Optional: put Linux libonnxruntime.so* in lib/ before build for EMBED_BACKEND=onnx.
COPY lib/ /usr/local/lib/
RUN ldconfig 2>/dev/null || true

COPY ast-mcp /usr/local/bin/ast-mcp
RUN chmod +x /usr/local/bin/ast-mcp

ENV HOME=/data \
    ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so \
    AST_MCP_TIER=extended

WORKDIR /data
VOLUME ["/data"]

EXPOSE 7821 7830

CMD ["ast-mcp"]
