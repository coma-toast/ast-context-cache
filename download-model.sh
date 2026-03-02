#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model"

mkdir -p "$MODEL_DIR"

echo "Downloading all-mpnet-base-v2 ONNX model..."

if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    curl -L --progress-bar -o "$MODEL_DIR/model.onnx" \
        "https://huggingface.co/onnx-models/all-mpnet-base-v2-onnx/resolve/main/model.onnx"
    echo "  model.onnx downloaded"
else
    echo "  model.onnx already exists, skipping"
fi

if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    curl -L --progress-bar -o "$MODEL_DIR/tokenizer.json" \
        "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json"
    echo "  tokenizer.json downloaded"
else
    echo "  tokenizer.json already exists, skipping"
fi

echo "Done. Model files in $MODEL_DIR"
