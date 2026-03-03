BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
ORT_LIB := $(BREW_PREFIX)/lib
ORT_INC := $(BREW_PREFIX)/include/onnxruntime
PROJ_DIR := $(shell pwd)

CGO_FLAGS := CGO_LDFLAGS="-L$(PROJ_DIR) -L$(ORT_LIB)" CGO_CFLAGS="-I$(ORT_INC)"

BINARY := ast-mcp
TOKENIZER_LIB := libtokenizers.a
# Pre-built libtokenizers: pick by GOOS-GOARCH (see https://github.com/daulet/tokenizers/releases)
GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
TOKENIZER_ARCH := $(GOOS)-$(GOARCH)
TOKENIZER_BASE := https://github.com/daulet/tokenizers/releases/latest/download
TOKENIZER_URL := $(TOKENIZER_BASE)/libtokenizers.$(TOKENIZER_ARCH).tar.gz

.PHONY: help build run download-model download-tokenizer-lib clean test

help:
	@echo "Targets:"
	@echo "  make download-model          # download ONNX model + tokenizer from HuggingFace"
	@echo "  make download-tokenizer-lib  # download pre-built libtokenizers.a"
	@echo "  make build                   # build the unified binary"
	@echo "  make run                     # build and run"
	@echo "  make clean                   # remove binary"

download-model:
	./download-model.sh

download-tokenizer-lib:
	@if [ ! -f $(TOKENIZER_LIB) ]; then \
		echo "Downloading libtokenizers for $(TOKENIZER_ARCH)..."; \
		curl -sfL -o libtokenizers.tar.gz $(TOKENIZER_URL) || (echo "Unsupported or missing build for $(TOKENIZER_ARCH). Download manually from $(TOKENIZER_BASE) and extract libtokenizers.a into project root."; exit 1); \
		tar xzf libtokenizers.tar.gz; \
		rm -f libtokenizers.tar.gz; \
	else \
		echo "libtokenizers.a already exists, skipping"; \
	fi

build: download-model download-tokenizer-lib
	$(CGO_FLAGS) go build -tags sqlite_fts5 -o $(BINARY) ./cmd/ast-mcp/

run: build
	ONNXRUNTIME_LIB=$(ORT_LIB)/libonnxruntime.dylib ./$(BINARY)

clean:
	rm -f $(BINARY)
