BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
ORT_LIB     := $(BREW_PREFIX)/lib
ORT_INC     := $(BREW_PREFIX)/include/onnxruntime
PROJ_DIR    := $(shell pwd)

CGO_FLAGS := CGO_LDFLAGS="-L$(PROJ_DIR) -L$(ORT_LIB)" CGO_CFLAGS="-I$(ORT_INC)"

BINARY       := ast-mcp
TOKENIZER_LIB := libtokenizers.a

GOOS   := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
TOKENIZER_ARCH := $(GOOS)-$(GOARCH)
TOKENIZER_BASE := https://github.com/daulet/tokenizers/releases/latest/download
TOKENIZER_URL  := $(TOKENIZER_BASE)/libtokenizers.$(TOKENIZER_ARCH).tar.gz

UNAME_S := $(shell uname -s)

ORT_DYLIB_DARWIN := $(ORT_LIB)/libonnxruntime.dylib
ORT_DYLIB_LINUX  := $(firstword $(wildcard /usr/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so $(ORT_LIB)/libonnxruntime.so))

ifeq ($(UNAME_S),Darwin)
  ORT_DYLIB := $(ORT_DYLIB_DARWIN)
else
  ORT_DYLIB := $(ORT_DYLIB_LINUX)
endif

.PHONY: help setup deps build run clean install uninstall

help:
	@echo "ast-context-cache"
	@echo ""
	@echo "  make setup    — install everything and build (start here)"
	@echo "  make build    — download deps + build binary"
	@echo "  make run      — build + run the server"
	@echo "  make install  — copy shell functions to your shell config"
	@echo "  make clean    — remove binary"
	@echo ""
	@echo "  make deps                  — install onnxruntime + download model + tokenizer lib"
	@echo "  make install-onnxruntime   — install onnxruntime via brew (macOS) or print instructions"
	@echo "  make download-model        — download ONNX model + tokenizer from HuggingFace"
	@echo "  make download-tokenizer-lib — download pre-built libtokenizers.a"

# ── One-command setup ──────────────────────────────────────────────

setup: deps build
	@echo ""
	@echo "Setup complete! Run with:"
	@echo "  make run"
	@echo ""
	@echo "Or install the shell function for easier management:"
	@echo "  make install"

# ── Dependencies ───────────────────────────────────────────────────

deps: install-onnxruntime download-model download-tokenizer-lib

install-onnxruntime:
ifeq ($(UNAME_S),Darwin)
	@if [ -f "$(ORT_DYLIB_DARWIN)" ]; then \
		echo "onnxruntime: already installed"; \
	else \
		echo "Installing onnxruntime via brew..."; \
		brew install onnxruntime; \
	fi
else
	@if [ -n "$(ORT_DYLIB_LINUX)" ]; then \
		echo "onnxruntime: found at $(ORT_DYLIB_LINUX)"; \
	else \
		echo "onnxruntime: NOT FOUND"; \
		echo "  Install from https://github.com/microsoft/onnxruntime/releases"; \
		echo "  or: apt-get install libonnxruntime-dev"; \
		exit 1; \
	fi
endif

download-model:
	@mkdir -p model
	@if [ -f model/model.onnx ] && [ -f model/tokenizer.json ]; then \
		echo "model files: already exist"; \
	else \
		echo "Downloading ONNX model (all-mpnet-base-v2)..."; \
		[ -f model/model.onnx ] || curl -L --progress-bar -o model/model.onnx \
			"https://huggingface.co/onnx-models/all-mpnet-base-v2-onnx/resolve/main/model.onnx"; \
		[ -f model/tokenizer.json ] || curl -L --progress-bar -o model/tokenizer.json \
			"https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json"; \
		echo "model files: downloaded"; \
	fi

download-tokenizer-lib:
	@if [ -f $(TOKENIZER_LIB) ]; then \
		echo "libtokenizers.a: already exists"; \
	else \
		echo "Downloading libtokenizers for $(TOKENIZER_ARCH)..."; \
		curl -sfL -o libtokenizers.tar.gz $(TOKENIZER_URL) \
			|| (echo "No pre-built libtokenizers for $(TOKENIZER_ARCH)."; \
			    echo "Download manually from $(TOKENIZER_BASE)"; exit 1); \
		tar xzf libtokenizers.tar.gz; \
		rm -f libtokenizers.tar.gz; \
		echo "libtokenizers.a: downloaded"; \
	fi

# ── Build & Run ────────────────────────────────────────────────────

build: download-model download-tokenizer-lib
	@echo "Building ast-mcp..."
	$(CGO_FLAGS) go build -tags sqlite_fts5 -o $(BINARY) ./cmd/ast-mcp/
	@echo "Built: ./$(BINARY)"

run: build
	ONNXRUNTIME_LIB=$(ORT_DYLIB) ./$(BINARY)

clean:
	rm -f $(BINARY)

# ── Shell function install ─────────────────────────────────────────

install:
	@echo "Installing shell functions..."
	@AST_DIR="$(PROJ_DIR)"; \
	ORT="$(ORT_DYLIB)"; \
	if [ -d "$$HOME/.config/fish/functions" ]; then \
		sed -e "s|__AST_DIR__|$$AST_DIR|g" -e "s|__ORT_LIB__|$$ORT|g" \
			scripts/ast-mcp.fish > "$$HOME/.config/fish/functions/ast-mcp.fish"; \
		echo "  fish: installed to ~/.config/fish/functions/ast-mcp.fish"; \
	else \
		echo "  fish: skipped (no ~/.config/fish/functions/)"; \
	fi; \
	BASH_RC="$$HOME/.bashrc"; \
	if [ -f "$$BASH_RC" ]; then \
		MARKER="# ast-mcp-function"; \
		if grep -q "$$MARKER" "$$BASH_RC" 2>/dev/null; then \
			echo "  bash: already in ~/.bashrc"; \
		else \
			echo "" >> "$$BASH_RC"; \
			echo "$$MARKER" >> "$$BASH_RC"; \
			sed -e "s|__AST_DIR__|$$AST_DIR|g" -e "s|__ORT_LIB__|$$ORT|g" \
				scripts/ast-mcp.bash >> "$$BASH_RC"; \
			echo "  bash: appended to ~/.bashrc (source it or open a new terminal)"; \
		fi; \
	fi; \
	ZSH_RC="$$HOME/.zshrc"; \
	if [ -f "$$ZSH_RC" ]; then \
		MARKER="# ast-mcp-function"; \
		if grep -q "$$MARKER" "$$ZSH_RC" 2>/dev/null; then \
			echo "  zsh: already in ~/.zshrc"; \
		else \
			echo "" >> "$$ZSH_RC"; \
			echo "$$MARKER" >> "$$ZSH_RC"; \
			sed -e "s|__AST_DIR__|$$AST_DIR|g" -e "s|__ORT_LIB__|$$ORT|g" \
				scripts/ast-mcp.bash >> "$$ZSH_RC"; \
			echo "  zsh: appended to ~/.zshrc (source it or open a new terminal)"; \
		fi; \
	fi
	@echo "Done. Usage: ast-mcp start|stop|restart|status|health"

uninstall:
	@echo "Removing shell functions..."
	@rm -f "$$HOME/.config/fish/functions/ast-mcp.fish" 2>/dev/null && \
		echo "  fish: removed" || echo "  fish: not found"
	@for rc in "$$HOME/.bashrc" "$$HOME/.zshrc"; do \
		if [ -f "$$rc" ] && grep -q "# ast-mcp-function" "$$rc" 2>/dev/null; then \
			sed -i.bak '/# ast-mcp-function/,/^$$/d' "$$rc"; \
			rm -f "$$rc.bak"; \
			echo "  removed from $$rc"; \
		fi; \
	done
	@echo "Done."
