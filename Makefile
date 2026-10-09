# Intel (/usr/local) and Apple Silicon (/opt/homebrew) Homebrews can coexist, and the first brew on
# PATH may be the one without onnxruntime, so prefer whichever prefix actually has the library.
BREW_DEFAULT := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
BREW_PREFIX  := $(or $(firstword $(patsubst %/lib/libonnxruntime.dylib,%,$(wildcard $(BREW_DEFAULT)/lib/libonnxruntime.dylib /opt/homebrew/lib/libonnxruntime.dylib /usr/local/lib/libonnxruntime.dylib))),$(BREW_DEFAULT))
ORT_LIB     := $(BREW_PREFIX)/lib
ORT_INC     := $(BREW_PREFIX)/include/onnxruntime
PROJ_DIR    := $(shell pwd)

# GOWORK=off: this repo is not part of any go.work workspace, but Go's workspace
# auto-detection walks up from cwd — a go.work file in a parent directory (e.g. from an
# unrelated multi-module checkout one level up) would otherwise break module resolution.
CGO_FLAGS := GOWORK=off CGO_LDFLAGS="-L$(PROJ_DIR) -L$(ORT_LIB)" CGO_CFLAGS="-I$(ORT_INC)"

VERSION     := $(shell tr -d '[:space:]' < VERSION 2>/dev/null || echo dev)
LDFLAGS     := -X 'github.com/coma-toast/ast-context-cache/internal/version.Version=$(VERSION)'
BINARY       := ast-mcp
TOKENIZER_LIB := libtokenizers.a

GOOS   := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
TOKENIZER_ARCH := $(GOOS)-$(GOARCH)
# daulet/tokenizers release tarballs use x86_64, not amd64, on macOS.
TOKENIZER_RELEASE_ARCH := $(TOKENIZER_ARCH)
ifeq ($(TOKENIZER_ARCH),darwin-amd64)
  TOKENIZER_RELEASE_ARCH := darwin-x86_64
endif
TOKENIZER_BASE := https://github.com/daulet/tokenizers/releases/latest/download
TOKENIZER_URL  := $(TOKENIZER_BASE)/libtokenizers.$(TOKENIZER_RELEASE_ARCH).tar.gz

UNAME_S := $(shell uname -s)

ORT_DYLIB_DARWIN := $(ORT_LIB)/libonnxruntime.dylib
ORT_DYLIB_LINUX  := $(firstword $(wildcard /usr/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so $(ORT_LIB)/libonnxruntime.so))

ifeq ($(UNAME_S),Darwin)
  ORT_DYLIB := $(ORT_DYLIB_DARWIN)
else
  ORT_DYLIB := $(ORT_DYLIB_LINUX)
endif

.PHONY: help setup deps generate build run clean install uninstall test race bench bench-tokens bench-tokens-update fmt lint ui-test storybook build-storybook dashboard-screenshot verify-stories ui-build ui-dev

ui-build:
	cd ui && npm ci && npm run build

ui-dev:
	cd ui && npm run dev

ui-test:
	cd ui && npm test

help:
	@echo "ast-context-cache"
	@echo ""
	@echo "  make setup    — install everything and build (start here)"
	@echo "  make build    — download deps + build binary"
	@echo "  make run      — build + run the server"
	@echo "  make test     — run unit tests"
	@echo "  make race     — run unit tests with -race"
	@echo "  make bench    — run benchmarks (BENCH_PKGS, default ./internal/...)"
	@echo "  make bench-tokens        — token benchmark table vs internal/tokenbench/baseline.json"
	@echo "  make bench-tokens-update — rerun the token benchmark and rewrite its baseline.json"
	@echo "  make fmt      — format Go code with gofumpt"
	@echo "  make lint     — go vet + gofumpt check"
	@echo "  make ui-test  — run ui/ Vitest suite"
	@echo "  make install  — copy shell functions to your shell config"
	@echo "  make clean    — remove binary"
	@echo ""
	@echo "  make generate              — no-op (legacy templ generate removed; React UI is source of truth)"
	@echo "  make deps                  — install onnxruntime + download model + tokenizer lib"
	@echo "  make install-onnxruntime   — install onnxruntime via brew (macOS) or print instructions"
	@echo "  make download-model        — download ONNX model + tokenizer from HuggingFace"
	@echo "  make download-tokenizer-lib — download pre-built libtokenizers.a"
	@echo ""
	@echo "  make storybook             — run ui/ Storybook dev server (dashboard component previews)"
	@echo "  make build-storybook       — build ui/ Storybook to docs/storybook-static (gitignored)"
	@echo "  make dashboard-screenshot  — build Storybook + capture docs/images/*.png"
	@echo "  make verify-stories        — build Storybook + Playwright smoke on key stories"

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

# Native arch inside libtokenizers.a (lipo/file) vs go env GOARCH.
tokenizer_native_arch = $(if $(filter amd64,$(GOARCH)),x86_64,$(GOARCH))
TOKENIZER_STAMP := .libtokenizers.$(TOKENIZER_ARCH).stamp

download-tokenizer-lib:
	@need=1; \
	if [ -f $(TOKENIZER_LIB) ] && [ -f $(TOKENIZER_STAMP) ]; then \
		need=0; \
		if [ "$(UNAME_S)" = Darwin ]; then \
			have=$$(lipo -info $(TOKENIZER_LIB) 2>/dev/null | sed -E 's/.*architecture: //'); \
			if [ -n "$$have" ] && [ "$$have" != "$(tokenizer_native_arch)" ]; then \
				echo "libtokenizers.a: wrong arch ($$have, want $(tokenizer_native_arch)) — re-downloading"; \
				need=1; \
			fi; \
		else \
			have=$$(file -b $(TOKENIZER_LIB)); \
			case "$(GOARCH)" in \
				amd64) echo "$$have" | grep -qE 'x86-64|x86_64' || need=1 ;; \
				arm64) echo "$$have" | grep -qE 'aarch64|ARM' || need=1 ;; \
			esac; \
			if [ "$$need" = 1 ]; then echo "libtokenizers.a: wrong arch for $(TOKENIZER_ARCH) — re-downloading"; fi; \
		fi; \
	fi; \
	if [ "$$need" = 0 ]; then \
		echo "libtokenizers.a: already exists ($(TOKENIZER_ARCH))"; \
	else \
		rm -f $(TOKENIZER_LIB) $(TOKENIZER_STAMP) .libtokenizers.*.stamp libtokenizers.tar.gz; \
		echo "Downloading libtokenizers for $(TOKENIZER_ARCH) ($(TOKENIZER_RELEASE_ARCH))..."; \
		if ! curl -sfL -o libtokenizers.tar.gz $(TOKENIZER_URL); then \
			echo "No pre-built libtokenizers for $(TOKENIZER_ARCH)."; \
			echo "Download manually from $(TOKENIZER_BASE)"; \
			exit 1; \
		fi; \
		if ! tar xzf libtokenizers.tar.gz; then \
			rm -f libtokenizers.tar.gz; \
			echo "Failed to extract libtokenizers for $(TOKENIZER_ARCH)."; \
			exit 1; \
		fi; \
		rm -f libtokenizers.tar.gz; \
		if [ ! -f $(TOKENIZER_LIB) ]; then \
			echo "Download did not produce $(TOKENIZER_LIB)."; \
			exit 1; \
		fi; \
		touch $(TOKENIZER_STAMP); \
		echo "libtokenizers.a: downloaded"; \
	fi

# ── Build & Run ────────────────────────────────────────────────────

# Legacy templ generate removed; dashboard UI is React (`ui/`). Keep target for scripts/docs that still call it.
generate:
	@echo "generate: no-op (no .templ sources)"

internal/version/VERSION: VERSION
	cp VERSION internal/version/VERSION

build: download-model download-tokenizer-lib ui-build internal/version/VERSION
	@echo "Building ast-mcp..."
	$(CGO_FLAGS) go build -tags sqlite_fts5 -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/ast-mcp/
	@echo "$(ORT_DYLIB)" > $(BINARY).ortlib
	@echo "Built: ./$(BINARY)"

run: build
	ONNXRUNTIME_LIB=$(ORT_DYLIB) ./$(BINARY)

run-safe: build
	AST_EMBED_WORKERS=0 ONNXRUNTIME_LIB=$(ORT_DYLIB) ./$(BINARY)

# TEST_PKGS narrows the run, e.g. make test TEST_PKGS=./internal/errs/...
# TEST_FLAGS adds go test flags, e.g. make test TEST_FLAGS=-v
TEST_PKGS ?= ./...
TEST_FLAGS ?=

test: download-tokenizer-lib internal/version/VERSION
	$(CGO_FLAGS) CGO_ENABLED=1 go test -tags sqlite_fts5 -count=1 $(TEST_FLAGS) $(TEST_PKGS)

# bench-tokens prints the token benchmark table (tokens, results, recall, determinism)
# against internal/tokenbench/baseline.json; bench-tokens-update rewrites that baseline.
bench-tokens:
	AST_TOKENBENCH_VERBOSE=1 $(MAKE) test TEST_PKGS=./internal/tokenbench/... TEST_FLAGS="-v -run TestTokenBench"

bench-tokens-update:
	AST_TOKENBENCH_VERBOSE=1 AST_TOKENBENCH_UPDATE=1 $(MAKE) test TEST_PKGS=./internal/tokenbench/... TEST_FLAGS="-v -run TestTokenBench"

race: download-tokenizer-lib internal/version/VERSION
	$(CGO_FLAGS) CGO_ENABLED=1 go test -tags sqlite_fts5 -count=1 -race $(TEST_PKGS)

# BENCH_PKGS narrows the run, e.g. make bench BENCH_PKGS=./internal/errs/...
BENCH_PKGS ?= ./internal/...

bench: download-tokenizer-lib internal/version/VERSION
	$(CGO_FLAGS) CGO_ENABLED=1 go test -tags sqlite_fts5 -run '^$$' -bench . -benchmem $(BENCH_PKGS)

GOFUMPT := GOWORK=off go run mvdan.cc/gofumpt@v0.7.0

fmt:
	$(GOFUMPT) -w ./cmd ./internal

lint: download-tokenizer-lib internal/version/VERSION
	$(CGO_FLAGS) CGO_ENABLED=1 go vet -tags sqlite_fts5 ./...
	@out=$$($(GOFUMPT) -l ./cmd ./internal) || exit 1; \
	if [ -n "$$out" ]; then \
		echo "gofumpt: these files need formatting (run make fmt):"; \
		echo "$$out"; \
		exit 1; \
	fi

storybook:
	cd ui && npm ci && npm run storybook

build-storybook:
	cd ui && npm ci && npm run build-storybook

dashboard-screenshot: build-storybook
	cd ui && npx playwright install chromium && npm run capture-screenshot

verify-stories: build-storybook
	cd ui && npx playwright install chromium && npm run verify-stories

clean:
	rm -f $(BINARY)

clean-tokenizer-lib:
	rm -f $(TOKENIZER_LIB) libtokenizers.tar.gz .libtokenizers.*.stamp

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
