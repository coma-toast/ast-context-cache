# PRD: Gigatoken Integration for ast-context-cache Tokenization

## Title
Gigatoken Compatibility Mode Integration for ast-context-cache

## Date
2026-08-17

## Status
Draft

## Problem Statement
The ast-context-cache system currently uses the `github.com/daulet/tokenizers` Go library with the `all-mpnet-base-v2` tokenizer for code search and semantic indexing. Tokenization performance is a bottleneck in the code indexing pipeline. Integration of Gigatoken's compatibility mode would provide significant tokenization speedup (200-300x) while preserving exact output parity with the existing HuggingFace tokenizer, ensuring embedding vectors remain unchanged.

## Goals
1. Integrate Gigatoken in compatibility mode with the existing ast-context-cache tokenizer
2. Achieve 200-300x tokenization speedup over current `github.com/daulet/tokenizers` implementation
3. Preserve exact embedding output parity - identical embeddings before and after integration
4. Support all target platforms: macOS, Linux, Windows via WSL
5. Minimize maintenance burden while maintaining code quality

## Non-Goals (Out of Scope)
1. Achieving the full 989x speedup from Gigatoken's native API (compatibility mode targets 200-300x)
2. Supporting tokenizer formats beyond the existing `all-mpnet-base-v2` format
3. Replacing the ONNX embedding model (`all-mpnet-base-v2`) - only the tokenizer step is being optimized
4. Supporting Windows natively (WSL compatibility is acceptable)
5. Integrating Gigatoken for other system components outside of tokenization

## Functional Requirements

### FR-1: Gigatoken Compatibility Wrapper
- **MUST**: Implement a Go wrapper that integrates Gigatoken's compatibility mode with the existing `github.com/daulet/tokenizers` interface
- **MUST**: The wrapper must accept the existing `model/tokenizer.json` format and produce identical token IDs to the current implementation
- **MUST**: Provide a drop-in replacement for the current tokenizer loading: `tokenizer := gt.Tokenizer(hf_tokenizer).as_hf()`
- **SHOULD**: Include benchmark metrics comparing Gigatoken performance vs current implementation on target platforms
- **MAY**: Support incremental adoption with a config flag to toggle between current and Gigatoken implementations

### FR-2: Output Parity Verification
- **MUST**: Implement automated tests that validate embedding output parity between current tokenizer and Gigatoken compatibility mode
- **MUST**: Tests must run on at least macOS and Linux platforms
- **SHOULD**: Include a regression test suite that can be run as part of CI/CD
- **MAY**: Provide a side-by-side comparison tool for debugging any output differences

### FR-3: Platform Support
- **MUST**: Ensure compatibility on macOS (Apple Silicon and Intel)
- **MUST**: Ensure compatibility on Linux (x86_64 and ARM64)
- **MAY**: Support Windows via WSL (not a blocker if not fully tested)
- **SHOULD**: Test on the same hardware configurations used by the existing ast-context-cache deployment

### FR-4: Error Handling & Fallback
- **MUST**: Graceful fallback to current tokenizer if Gigatoken compatibility mode fails or produces incompatible output
- **MUST**: Clear error messages when Gigatoken integration cannot be initialized
- **SHOULD**: Logging of performance metrics when Gigatoken is active
- **MAY**: Configurable timeout for Gigatoken initialization

### FR-5: Maintenance & Observability
- **SHOULD**: Include documentation for maintaining the Gigatoken integration
- **SHOULD**: Add metrics for tokenization throughput (tokens/sec) and error rates
- **MAY**: Provide a simple benchmark command to measure performance improvements

## Non-Functional Requirements

### NFR-1: Performance
- Target: 200-300x speedup in tokenization throughput compared to `github.com/daulet/tokenizers`
- Measure: Tokens per second on representative code corpora
- Acceptable: Any speedup > 10x is considered successful; full 200-300x is ideal

### NFR-2: Output Parity
- Embedding vectors must be bit-identical between current implementation and Gigatoken compatibility mode
- Verified through automated test suite running on representative code samples

### NFR-3: Backward Compatibility
- No breaking changes to existing APIs or interfaces
- Existing code using the tokenizer should work without modification

### NFR-4: Deployment Simplicity
- Integration should not require changes to deployment scripts or infrastructure
- New dependencies should be minimal and well-documented

## User Workflows

### HW-1: Standard Tokenization (Happy Path)
1. System starts ast-context-cache service
2. Tokenizer loads via Gigatoken compatibility wrapper
3. Code indexing pipeline tokenizes source files
4. Embeddings generated from tokens remain identical to pre-integration behavior
5. Code search and semantic indexing function normally

### HW-2: Integration Verification
1. Run integration test suite
2. Compare embedding outputs from current tokenizer vs Gigatoken compatibility mode
3. Verify throughput metrics meet target thresholds
4. If parity fails, system automatically falls back to current tokenizer

### HW-3: Platform-Specific Operation
1. On macOS: Tokenizer loads and operates without issues
2. On Linux: Tokenizer loads and operates without issues
3. On Windows (WSL): Tokenizer operates (may have limited testing)

## Integration Points

### Int-1: ast-context-cache Tokenizer Layer
- **Replaces**: `github.com/daulet/tokenizers` loading and usage
- **Input**: Existing `model/tokenizer.json` (HuggingFace format)
- **Output**: Token IDs compatible with existing ONNX model input
- **Interface**: Same functions/signatures as current tokenizer

### Int-2: ONNX Embedding Model (`all-mpnet-base-v2`)
- **Unchanged**: Receives same token IDs as before
- **Output**: Identical 768-dimension embedding vectors
- **Dependency**: Only the tokenizer step changes; embedding model remains the same

### Int-3: ast-context-cache Indexing Pipeline
- **Unchanged**: Rest of indexing pipeline is unaffected
- **Data flow**: Text → Tokenizer → Token IDs → ONNX Model → Embeddings → Milvus/ast-cache

### Int-4: Configuration (if applicable)
- **Optional**: Feature flag or config option to toggle between tokenizer implementations
- **Default**: Current implementation active; Gigatoken can be enabled via configuration

## Acceptance Criteria

### AC-1: Performance Benchmark
- Tokenization throughput measured at > 10x improvement over current implementation on at least one target platform
- Full 200-300x target met on representative code corpora

### AC-2: Output Parity
- Embedding vectors from Gigatoken compatibility mode are bit-identical to current implementation
- Tested on minimum 100 code samples spanning different languages and sizes
- Zero mismatches in embedding output across all test samples

### AC-3: Platform Compatibility
- Integration successfully builds and runs on macOS (tested on Apple Silicon)
- Integration successfully builds and runs on Linux (tested on x86_64)
- No critical bugs preventing operation on target platforms

### AC-4: Fallback Reliability
- System gracefully falls back to current tokenizer when Gigatoken compatibility mode fails
- No data loss or service interruption during fallback
- Clear error messages logged when fallback occurs

### AC-5: Test Coverage
- Minimum 90% test coverage for Gigatoken integration code
- Integration tests run in CI on every pull request
- Performance benchmarks included in CI pipeline

## Open Questions

1. **Tokenizer.json format**: The existing `model/tokenizer.json` uses HuggingFace format - need to verify Gigatoken compatibility mode can load this directly or requires mapping/translation.

2. **Cgo vs pure Go**: Gigatoken provides Rust bindings - best approach is Go wrapper via Cgo, or is there a pure Go alternative that achieves similar speedup?

3. **Benchmark corpus**: What representative code corpus should be used for performance benchmarking? Should use existing ast-context-cache indexing data.

4. **Test data**: What code samples should be used for output parity verification? Should cover multiple languages, file sizes, and complexity levels.

5. **Feature flag necessity**: Given "full replacement" rollout strategy was chosen, is a feature flag still needed for emergency fallback, or is the graceful fallback sufficient?

## Decision-Made Refinements

- **Integration mode**: Compatibility mode (Option A) selected over native API (Option B) to preserve exact embedding output parity
- **Rollout strategy**: Full replacement rather than gradual rollout, with graceful fallback to current tokenizer on failure
- **Platform priority**: macOS and Linux first-class; Windows via WSL as "may support"
- **Performance target**: 200-300x speedup from compatibility mode (vs 989x from native API, which was rejected due to output parity concerns)
- **Maintenance**: Team willing to maintain integration code; documentation included as part of deliverable

---

**Related**: This PRD supports the Gigatoken integration initiative for configSync's ast-context-cache code search system.

**Status**: Draft - pending user review and approval before implementation planning.