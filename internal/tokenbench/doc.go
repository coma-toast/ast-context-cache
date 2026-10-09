// Package tokenbench measures the tokens, recall, and determinism of the MCP search tools
// against a small fixed fixture repo (testdata/fixture) and the scenarios in
// scenarios.yaml. The benchmark is TestTokenBench; baseline.json holds the last accepted
// numbers. Run it with make bench-tokens and rewrite the baseline with make
// bench-tokens-update.
package tokenbench
