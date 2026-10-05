//go:build race

package handoff

// raceEnabled reports whether the test binary was built with -race, which slows every
// operation several-fold; latency budgets are skipped under it.
const raceEnabled = true
