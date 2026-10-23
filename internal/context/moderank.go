package context

// Mode ranks order how much of a symbol a mode delivers, for mode-aware dedup:
// a symbol already returned at rank r satisfies any later request of rank <= r.
const (
	rankLocations = 0
	rankSummary   = 1
	rankSkeleton  = 2
	rankFull      = 3
	// autoFullCount is how many top-ranked hits auto mode returns in full.
	autoFullCount = 3
)

// ModeRank ranks a delivered mode: locations 0 < summary 1 < skeleton 2 < full = edit 3.
// Unknown modes (including auto, which is never delivered) rank as skeleton.
func ModeRank(mode string) int {
	switch mode {
	case "locations":
		return rankLocations
	case "summary":
		return rankSummary
	case "full", "edit":
		return rankFull
	default:
		return rankSkeleton
	}
}

// EffectiveModeV2 resolves auto by result rank (0-based): full for the top three,
// skeleton for the rest. It never returns summary for auto; other modes pass through.
func EffectiveModeV2(mode string, rank int) string {
	if mode != "auto" {
		return mode
	}
	if rank < autoFullCount {
		return "full"
	}
	return "skeleton"
}
