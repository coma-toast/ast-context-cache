package search

// RRFK is the reciprocal rank fusion constant: larger values flatten the gap between ranks.
const RRFK = 60

// RRF fuses ranked id lists (best first) with reciprocal rank fusion: each id scores the sum
// of 1/(RRFK+rank+1) over the lists it appears in. Empty ids and repeats of an id within one
// list are skipped, so an id counts at most once per list, at its best rank.
func RRF(lists ...[]string) map[string]float64 {
	scores := map[string]float64{}
	for _, list := range lists {
		seen := make(map[string]bool, len(list))
		for rank, id := range list {
			if id == "" || seen[id] {
				continue
			}
			seen[id] = true
			scores[id] += 1.0 / float64(RRFK+rank+1)
		}
	}
	return scores
}
