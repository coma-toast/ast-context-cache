package tokens

import (
	"math"
	"unicode"
	"unicode/utf8"
)

// Character classes of the o200k_base pre-tokenizer pattern:
//
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// The scanner below reproduces tiktoken-go's backtracking regexp2 matches for
// this pattern without running a regex engine.
const (
	clsLetter = 1 << iota // \p{L}
	clsNumber             // \p{N}
	clsSpace              // \s (unicode.IsSpace in regexp2)
	clsUpper              // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
	clsLower              // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
)

var asciiClass = func() (t [utf8.RuneSelf]uint8) {
	for r := range t {
		t[r] = classifyRune(rune(r))
	}
	return t
}()

func classifyRune(r rune) uint8 {
	var c uint8
	switch {
	case unicode.Is(unicode.Lu, r), unicode.Is(unicode.Lt, r):
		c = clsLetter | clsUpper
	case unicode.Is(unicode.Ll, r):
		c = clsLetter | clsLower
	case unicode.Is(unicode.Lm, r), unicode.Is(unicode.Lo, r):
		c = clsLetter | clsUpper | clsLower
	case unicode.Is(unicode.M, r):
		c = clsUpper | clsLower
	case unicode.IsNumber(r):
		c = clsNumber
	}
	if unicode.IsSpace(r) {
		c |= clsSpace
	}
	return c
}

// classAt decodes the rune at byte i of s and returns its class and width.
func classAt(s string, i int) (rune, uint8, int) {
	if b := s[i]; b < utf8.RuneSelf {
		return rune(b), asciiClass[b], 1
	}
	r, w := utf8.DecodeRuneInString(s[i:])
	return r, classifyRune(r), w
}

// isOther reports [^\s\p{L}\p{N}].
func isOther(c uint8) bool {
	return c&(clsSpace|clsLetter|clsNumber) == 0
}

// countPieces returns the o200k token count of s, which must be valid UTF-8.
func countPieces(ranks map[string]int, s string) int {
	n := 0
	for i := 0; i < len(s); {
		e := pieceEnd(s, i)
		n += pieceTokens(ranks, s[i:e])
		i = e
	}
	return n
}

// pieceEnd returns the end of the pre-token that starts at byte i of s.
func pieceEnd(s string, i int) int {
	r, c, w := classAt(s, i)
	prefix := c&(clsLetter|clsNumber) == 0 && r != '\r' && r != '\n'
	if prefix {
		if e := upperLowerEnd(s, i+w); e >= 0 {
			return contractionEnd(s, e)
		}
	}
	if e := upperLowerEnd(s, i); e >= 0 {
		return contractionEnd(s, e)
	}
	if prefix {
		if e := upperRunEnd(s, i+w); e >= 0 {
			return contractionEnd(s, e)
		}
	}
	if e := upperRunEnd(s, i); e >= 0 {
		return contractionEnd(s, e)
	}
	if c&clsNumber != 0 {
		e := i + w
		for k := 1; k < 3 && e < len(s); k++ {
			_, c2, w2 := classAt(s, e)
			if c2&clsNumber == 0 {
				break
			}
			e += w2
		}
		return e
	}
	if start := otherStart(s, i, r, c, w); start >= 0 {
		return otherEnd(s, start)
	}
	return spaceEnd(s, i)
}

// upperLowerEnd matches [Upper]*[Lower]+ from j and returns its end or -1.
func upperLowerEnd(s string, j int) int {
	e, lastLower := j, -1
	for e < len(s) {
		_, c, w := classAt(s, e)
		if c&clsUpper == 0 {
			break
		}
		e += w
		if c&clsLower != 0 {
			lastLower = e
		}
	}
	k := e
	for k < len(s) {
		_, c, w := classAt(s, k)
		if c&clsLower == 0 {
			break
		}
		k += w
	}
	if k > e {
		return k
	}
	return lastLower // backtrack into the upper run to its last lower-class rune
}

// upperRunEnd matches [Upper]+[Lower]* from j and returns its end or -1.
func upperRunEnd(s string, j int) int {
	e := j
	for e < len(s) {
		_, c, w := classAt(s, e)
		if c&clsUpper == 0 {
			break
		}
		e += w
	}
	if e == j {
		return -1
	}
	for e < len(s) {
		_, c, w := classAt(s, e)
		if c&clsLower == 0 {
			break
		}
		e += w
	}
	return e
}

// contractionEnd extends e over an optional (?i:'s|'t|'re|'ve|'m|'ll|'d).
func contractionEnd(s string, e int) int {
	if e >= len(s) || s[e] != '\'' {
		return e
	}
	r1, w1 := utf8.DecodeRuneInString(s[e+1:])
	switch unicode.ToLower(r1) {
	case 's', 't', 'm', 'd':
		return e + 1 + w1
	}
	r2, w2 := utf8.DecodeRuneInString(s[e+1+w1:])
	switch l1, l2 := unicode.ToLower(r1), unicode.ToLower(r2); {
	case (l1 == 'r' || l1 == 'v') && l2 == 'e', l1 == 'l' && l2 == 'l':
		return e + 1 + w1 + w2
	}
	return e
}

// otherStart returns where [^\s\p{L}\p{N}]+ begins for " ?[^\s\p{L}\p{N}]+", or -1.
func otherStart(s string, i int, r rune, c uint8, w int) int {
	if r == ' ' && i+w < len(s) {
		if _, c2, _ := classAt(s, i+w); isOther(c2) {
			return i + w
		}
	}
	if isOther(c) {
		return i
	}
	return -1
}

// otherEnd matches [^\s\p{L}\p{N}]+[\r\n/]* from start.
func otherEnd(s string, start int) int {
	e := start
	for e < len(s) {
		_, c, w := classAt(s, e)
		if !isOther(c) {
			break
		}
		e += w
	}
	for e < len(s) && (s[e] == '\r' || s[e] == '\n' || s[e] == '/') {
		e++
	}
	return e
}

// spaceEnd matches \s*[\r\n]+|\s+(?!\S)|\s+ from i, where s[i] is whitespace.
func spaceEnd(s string, i int) int {
	e, lastNewline, lastStart, n := i, -1, i, 0
	for e < len(s) {
		r, c, w := classAt(s, e)
		if c&clsSpace == 0 {
			break
		}
		if r == '\r' || r == '\n' {
			lastNewline = e + w
		}
		lastStart = e
		e += w
		n++
	}
	switch {
	case lastNewline >= 0:
		return lastNewline
	case e == len(s), n == 1:
		return e
	default:
		return lastStart // leave the last space to prefix the next word
	}
}

type bpePart struct {
	start int
	rank  int
}

// pieceTokens counts the BPE tokens of one pre-token with tiktoken's merge order.
func pieceTokens(ranks map[string]int, piece string) int {
	if len(piece) <= 1 {
		return len(piece)
	}
	if _, ok := ranks[piece]; ok {
		return 1
	}
	var stack [64]bpePart
	parts := stack[:0]
	if len(piece)+1 > len(stack) {
		parts = make([]bpePart, 0, len(piece)+1)
	}
	for i := 0; i <= len(piece); i++ {
		parts = append(parts, bpePart{start: i, rank: math.MaxInt})
	}
	rankOf := func(i, skip int) int {
		if i+skip+2 < len(parts) {
			if r, ok := ranks[piece[parts[i].start:parts[i+skip+2].start]]; ok {
				return r
			}
		}
		return math.MaxInt
	}
	for i := 0; i < len(parts)-2; i++ {
		parts[i].rank = rankOf(i, 0)
	}
	for len(parts) > 1 {
		minRank, minIdx := math.MaxInt, -1
		for i := 0; i < len(parts)-1; i++ {
			if parts[i].rank < minRank {
				minRank, minIdx = parts[i].rank, i
			}
		}
		if minIdx < 0 {
			break
		}
		parts[minIdx].rank = rankOf(minIdx, 1)
		if minIdx > 0 {
			parts[minIdx-1].rank = rankOf(minIdx-1, 1)
		}
		parts = append(parts[:minIdx+1], parts[minIdx+2:]...)
	}
	return len(parts) - 1
}
