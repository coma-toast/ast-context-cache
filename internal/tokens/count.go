// Package tokens counts LLM tokens with the o200k_base BPE vocabulary, served
// offline from an embedded copy of o200k_base.tiktoken.
package tokens

import (
	"container/list"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/cespare/xxhash/v2"

	"github.com/coma-toast/ast-context-cache/internal/logging"
)

const (
	encodingName = "o200k_base"
	vocabURL     = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
	methodBPE    = "o200k"
	methodBytes  = "bytes4"
	memoMinBytes = 512
	memoCapacity = 4096
	// approxBytesPerToken is bytes per o200k token over this repo's .go files (3.644).
	approxBytesPerToken = 3.64
	// A Truncate cut moves back at most 1/boundarySlack of the prefix to land on whitespace.
	boundarySlack = 4
)

var logger = logging.Tagged("tokens")

// loadRanks loads the o200k mergeable ranks. Tests swap it to exercise the fallback.
var loadRanks = func() (map[string]int, error) {
	return embeddedLoader{}.LoadTiktokenBpe(vocabURL)
}

var (
	initOnce sync.Once
	ranks    map[string]int // nil when the vocabulary failed to load
	memo     = newLRU(memoCapacity)
)

func ranksInstance() map[string]int {
	initOnce.Do(func() {
		r, err := loadRanks()
		if err != nil {
			logger.Warn("TOKENIZER UNAVAILABLE, FALLING BACK TO BYTES/4", "error", err)
			return
		}
		ranks = r
	})
	return ranks
}

// Count returns the o200k token count of text, or len(text)/4 when the
// tokenizer cannot be loaded. Special-token text is counted as ordinary text.
func Count(text string) int {
	r := ranksInstance()
	if r == nil {
		return len(text) / 4
	}
	if len(text) < memoMinBytes {
		return count(r, text)
	}
	key := xxhash.Sum64String(text)
	if n, ok := memo.get(key); ok {
		return n
	}
	n := count(r, text)
	memo.put(key, n)
	return n
}

// count tokenizes like tiktoken-go's EncodeOrdinary, which reads invalid UTF-8
// bytes as U+FFFD.
func count(r map[string]int, text string) int {
	if !utf8.ValidString(text) {
		text = string([]rune(text))
	}
	return countPieces(r, text)
}

// Approx estimates the token count of source code from its length, calibrated
// against Count. It is for hot paths where an exact count isn't needed.
func Approx(text string) int {
	return int(float64(len(text))/approxBytesPerToken + 0.5)
}

// Truncate returns a prefix of text whose Count is at most maxTokens, cut on a
// rune boundary and, when one is near, on whitespace. The cut is found by binary
// search over byte offsets, so it costs O(log n) counts.
func Truncate(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}
	n := countFunc()
	if n(text) <= maxTokens {
		return text
	}
	lo, hi := 0, len(text) // n(text[:lo]) <= maxTokens < n(text[:hi])
	for {
		mid := (lo + hi) / 2
		for mid > lo && !utf8.RuneStart(text[mid]) {
			mid--
		}
		if mid == lo {
			_, w := utf8.DecodeRuneInString(text[lo:])
			mid += w
		}
		if mid >= hi {
			break
		}
		if n(text[:mid]) <= maxTokens {
			lo = mid
		} else {
			hi = mid
		}
	}
	if i := strings.LastIndexAny(text[:lo], " \n\t"); i > 0 && i >= lo-lo/boundarySlack && n(text[:i]) <= maxTokens {
		return text[:i]
	}
	return text[:lo]
}

// countFunc returns an unmemoized counter: memoizing every probe of a binary
// search would only evict useful entries.
func countFunc() func(string) int {
	r := ranksInstance()
	if r == nil {
		return func(s string) int { return len(s) / 4 }
	}
	return func(s string) int { return count(r, s) }
}

// Method names the counting method in use: "o200k" or "bytes4".
func Method() string {
	if ranksInstance() == nil {
		return methodBytes
	}
	return methodBPE
}

type lruEntry struct {
	key   uint64
	count int
}

// lru is a fixed-capacity least-recently-used map of text hash to token count.
type lru struct {
	capacity int

	mu    sync.Mutex // protects order and items
	order *list.List
	items map[uint64]*list.Element
}

func newLRU(capacity int) *lru {
	return &lru{capacity: capacity, order: list.New(), items: make(map[uint64]*list.Element, capacity)}
}

func (l *lru) get(key uint64) (int, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	el, ok := l.items[key]
	if !ok {
		return 0, false
	}
	l.order.MoveToFront(el)
	return el.Value.(*lruEntry).count, true
}

func (l *lru) put(key uint64, count int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if el, ok := l.items[key]; ok {
		el.Value.(*lruEntry).count = count
		l.order.MoveToFront(el)
		return
	}
	l.items[key] = l.order.PushFront(&lruEntry{key: key, count: count})
	if l.order.Len() > l.capacity {
		oldest := l.order.Back()
		l.order.Remove(oldest)
		delete(l.items, oldest.Value.(*lruEntry).key)
	}
}

func (l *lru) len() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.order.Len()
}

func (l *lru) reset() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.order.Init()
	l.items = make(map[uint64]*list.Element, l.capacity)
}
