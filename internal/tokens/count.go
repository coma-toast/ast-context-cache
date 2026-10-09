// Package tokens counts LLM tokens with the o200k_base BPE vocabulary, served
// offline from the vocabulary embedded by tiktoken-go-loader.
package tokens

import (
	"container/list"
	"sync"

	"github.com/cespare/xxhash/v2"
	"github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/logging"
)

const (
	encodingName = "o200k_base"
	methodBPE    = "o200k"
	methodBytes  = "bytes4"
	memoMinBytes = 512
	memoCapacity = 4096
)

var logger = logging.Tagged("tokens")

// loadEncoder builds the o200k encoder. Tests swap it to exercise the fallback.
var loadEncoder = func() (*tiktoken.Tiktoken, error) {
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
	enc, err := tiktoken.GetEncoding(encodingName)
	if err != nil {
		return nil, errs.WrapMessage("unable to load tokenizer encoding", err, "encoding", encodingName)
	}
	return enc, nil
}

var (
	initOnce sync.Once
	encoder  *tiktoken.Tiktoken // nil when the encoder failed to load
	memo     = newLRU(memoCapacity)
)

func encoderInstance() *tiktoken.Tiktoken {
	initOnce.Do(func() {
		enc, err := loadEncoder()
		if err != nil {
			logger.Warn("TOKENIZER UNAVAILABLE, FALLING BACK TO BYTES/4", "error", err)
			return
		}
		encoder = enc
	})
	return encoder
}

// Count returns the o200k token count of text, or len(text)/4 when the
// tokenizer cannot be loaded. Special-token text is counted as ordinary text.
func Count(text string) int {
	enc := encoderInstance()
	if enc == nil {
		return len(text) / 4
	}
	if len(text) < memoMinBytes {
		return len(enc.EncodeOrdinary(text))
	}
	key := xxhash.Sum64String(text)
	if n, ok := memo.get(key); ok {
		return n
	}
	n := len(enc.EncodeOrdinary(text))
	memo.put(key, n)
	return n
}

// Method names the counting method in use: "o200k" or "bytes4".
func Method() string {
	if encoderInstance() == nil {
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
