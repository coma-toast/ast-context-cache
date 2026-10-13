package mcp

import (
	"bytes"
	"encoding/json"
	"sort"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// codeJSONKeyOrder is the key order of a code tool's output=json body (DR-3): the offload stub,
// then the results, then what the call was, counts and budget, savings, and hints. Keys not
// listed follow in sorted order.
var codeJSONKeyOrder = []string{
	"offloaded",
	"results", "symbols", "context", "chunks",
	"query", "file", "language", "mode", "doc_type",
	"total", "withheld", "no_match", "hint", "below_floor", "collapsed", "deduped", "stats", "token_budget", "tokens_remaining",
	"tokens_used", "symbol_baseline_tokens", "full_baseline_tokens", "file_baseline_tokens", "dedup_tokens_saved", "tokens_saved",
	"savings_vs_files", "input_tokens", "output_tokens",
	"code_script_hints", "handoff",
}

// orderedMap is a JSON object that marshals its keys in insertion order.
type orderedMap struct {
	keys []string
	vals map[string]any
}

// orderedBody orders m's keys by order, then the rest sorted.
func orderedBody(m map[string]any, order []string) orderedMap {
	o := orderedMap{vals: m}
	listed := make(map[string]bool, len(order))
	for _, k := range order {
		listed[k] = true
		if _, ok := m[k]; ok {
			o.keys = append(o.keys, k)
		}
	}
	var rest []string
	for k := range m {
		if !listed[k] {
			rest = append(rest, k)
		}
	}
	sort.Strings(rest)
	o.keys = append(o.keys, rest...)
	return o
}

// MarshalJSON emits the keys in order.
func (o orderedMap) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range o.keys {
		if i > 0 {
			b.WriteByte(',')
		}
		key, err := json.Marshal(k)
		if err != nil {
			return nil, errs.WrapMessage("marshal ordered key", err, "key", k)
		}
		val, err := json.Marshal(o.vals[k])
		if err != nil {
			return nil, errs.WrapMessage("marshal ordered value", err, "key", k)
		}
		b.Write(key)
		b.WriteByte(':')
		b.Write(val)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}
