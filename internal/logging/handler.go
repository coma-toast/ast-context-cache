package logging

import (
	"context"
	"iter"
	"log/slog"
	"maps"
	"slices"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// badKey is the key slog assigns to an argument that is not part of a key-value pair.
const badKey = "!BADKEY"

// handler rewrites bare arguments before passing records to the inner handler:
// an error becomes error=<message> plus its errs fields, and a LogValuer whose value is a
// group is inlined (so ID types can expand to "handoff=hof_..."). Other bare values keep !BADKEY.
// An error field whose key the call also passes explicitly is dropped, so the explicit value
// wins and the key appears once.
type handler struct {
	inner slog.Handler
	// keys are the attribute keys added through WithAttrs in the current group.
	keys map[string]bool
}

func (h *handler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.inner.Enabled(ctx, level)
}

func (h *handler) Handle(ctx context.Context, r slog.Record) error {
	out := slog.NewRecord(r.Time, r.Level, r.Message, r.PC)
	out.AddAttrs(h.expand(r.Attrs)...)
	return h.inner.Handle(ctx, out)
}

// WithAttrs records the keys it adds, so a later error field of the same name is dropped. An
// error field added here can't be removed later, so it may still repeat a record's own key.
func (h *handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	rewritten := h.expand(slices.Values(attrs))
	keys := maps.Clone(h.keys)
	if keys == nil {
		keys = map[string]bool{}
	}
	for _, a := range rewritten {
		keys[a.Key] = true
	}
	return &handler{inner: h.inner.WithAttrs(rewritten), keys: keys}
}

// WithGroup starts a new key namespace: later attributes can't collide with earlier ones.
func (h *handler) WithGroup(name string) slog.Handler {
	return &handler{inner: h.inner.WithGroup(name)}
}

// expand rewrites attrs in order. An error field is dropped when an explicit attribute (here or
// through WithAttrs) or an earlier error field already has its key.
func (h *handler) expand(attrs iter.Seq[slog.Attr]) []slog.Attr {
	type part struct{ own, fields []slog.Attr }
	var parts []part
	taken := maps.Clone(h.keys)
	if taken == nil {
		taken = map[string]bool{}
	}
	for a := range attrs {
		own, fields := rewrite(a)
		parts = append(parts, part{own, fields})
		for _, o := range own {
			taken[o.Key] = true
		}
	}
	var out []slog.Attr
	for _, p := range parts {
		out = append(out, p.own...)
		for _, f := range p.fields {
			if !taken[f.Key] {
				taken[f.Key] = true
				out = append(out, f)
			}
		}
	}
	return out
}

// rewrite expands errors (bare or keyed) and bare LogValuers; everything else passes through.
// own are the attributes the call states; fields are an error's errs fields.
func rewrite(a slog.Attr) (own, fields []slog.Attr) {
	if err, ok := a.Value.Any().(error); ok {
		key := a.Key
		if key == badKey {
			key = "error"
		}
		return errorAttrs(key, err)
	}
	if a.Key != badKey {
		return []slog.Attr{a}, nil
	}
	v := a.Value.Resolve()
	if v.Kind() == slog.KindGroup {
		return v.Group(), nil
	}
	return []slog.Attr{{Key: badKey, Value: v}}, nil
}

// errorAttrs renders err as key=<message> and its errs codes, plus its fields sorted by key.
func errorAttrs(key string, err error) (own, fields []slog.Attr) {
	own = []slog.Attr{slog.String(key, err.Error())}
	if codes := errs.CodesOf(err); len(codes) > 0 {
		own = append(own, slog.Any(key+"_codes", codes.Strings()))
	}
	f := errs.FieldsOf(err)
	for _, k := range slices.Sorted(maps.Keys(f)) {
		fields = append(fields, slog.Any(k, f[k]))
	}
	return own, fields
}
