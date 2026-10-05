package logging

import (
	"context"
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
type handler struct {
	inner slog.Handler
}

func (h *handler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.inner.Enabled(ctx, level)
}

func (h *handler) Handle(ctx context.Context, r slog.Record) error {
	out := slog.NewRecord(r.Time, r.Level, r.Message, r.PC)
	r.Attrs(func(a slog.Attr) bool {
		out.AddAttrs(rewrite(a)...)
		return true
	})
	return h.inner.Handle(ctx, out)
}

func (h *handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	var rewritten []slog.Attr
	for _, a := range attrs {
		rewritten = append(rewritten, rewrite(a)...)
	}
	return &handler{inner: h.inner.WithAttrs(rewritten)}
}

func (h *handler) WithGroup(name string) slog.Handler {
	return &handler{inner: h.inner.WithGroup(name)}
}

// rewrite expands errors (bare or keyed) and bare LogValuers; everything else passes through.
func rewrite(a slog.Attr) []slog.Attr {
	if err, ok := a.Value.Any().(error); ok {
		key := a.Key
		if key == badKey {
			key = "error"
		}
		return errorAttrs(key, err)
	}
	if a.Key != badKey {
		return []slog.Attr{a}
	}
	v := a.Value.Resolve()
	if v.Kind() == slog.KindGroup {
		return v.Group()
	}
	return []slog.Attr{{Key: badKey, Value: v}}
}

// errorAttrs renders err as key=<message> followed by its errs codes and fields, sorted by key.
func errorAttrs(key string, err error) []slog.Attr {
	attrs := []slog.Attr{slog.String(key, err.Error())}
	if codes := errs.CodesOf(err); len(codes) > 0 {
		attrs = append(attrs, slog.Any(key+"_codes", codes.Strings()))
	}
	fields := errs.FieldsOf(err)
	for _, k := range slices.Sorted(maps.Keys(fields)) {
		attrs = append(attrs, slog.Any(k, fields[k]))
	}
	return attrs
}
