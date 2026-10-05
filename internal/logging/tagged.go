package logging

import (
	"context"
	"log/slog"
)

// Tagged returns a logger carrying tag=<tag> that always writes through the current
// slog.Default() handler. Use it for package-level loggers: slog.With binds to whatever default
// exists at package init, which is before main calls Setup.
func Tagged(tag string) *slog.Logger {
	return slog.New(&dynamicHandler{ops: []handlerOp{{attrs: []slog.Attr{slog.String("tag", tag)}}}})
}

// handlerOp is one WithAttrs or WithGroup call, replayed onto the current default handler.
type handlerOp struct {
	attrs []slog.Attr
	group string
}

type dynamicHandler struct {
	ops []handlerOp
}

func (h *dynamicHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return slog.Default().Handler().Enabled(ctx, level)
}

func (h *dynamicHandler) Handle(ctx context.Context, r slog.Record) error {
	return h.resolve().Handle(ctx, r)
}

func (h *dynamicHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &dynamicHandler{ops: append(h.cloneOps(), handlerOp{attrs: attrs})}
}

func (h *dynamicHandler) WithGroup(name string) slog.Handler {
	return &dynamicHandler{ops: append(h.cloneOps(), handlerOp{group: name})}
}

func (h *dynamicHandler) resolve() slog.Handler {
	out := slog.Default().Handler()
	for _, op := range h.ops {
		if op.group != "" {
			out = out.WithGroup(op.group)
			continue
		}
		out = out.WithAttrs(op.attrs)
	}
	return out
}

func (h *dynamicHandler) cloneOps() []handlerOp {
	return append([]handlerOp(nil), h.ops...)
}
