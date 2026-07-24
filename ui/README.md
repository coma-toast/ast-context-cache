# AST Context Cache Dashboard (React)

Operator UI built with **React + Vite + MUI**, embedded in the Go binary at `/dashboard/`.

## Development

```bash
# Terminal 1: ast-mcp (API + WS on :7830)
make run

# Terminal 2: Vite dev server (proxies /api and /ws)
make ui-dev
# open http://localhost:5173/dashboard/
```

## Production build

```bash
make ui-build   # outputs to internal/dashboard/ui/dist
make build      # includes ui-build
```

Root `/` redirects to `/dashboard/`. Realtime updates use WebSocket `/ws` with `refresh` messages (not HTML partials).

## API

JSON endpoints under `/api/dashboard/*` plus existing `/api/*` routes. See `internal/dashboard/react_api.go`. Prometheus scrape endpoint: `GET /metrics` on the same port (:7830).
