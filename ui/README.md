# Dashboard UI (`ui/`)

React + MUI SPA for the operator dashboard (production: **http://localhost:7830/dashboard/**).

## Dev

```bash
make ui-dev          # Vite :5173, proxies /api and /ws to :7830
make ui-build        # production bundle → internal/dashboard/ui/dist
```

## Storybook

Stories render the **same React components** as production (dark `dashboardTheme`), with fixture data — not the removed templ/HTMX UI.

```bash
make storybook              # http://localhost:6008
make build-storybook        # → docs/storybook-static/ (gitignored)
make dashboard-screenshot   # capture docs/images/*.png from Storybook
make verify-stories         # Playwright smoke on Overview + Memory
```

Optional: with ast-mcp running, `npm run verify-visual-vs-live` opens Storybook Overview vs live `:7830` for a manual visual check (Webwright is also fine).

Story IDs: [`src/storybook/STORY_IDS.md`](src/storybook/STORY_IDS.md).
