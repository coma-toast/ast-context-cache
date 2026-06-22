# Dashboard Storybook

Storybook for the **operator dashboard** (Go `templ` + static CSS). Uses the same styles as production (`internal/dashboard/static/styles.css`).

## Commands

```bash
cd dashboard-storybook
npm ci
npm run storybook          # http://localhost:6008
npm run build-storybook    # output: docs/storybook-static/
npm run capture-screenshot # writes docs/images/dashboard-overview.png (requires build)
```

From repo root: `make storybook`, `make dashboard-screenshot`.

## Storybook MCP

With Storybook running, register the MCP server (Cursor project config):

```json
{
  "mcpServers": {
    "ast-dashboard-storybook": {
      "url": "http://localhost:6008/mcp"
    }
  }
}
```

Tools include `preview-stories`, `get-storybook-story-instructions`, and docs listing. Requires `@storybook/addon-mcp` (React Storybook 10.3+).

## Notes

- Fixtures are **HTML snapshots** of dashboard markup, not live Go templates.
- For the live dashboard, run `make run` and open http://localhost:7830.
