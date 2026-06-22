# Code-mode scripts

Per-repo JavaScript scripts for `execute_code`. Search tools (`get_context_capsule`, `search_semantic`, `retrieve`) may attach **`code_script_hints`** when a script matches the query and result count.

## Requirements

- MCP tier: **complete**
- `AST_MCP_CODE_MODE=true` (or enabled in dashboard policy)

## Layout

```
{project_root}/scripts/code-mode/
  manifest.json
  my-filter.js
```

Built-in scripts ship with ast-mcp under `internal/codescripts/builtin/`. Repo entries with the same `id` override built-ins.

## Manifest schema

`manifest.json` is a JSON **array** of entries:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Stable script id (used in hints and `execute_code script_id`) |
| `title` | yes | Short label for hints |
| `description` | no | Why/when to use (shown in hint `why`) |
| `match.tools` | no | Tools that may suggest this script (default: all three search tools) |
| `match.query_regex` | no | Go regexp; if set, query must match |
| `match.min_results` | no | Minimum symbol hits before suggesting |
| `code_file` | yes* | JS file relative to `scripts/code-mode/` |
| `extends` | no | Built-in id to inherit metadata from; repo `code_file` overrides code |

\* Or inherit code from `extends` built-in when `code_file` omitted.

### Example

```json
{
  "id": "group-by-file",
  "title": "Group hits by file",
  "description": "Return file → symbol names instead of full payloads",
  "match": {
    "tools": ["get_context_capsule", "search_semantic", "retrieve"],
    "query_regex": "(?i)(overview|structure|files)",
    "min_results": 8
  },
  "code_file": "group-by-file.js",
  "extends": "group-by-file"
}
```

## Script contract

- Runs in the goja sandbox with **`DATA`** set to parsed JSON from `execute_code` `data`.
- Accept either a results **array** or an object with a **`results`** array (full prior tool JSON).
- **Return** a JSON-serializable value (usually an array or object); only that output enters the model context.

## Security

- `code_file` is resolved under `scripts/code-mode/` with a path jail (`..` rejected).
- Max script size: **32 KiB**.

## Built-in script ids

| id | Typical use |
|----|-------------|
| `compact-symbol-list` | Many hits — names/files/lines only |
| `group-by-file` | Broad exploration / structure queries |
| `filter-by-kind` | Query mentions function/method/class |
| `exports-only` | Export / public API surface queries |
| `dedupe-by-file` | Duplicate files in results |
| `impact-candidates` | Before `get_impact_graph` |

## Agent workflow

1. Search as usual.
2. If `code_script_hints[]` is non-empty, note the top `script_id`.
3. `execute_code(script_id=..., data=<JSON string of results>, project_path=<abs root>)`
4. Use **`result` only** — do not paste full search JSON into chat.
5. Check `tokens_saved` in the response; dashboard **Tokens saved** includes `execute_code`.

## Savings

`tokens_saved = max(0, data_baseline_tokens − tokens_used)` where `data_baseline_tokens` estimates the input `data` JSON and `tokens_used` estimates the returned `result`.
