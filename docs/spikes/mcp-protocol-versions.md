# Spike: MCP protocol versions for the v4.0 Streamable HTTP server

**2026-07-28: VERIFIED (official spec URL https://modelcontextprotocol.io/specification/2026-07-28)**

Verified 2026-10-05. Evidence:

- https://modelcontextprotocol.io/specification/versioning says: "The **current** protocol version is **2026-07-28**".
- The spec repo `modelcontextprotocol/modelcontextprotocol` has a GitHub release and tag `2026-07-28`, published 2026-07-28T16:47:49Z. The `2026-07-28-RC` tag was published 2026-05-29. The tags `2025-11-25`, `2025-06-18`, `2025-03-26` and `2024-11-05` also exist.
- Blog posts: https://blog.modelcontextprotocol.io/posts/2026-07-28/ and the release candidate post https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/.
- Changelog: https://modelcontextprotocol.io/specification/2026-07-28/changelog

Revisions in play: `2024-11-05` (HTTP+SSE, deprecated), `2025-03-26`, `2025-06-18`, `2025-11-25` (the last "legacy" revision, which uses the `initialize` handshake) and `2026-07-28` (the first "modern" revision: stateless, with per-request `_meta`).

> **Implication for v4.0:** this spike was meant to decide whether to negotiate only `2025-06-18` and `2024-11-05`. That fallback is not needed. A server that targets 2026-10 clients should be **dual-era**:
>
> - If a request arrives with `initialize`, serve the legacy versions over the same `/mcp` endpoint. Negotiate `2025-11-25`, `2025-06-18` and `2025-03-26`, and accept `2024-11-05` only if the old HTTP+SSE endpoints are hosted too.
> - If a request carries `_meta["io.modelcontextprotocol/protocolVersion"]`, serve it statelessly as `2026-07-28`.
>
> Claude Code's v2 MCP runtime (TS SDK 2.0) "asks HTTP servers whether they support the newer revision, and uses it with those that do". It then receives `list_changed` over a held-open `subscriptions/listen` stream (source: https://code.claude.com/docs/en/mcp, "MCP client runtimes"). As a result, live `tools/list_changed` delivery to Claude Code on the new revision **requires** `subscriptions/listen`. A legacy GET SSE stream is not enough.

---

## Part A1 — What a Streamable HTTP server must do (2025-06-18)

Sources: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports and https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle (fetched via ast-context-cache `fetch_doc`, 2026-10-05).

### Endpoint and security
- Provide a **single MCP endpoint path** (for example `/mcp`) that supports **POST and GET**.
- Servers **MUST validate the `Origin` header** on all incoming connections to prevent DNS rebinding.
  - The 2025-06-18 text states only the MUST.
  - **2025-11-25 adds that an invalid `Origin` that is present MUST get `403 Forbidden`.** The body MAY be a JSON-RPC error with no `id`. 2026-07-28 keeps this.
- When running locally, servers SHOULD bind to `127.0.0.1` only, not `0.0.0.0`, and SHOULD implement authentication.
- Messages are UTF-8 JSON-RPC.

### POST (client → server)
1. Every client JSON-RPC message is a **new HTTP POST** to the endpoint.
2. The client **MUST** send `Accept` listing **both** `application/json` and `text/event-stream`.
3. The body is a **single** JSON-RPC request, notification, or response. 2025-06-18 removed JSON-RPC batching.
4. If the body is a **notification or response**, the server returns **`202 Accepted` with no body**. If it cannot accept it, the server returns an HTTP error status such as 400, and the body MAY be a JSON-RPC error with no `id`.
5. If the body is a **request**, the server returns either `Content-Type: application/json` (one JSON object) or `Content-Type: text/event-stream` (an SSE stream). The client MUST support both.
6. When the server uses SSE for a POST:
   - The stream SHOULD eventually contain the response.
   - The server MAY send related requests and notifications before the response.
   - The server SHOULD NOT close the stream before the response, unless the session expires, and SHOULD close it after the response.
   - A disconnect is **not** a cancellation. The client sends `notifications/cancelled`.

### GET (server → client stream)
- The client MAY send GET with `Accept: text/event-stream` to open a standalone SSE stream.
- The server **MUST** respond with either `Content-Type: text/event-stream` or **`405 Method Not Allowed`** (no SSE offered).
- On that stream the server MAY send requests and notifications that are unrelated to in-flight requests. It **MUST NOT** send a JSON-RPC response there unless it is resuming.
- With multiple streams, each server message goes on **exactly one** stream. The server must not broadcast.
- Resumability is optional. SSE `id`s must be unique per session or stream. The client resumes with GET plus `Last-Event-ID`, and the server replays only that stream.

### Sessions (`Mcp-Session-Id`)
1. The server MAY assign a session ID by setting the `Mcp-Session-Id` header **on the HTTP response that carries the `InitializeResult`**.
   - The ID should be cryptographically secure (UUID, JWT or hash).
   - It may use only visible ASCII (0x21–0x7E).
2. If the server assigned one, the client **MUST echo `Mcp-Session-Id` on every later request**. A server that requires sessions SHOULD answer a non-initialize request without the header with **400**.
3. The server MAY end a session at any time. After that it **MUST answer requests carrying that ID with `404 Not Found`**, and the client then MUST re-initialize without an ID.
4. The client SHOULD send **HTTP DELETE** with `Mcp-Session-Id` to end a session. The server MAY answer **405** if it does not allow client-side termination.
5. 2025-11-25 spells the header `MCP-Session-Id`. HTTP header names are case-insensitive, so this is the same header.

### `MCP-Protocol-Version` header
- After initialization, the client **MUST** send `MCP-Protocol-Version: <negotiated>` on all later HTTP requests.
- If the header is missing and the server has no other way to know the version, it SHOULD assume `2025-03-26`.
- If the header carries an invalid or unsupported version, the server **MUST** return **`400 Bad Request`**.

### Lifecycle and version negotiation
- `initialize` MUST be the first interaction. The client sends `protocolVersion`, `capabilities` and `clientInfo`.
- **Negotiation:**
  - The client sends the latest version it supports.
  - If the server supports that version, it **MUST echo the same version**.
  - Otherwise it **MUST respond with another version it supports**, which SHOULD be its latest.
  - If the client does not support the returned version, it SHOULD disconnect.
- The server returns `protocolVersion`, `capabilities`, `serverInfo` and optional `instructions`.
- The client then sends `notifications/initialized`. On HTTP this is a POST, and the server answers 202.
- Before `initialized`, the server SHOULD NOT send requests other than `ping` and logging.
- Example error for an unsupported version: JSON-RPC `-32602` with `data: {supported:[...], requested}`.

### Tools `listChanged`
- Declare `capabilities.tools = { "listChanged": true }` to promise change notifications.
- When the tool set changes, send `{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}`. Deliver it on the GET SSE stream, or on a POST SSE stream if it relates to that request. The client then calls `tools/list` again.

### `ping`
- Either side may send `{"jsonrpc":"2.0","id":..,"method":"ping"}`. The receiver **MUST promptly reply with an empty result `{}`**. The server must accept `ping` even before `initialized`.

### Shutdown (HTTP)
- Close the HTTP connections. Clients SHOULD also send DELETE to end the session.

---

## Part A2 — 2026-07-28 changes relevant to a server

Sources:
- https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
- https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning
- https://modelcontextprotocol.io/specification/2026-07-28/basic/index
- https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions
- https://modelcontextprotocol.io/specification/2026-07-28/server/discover
- https://modelcontextprotocol.io/specification/2026-07-28/changelog

### Stateless core, no handshake, no sessions
- **`initialize` and `notifications/initialized` are removed.** Every request carries the following in `params._meta`:
  - `io.modelcontextprotocol/protocolVersion` (**required**)
  - `io.modelcontextprotocol/clientCapabilities` (**required**)
  - `io.modelcontextprotocol/clientInfo` (SHOULD)
  - `io.modelcontextprotocol/logLevel` (optional)
- If a required `_meta` field is missing, reject with JSON-RPC `-32602` and HTTP **400**.
- Servers SHOULD put `io.modelcontextprotocol/serverInfo` in every result's `_meta`.
- Servers **MUST NOT** depend on earlier requests or the connection for context.
- **`Mcp-Session-Id` is removed.** A server that implements only 2026-07-28 SHOULD **ignore** an incoming `Mcp-Session-Id` header and never mint or echo one.
- Cross-call state uses explicit, server-minted handles passed as tool arguments.
- List endpoints no longer vary by connection.
- Every result needs **`resultType`**: `"complete"` for normal results, `"input_required"` for MRTR.

### Versioning
- **There is no negotiation handshake.** Each request declares its version, and the server accepts or rejects each one independently.
- If the server does not support the version, it returns **`UnsupportedProtocolVersionError`** with code **`-32022`**, `data: {supported:[...], requested}`, and HTTP **400**.
- **`server/discover` is mandatory.** It returns `supportedVersions`, `capabilities`, `instructions`, `_meta.serverInfo`, `ttlMs` and `cacheScope`.
- **Dual-era servers are explicitly allowed.** If a request carries modern `_meta`, serve it statelessly. If an `initialize` arrives, use legacy semantics scoped to that HTTP session. Both may run on the same endpoint.
- A modern-only server SHOULD name its supported versions in any error it returns to `initialize`.
- Error codes:
  - `-32020` HeaderMismatch
  - `-32021` MissingRequiredClientCapability
  - `-32022` UnsupportedProtocolVersion
  - New code must not emit `-32000..-32019`. `-32002` (resource not found) is replaced by `-32602`.

### Transport (POST only)
- **The GET stream endpoint is removed.** If GET or DELETE hits the endpoint, return **405**.
- **SSE resumability is removed.** Ignore `Last-Event-ID`.
- The client sends `Accept: application/json, text/event-stream` and POSTs a single request or notification. **Clients no longer send JSON-RPC responses.**
- A notification gets 202. A request gets `application/json` or a request-scoped `text/event-stream`.
- On an SSE response stream:
  - Only notifications related to the request (progress or log) may appear, followed by the final response.
  - **The server MUST NOT send JSON-RPC requests.** Sampling, elicitation and roots now go through **MRTR**: the server returns an `InputRequiredResult` with `inputRequests`, and the client retries with `inputResponses`.
  - When opening SSE, servers SHOULD send `X-Accel-Buffering: no`.
  - For long streams, send `:` comment keep-alives.
- **Cancellation:** if the client closes the SSE response stream, that **is** the cancellation. The server MUST stop sending for that request. `notifications/cancelled` is used on stdio only.
- `ping`, `logging/setLevel` and `notifications/roots/list_changed` are **removed**.
- Roots, Sampling and Logging are deprecated.

### Required request headers (validate against the body)
- **`MCP-Protocol-Version`** is required on **every** POST. It **must equal** `_meta["io.modelcontextprotocol/protocolVersion"]`.
- **`Mcp-Method`** is required on all requests. **`Mcp-Name`** is required for `tools/call`, `resources/read` and `prompts/get` (`params.name` or `params.uri`).
- Values that are not safe ASCII arrive as `=?base64?…?=`. Decode them before comparing.
- Optional `x-mcp-header` tool parameters map to `Mcp-Param-{Name}` headers, which must be validated if recognized.
- If a header is missing or mismatched, return **400** with JSON-RPC **`-32020` HeaderMismatch**.
- An unknown method returns **404** with `-32601`. The JSON-RPC body distinguishes this from a legacy server's 404.
- A request without `MCP-Protocol-Version` MAY be treated as `2025-03-26` if the server supports pre-2025-06-18 clients. Otherwise reject it.

### Server → client notifications (`list_changed`)
- Clients get long-lived change notifications by POSTing **`subscriptions/listen`** with a `notifications` filter: `toolsListChanged`, `promptsListChanged`, `resourcesListChanged` and `resourceSubscriptions:[uri]`.
- The server answers with an SSE stream that stays open:
  1. **First message:** `notifications/subscriptions/acknowledged`, carrying `_meta["io.modelcontextprotocol/subscriptionId"] = <listen request id>` and the agreed filter. Unsupported types are omitted.
  2. **Then** `notifications/tools/list_changed` and other opted-in notifications, each tagged with the same `subscriptionId`. The server **MUST NOT** send types the client did not request.
  3. **Graceful end:** a normal JSON-RPC response to the listen request (`resultType:"complete"` plus `subscriptionId`), then close.
- Progress and log notifications are **never** sent on the listen stream.
- This replaces both the legacy GET stream and `resources/subscribe`.
- Claude Code behaviour on v2 (from its docs): if the listen stream closes within 10 s, Claude Code retries up to 3 times. If streams that lived longer than 10 s close 5 times in an hour, it backs off for about 6 h. **Keep the stream alive** with comment keep-alives and do not close it on idle.

### Other server-visible changes
- `tools/list`, `prompts/list`, `resources/list`, `resources/read` and `resources/templates/list` results **require `ttlMs` and `cacheScope`** (`"public"` or `"private"`).
- `tools/list` SHOULD return tools in a deterministic order.
- `capabilities.extensions` map. Tasks moved to the extension `io.modelcontextprotocol/tasks`.
- JSON Schema 2020-12 is the default dialect. Do not auto-dereference network `$ref`s.

---

## Part A3 — 2024-11-05 HTTP+SSE (legacy back-compat)

Source: https://modelcontextprotocol.io/specification/2024-11-05/basic/transports. The HTTP+SSE transport has been Deprecated since 2025-03-26 and is formally classified Deprecated in 2026-07-28 (SEP-2596).

- **Two endpoints:** a GET SSE endpoint and a separate POST endpoint.
- On connect, the server **MUST** send an SSE `endpoint` event whose data is the URI for POSTs. All server messages then arrive as SSE `message` events.
- No `Mcp-Session-Id` header. Sessions are implicit, usually a query param in the endpoint URI. No `MCP-Protocol-Version` header and no resumability.
- Origin validation MUST and the localhost-bind SHOULD apply here too.
- **Client fallback**, which is how old clients find old servers:
  - POST `InitializeRequest` to the URL.
  - On 4xx (400, 404 or 405), GET the URL and expect an `endpoint` event.
- **Server back-compat:** keep hosting the old SSE and POST endpoints next to `/mcp`. Combining the old POST endpoint with `/mcp` is possible but discouraged.
- **Recommendation:** do **not** advertise `2024-11-05` from `/mcp` unless legacy `/sse` and `/messages` endpoints actually exist. Otherwise answer an `initialize` that asks for `2024-11-05` with the server's latest legacy version (`2025-11-25`), as the negotiation rules require.

---

## Recommended negotiation table (v4.0 server)

| Client sends | Server behaviour |
| - | - |
| POST with `_meta.protocolVersion = "2026-07-28"` and `MCP-Protocol-Version: 2026-07-28` | Stateless modern path. Validate headers, no session, `subscriptions/listen` for `list_changed`. |
| POST with `_meta.protocolVersion` = other or unknown | 400 with `-32022`, `supported: ["2026-07-28","2025-11-25","2025-06-18","2025-03-26"]` |
| POST `initialize` with `protocolVersion` in {2025-11-25, 2025-06-18, 2025-03-26} | Legacy path. Echo the version, mint `Mcp-Session-Id`, support GET SSE (or 405) and DELETE. |
| POST `initialize` with `2024-11-05` or unknown | Reply `2025-11-25`, the latest legacy version. The client decides. |
| GET or DELETE with no `Mcp-Session-Id` | 405 (modern) |
| `Origin` present and not localhost | 403 |

## UNVERIFIED in this spike
- Which shipping clients send modern (2026-07-28) requests today, other than Claude Code v2 runtime (documented) and the official SDK 2.0 betas. Cursor, OpenCode, Codex, VS Code and JetBrains do not document their MCP protocol revision.
- Whether any client still requires `2024-11-05` HTTP+SSE against a local URL. No target host in docs/host-integration.md needs it: all document Streamable HTTP, and some document automatic SSE fallback.
