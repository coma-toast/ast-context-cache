package dashboard

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func serveKB(t *testing.T, h http.Handler, method, path, body, origin string) (int, http.Header, []byte) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "127.0.0.1:7830"
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	return rr.Code, rr.Header(), rr.Body.Bytes()
}

func TestKBCRUDLifecycleAndFilters(t *testing.T) {
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)

	origEmb := mcp.GetEmbedder()
	mcp.SetEmbedder(embedder.NewHashEmbedder(search.VectorDims))
	t.Cleanup(func() { mcp.SetEmbedder(origEmb) })

	h := NewHandler("127.0.0.1")

	// 1. Create a fact (default actor = "human", secret redaction warning).
	createBody := `{
		"kind": "fact",
		"scope": "project",
		"project_id": "/repo/sauger",
		"subject": "api_key_note",
		"predicate": "uses",
		"object": "token sk-proj-abcdefghijklmnopqrstuvwxyz123456",
		"tags": ["security", "auth"]
	}`
	code, _, raw := serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry", createBody, "")
	require.Equal(t, http.StatusOK, code, string(raw))

	var createResp struct {
		Entry          memory.Entry      `json:"entry"`
		Warnings       []string          `json:"warnings"`
		RedactedFields []string          `json:"redacted_fields"`
		Conflicts      []memory.Conflict `json:"conflicts"`
	}
	require.NoError(t, json.Unmarshal(raw, &createResp), string(raw))
	require.NotEmpty(t, createResp.Entry.Ref)
	assert.Equal(t, "human", createResp.Entry.Actor)
	assert.Equal(t, memory.ScopeProject, createResp.Entry.Scope)
	assert.Equal(t, "/repo/sauger", createResp.Entry.ProjectPath)
	assert.Contains(t, createResp.Entry.Object, "[REDACTED:api_key]")
	assert.NotContains(t, createResp.Entry.Object, "sk-proj-abcdefghijklmnopqrstuvwxyz123456")
	assert.NotEmpty(t, createResp.Warnings)
	assert.Equal(t, []string{"object"}, createResp.RedactedFields)
	factRef := createResp.Entry.Ref

	// Create a global preference and another project's event to test scoping/filtering.
	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry", `{
		"kind": "preference",
		"scope": "global",
		"subject": "user",
		"predicate": "prefers",
		"object": "gofumpt formatting",
		"actor": "jason",
		"verified": true
	}`, "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var prefResp struct {
		Entry memory.Entry `json:"entry"`
	}
	require.NoError(t, json.Unmarshal(raw, &prefResp))
	prefRef := prefResp.Entry.Ref

	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry", `{
		"kind": "event",
		"scope": "project",
		"project_path": "/repo/other",
		"title": "Other project deploy",
		"severity": "warning",
		"occurred_at": "2026-11-10T12:00:00Z"
	}`, "")
	require.Equal(t, http.StatusOK, code, string(raw))

	// 2. List/Filter: project_id without scope includes project + global, excludes /repo/other.
	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb?project_id=/repo/sauger", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var listResp memory.KBQueryResult
	require.NoError(t, json.Unmarshal(raw, &listResp))
	assert.Equal(t, 2, listResp.Total)
	assert.Len(t, listResp.Entries, 2)

	// Filter by kinds, tags, verified_only, q.
	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb?kinds=preference&verified_only=true&actor=jason&q=gofumpt", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &listResp))
	require.Len(t, listResp.Entries, 1)
	assert.Equal(t, prefRef, listResp.Entries[0].Ref)

	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb?tags=security&project_id=/repo/sauger", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &listResp))
	require.Len(t, listResp.Entries, 1)
	assert.Equal(t, factRef, listResp.Entries[0].Ref)

	// Invalid filter returns 400.
	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/kb?kinds=bogus", "", "")
	assert.Equal(t, http.StatusBadRequest, code)

	// 3. Get detail & history.
	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb/entry?ref="+factRef, "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var detailResp struct {
		Entry     memory.Entry          `json:"entry"`
		History   []memory.HistoryEntry `json:"history"`
		Conflicts []memory.Conflict     `json:"conflicts"`
	}
	require.NoError(t, json.Unmarshal(raw, &detailResp))
	assert.Equal(t, factRef, detailResp.Entry.Ref)
	require.Len(t, detailResp.History, 1)

	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/kb/entry?ref=mem_missing", "", "")
	assert.Equal(t, http.StatusNotFound, code)
	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/kb/entry", "", "")
	assert.Equal(t, http.StatusBadRequest, code)

	// 4. Update in-place with secret redaction warning.
	putInPlace := fmt.Sprintf(`{"ref":%q,"body":"password=supersecretpassword123"}`, factRef)
	code, _, raw = serveKB(t, h, http.MethodPut, "/api/dashboard/kb/entry", putInPlace, "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var updateResp struct {
		Entry          memory.Entry `json:"entry"`
		Warnings       []string     `json:"warnings"`
		RedactedFields []string     `json:"redacted_fields"`
	}
	require.NoError(t, json.Unmarshal(raw, &updateResp), string(raw))
	assert.Equal(t, factRef, updateResp.Entry.Ref)
	assert.Contains(t, updateResp.Entry.Body, "[REDACTED:password]")
	assert.NotContains(t, updateResp.Entry.Body, "supersecretpassword123")
	assert.NotEmpty(t, updateResp.Warnings)

	// 5. Update versioned (new_version: true).
	putVersioned := fmt.Sprintf(`{"ref":%q,"object":"clean token","new_version":true}`, factRef)
	code, _, raw = serveKB(t, h, http.MethodPut, "/api/dashboard/kb/entry", putVersioned, "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &updateResp))
	factV2Ref := updateResp.Entry.Ref
	require.NotEqual(t, factRef, factV2Ref)

	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb/entry?ref="+factV2Ref, "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &detailResp))
	require.Len(t, detailResp.History, 2)

	// 6. Verify action (set true then false).
	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry/action",
		fmt.Sprintf(`{"ref":%q,"action":"verify","verified":true}`, factV2Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))
	peeked, err := memory.Peek(factV2Ref)
	require.NoError(t, err)
	assert.True(t, peeked.Verified)

	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry/action",
		fmt.Sprintf(`{"ref":%q,"action":"verify","verified":false}`, factV2Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))
	peeked, err = memory.Peek(factV2Ref)
	require.NoError(t, err)
	assert.False(t, peeked.Verified)

	// 7. Forget & Restore actions.
	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry/action",
		fmt.Sprintf(`{"ref":%q,"action":"forget"}`, factV2Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))
	peeked, err = memory.Peek(factV2Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, peeked.ValidUntil)

	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb?status=archived&project_id=/repo/sauger", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &listResp))
	assert.GreaterOrEqual(t, listResp.Total, 1)

	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry/action",
		fmt.Sprintf(`{"ref":%q,"action":"restore"}`, factV2Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))
	peeked, err = memory.Peek(factV2Ref)
	require.NoError(t, err)
	assert.Empty(t, peeked.ValidUntil)

	// 8. Quarantine approve & reject actions.
	qApproveRes, err := memory.Store(memory.StoreInput{
		Kind:  memory.KindDiscovery,
		Scope: memory.ScopeGlobal,
		Title: "Quarantined discovery to approve",
	})
	require.NoError(t, err)
	qRejectRes, err := memory.Store(memory.StoreInput{
		Kind:  memory.KindDiscovery,
		Scope: memory.ScopeGlobal,
		Title: "Quarantined discovery to reject",
	})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET quarantined = 1 WHERE ref IN (?, ?)`, qApproveRes.Ref, qRejectRes.Ref)
	require.NoError(t, err)

	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb?status=quarantined", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &listResp))
	assert.Equal(t, 2, listResp.Total)

	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry/action",
		fmt.Sprintf(`{"ref":%q,"action":"approve_quarantine"}`, qApproveRes.Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))
	approvedEntry, err := memory.Peek(qApproveRes.Ref)
	require.NoError(t, err)
	assert.Equal(t, 0, approvedEntry.Quarantined)
	assert.Empty(t, approvedEntry.ValidUntil)

	code, _, raw = serveKB(t, h, http.MethodPost, "/api/dashboard/kb/entry/action",
		fmt.Sprintf(`{"ref":%q,"action":"reject_quarantine"}`, qRejectRes.Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))
	rejectedEntry, err := memory.Peek(qRejectRes.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, rejectedEntry.ValidUntil)
	assert.Equal(t, memory.ReasonQuarantine, rejectedEntry.InvalidatedReason)

	// 9. Export Markdown & JSON.
	code, hdr, raw := serveKB(t, h, http.MethodGet, "/api/dashboard/kb/export?format=markdown&project_id=/repo/sauger", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	assert.Equal(t, "text/markdown; charset=utf-8", hdr.Get("Content-Type"))
	assert.Equal(t, `attachment; filename="knowledge-base.md"`, hdr.Get("Content-Disposition"))
	assert.Contains(t, string(raw), "# Knowledge Base Export")
	assert.Contains(t, string(raw), factV2Ref)

	code, hdr, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/kb/export?format=json&project_id=/repo/sauger", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	assert.Equal(t, "application/json", hdr.Get("Content-Type"))
	assert.Equal(t, `attachment; filename="knowledge-base.json"`, hdr.Get("Content-Disposition"))
	var exported []memory.Entry
	require.NoError(t, json.Unmarshal(raw, &exported))
	assert.NotEmpty(t, exported)

	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/kb/export?format=xml", "", "")
	assert.Equal(t, http.StatusBadRequest, code)
}

func TestContextNotesEndpointsAndHandoffActive(t *testing.T) {
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)

	ctx, cancel := context.WithCancel(context.Background())
	svc := handoff.New(ctx, nil)
	handoff.SetDefault(svc)
	t.Cleanup(func() {
		handoff.SetDefault(nil)
		cancel()
	})

	// Regular note + edit so revisions exist.
	stored, err := contextnotes.Store("sess-cn", "Initial plan content\nline two", "auth-plan", "/repo/sauger", "auth,plan", "plan", nil, nil)
	require.NoError(t, err)
	_, err = contextnotes.Edit(contextnotes.EditInput{
		Action:    contextnotes.EditRewrite,
		Ref:       stored.Ref,
		SessionID: "sess-cn",
		Content:   "Updated plan content with more details",
	}, nil)
	require.NoError(t, err)

	// Active handoff result note.
	c1, err := svc.Create(ctx, handoff.CreateRequest{SessionID: "parent-active", ProjectPath: "/repo/sauger", Brief: "active work", Label: "active"})
	require.NoError(t, err)
	o1, err := svc.Open(ctx, handoff.OpenRequest{Handoff: c1.Ref})
	require.NoError(t, err)
	r1, err := svc.Complete(ctx, handoff.CompleteRequest{SessionID: o1.SessionID, Content: "active handoff result body", Summary: "active sum"})
	require.NoError(t, err)

	// Expired handoff result note.
	c2, err := svc.Create(ctx, handoff.CreateRequest{SessionID: "parent-expired", ProjectPath: "/repo/sauger", Brief: "expired work", Label: "expired"})
	require.NoError(t, err)
	o2, err := svc.Open(ctx, handoff.OpenRequest{Handoff: c2.Ref})
	require.NoError(t, err)
	r2, err := svc.Complete(ctx, handoff.CompleteRequest{SessionID: o2.SessionID, Content: "expired handoff result body", Summary: "expired sum"})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE handoff_trees SET last_access_at = '2000-01-01 00:00:00' WHERE tree_id = ?`, c2.TreeID)
	require.NoError(t, err)

	h := NewHandler("127.0.0.1")

	// List all context notes for /repo/sauger.
	code, _, raw := serveKB(t, h, http.MethodGet, "/api/dashboard/context-notes?project_id=/repo/sauger", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var listResp struct {
		Items   []contextNoteListItem `json:"items"`
		Total   int                   `json:"total"`
		Page    int                   `json:"page"`
		PerPage int                   `json:"per_page"`
	}
	require.NoError(t, json.Unmarshal(raw, &listResp), string(raw))
	assert.Equal(t, 3, listResp.Total)

	byRef := map[string]contextNoteListItem{}
	for _, it := range listResp.Items {
		byRef[it.Ref] = it
	}
	assert.False(t, byRef[stored.Ref].HandoffTreeActive)
	assert.Equal(t, 2, byRef[stored.Ref].Revision)
	assert.Contains(t, byRef[stored.Ref].Snippet, "Updated plan content")
	assert.True(t, byRef[r1.ResultRef].HandoffTreeActive)
	assert.False(t, byRef[r2.ResultRef].HandoffTreeActive)

	// Filter by kind, tags, session_id, q.
	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/context-notes?kind=plan&tags=auth&session_id=sess-cn&q=Updated", "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &listResp))
	require.Len(t, listResp.Items, 1)
	assert.Equal(t, stored.Ref, listResp.Items[0].Ref)

	// Detail with revisions & without bumping access_count.
	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/context-notes/detail?ref="+stored.Ref, "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var detailResp struct {
		Note struct {
			Ref         string `json:"ref"`
			Content     string `json:"content"`
			AccessCount int    `json:"access_count"`
			Revision    int    `json:"revision"`
		} `json:"note"`
		Revisions         []contextNoteRevisionRow `json:"revisions"`
		ActiveHandoffTree bool                     `json:"active_handoff_tree"`
	}
	require.NoError(t, json.Unmarshal(raw, &detailResp), string(raw))
	assert.Equal(t, stored.Ref, detailResp.Note.Ref)
	assert.Equal(t, "Updated plan content with more details", detailResp.Note.Content)
	assert.Equal(t, 0, detailResp.Note.AccessCount)
	assert.Equal(t, 2, detailResp.Note.Revision)
	assert.False(t, detailResp.ActiveHandoffTree)
	require.Len(t, detailResp.Revisions, 1)
	assert.Equal(t, 1, detailResp.Revisions[0].Revision)
	assert.Equal(t, "Initial plan content\nline two", detailResp.Revisions[0].Content)
	assert.Equal(t, contextnotes.EditRewrite, detailResp.Revisions[0].Op)

	// Detail for active & expired handoff results.
	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/context-notes/detail?ref="+r1.ResultRef, "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &detailResp))
	assert.True(t, detailResp.ActiveHandoffTree)

	code, _, raw = serveKB(t, h, http.MethodGet, "/api/dashboard/context-notes/detail?ref="+r2.ResultRef, "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &detailResp))
	assert.False(t, detailResp.ActiveHandoffTree)

	// DELETE context note.
	code, _, raw = serveKB(t, h, http.MethodDelete, "/api/dashboard/context-notes?ref="+stored.Ref, "", "")
	require.Equal(t, http.StatusOK, code, string(raw))

	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/context-notes/detail?ref="+stored.Ref, "", "")
	assert.Equal(t, http.StatusNotFound, code)

	code, _, _ = serveKB(t, h, http.MethodDelete, "/api/dashboard/context-notes?ref=ctx_missing", "", "")
	assert.Equal(t, http.StatusNotFound, code)
	code, _, _ = serveKB(t, h, http.MethodDelete, "/api/dashboard/context-notes", "", "")
	assert.Equal(t, http.StatusBadRequest, code)
}

func TestDocEntriesEndpoint(t *testing.T) {
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)

	docPath := filepath.Join(t.TempDir(), "guide.md")
	docContent := "# Getting Started\n\nInstall the CLI tool.\n\n## Authentication\n\nConfigure your bearer token.\n"
	require.NoError(t, os.WriteFile(docPath, []byte(docContent), 0o644))

	sourceID, err := docs.AddSource("guide", "markdown", "file://"+docPath, "1.0")
	require.NoError(t, err)
	_, err = docs.UpdateSource(sourceID)
	require.NoError(t, err)

	h := NewHandler("127.0.0.1")

	code, _, raw := serveKB(t, h, http.MethodGet, fmt.Sprintf("/api/dashboard/doc-entries?source_id=%d", sourceID), "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	var resp struct {
		SourceID int             `json:"source_id"`
		Entries  []docs.DocEntry `json:"entries"`
		Total    int             `json:"total"`
	}
	require.NoError(t, json.Unmarshal(raw, &resp), string(raw))
	assert.Equal(t, sourceID, resp.SourceID)
	assert.Equal(t, 2, resp.Total)
	require.Len(t, resp.Entries, 2)

	// Substring filter.
	code, _, raw = serveKB(t, h, http.MethodGet, fmt.Sprintf("/api/dashboard/doc-entries?source_id=%d&q=bearer", sourceID), "", "")
	require.Equal(t, http.StatusOK, code, string(raw))
	require.NoError(t, json.Unmarshal(raw, &resp))
	assert.Equal(t, 1, resp.Total)
	require.Len(t, resp.Entries, 1)
	assert.Equal(t, "Authentication", resp.Entries[0].Title)

	// Validation and 404.
	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/doc-entries", "", "")
	assert.Equal(t, http.StatusBadRequest, code)
	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/doc-entries?source_id=abc", "", "")
	assert.Equal(t, http.StatusBadRequest, code)
	code, _, _ = serveKB(t, h, http.MethodGet, "/api/dashboard/doc-entries?source_id=99999", "", "")
	assert.Equal(t, http.StatusNotFound, code)
}

func TestKBEndpointsMethodAndHTTPGuard(t *testing.T) {
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)

	h := NewHandler("127.0.0.1")

	// Method checks (405).
	methodCases := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/dashboard/kb"},
		{http.MethodDelete, "/api/dashboard/kb/entry"},
		{http.MethodGet, "/api/dashboard/kb/entry/action"},
		{http.MethodPost, "/api/dashboard/kb/export"},
		{http.MethodPost, "/api/dashboard/context-notes"},
		{http.MethodPost, "/api/dashboard/context-notes/detail"},
		{http.MethodPost, "/api/dashboard/doc-entries"},
	}
	for _, tc := range methodCases {
		code, _, _ := serveKB(t, h, tc.method, tc.path, "", "")
		assert.Equal(t, http.StatusMethodNotAllowed, code, "%s %s", tc.method, tc.path)
	}

	// httpguard rejects untrusted Origin on POST/PUT/DELETE with 403.
	guardCases := []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodPost, "/api/dashboard/kb/entry", `{"kind":"fact","scope":"global","subject":"a","object":"b"}`},
		{http.MethodPut, "/api/dashboard/kb/entry", `{"ref":"mem_123","body":"x"}`},
		{http.MethodPost, "/api/dashboard/kb/entry/action", `{"ref":"mem_123","action":"forget"}`},
		{http.MethodDelete, "/api/dashboard/context-notes?ref=ctx_123", ""},
	}
	for _, tc := range guardCases {
		code, _, raw := serveKB(t, h, tc.method, tc.path, tc.body, "https://evil.example")
		assert.Equal(t, http.StatusForbidden, code, "%s %s: %s", tc.method, tc.path, string(raw))
	}
}

func TestFlushContextAPIEmitsKnowledgeBaseNotification(t *testing.T) {
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)

	stored, err := contextnotes.Store("sess-flush", "flush me", "lbl", "/repo/flush", nil, "", nil, nil)
	require.NoError(t, err)

	var got atomic.Uint32
	done := make(chan struct{}, 1)
	realtime.SetHandler(func(mask realtime.Reason) {
		got.Store(uint32(mask))
		select {
		case done <- struct{}{}:
		default:
		}
	})
	t.Cleanup(func() { realtime.SetHandler(flushRefreshBroadcast) })

	h := NewHandler("127.0.0.1")
	code, _, raw := serveKB(t, h, http.MethodPost, "/api/flush-context",
		fmt.Sprintf(`{"refs":[%q]}`, stored.Ref), "")
	require.Equal(t, http.StatusOK, code, string(raw))

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for realtime notification")
	}
	mask := realtime.Reason(got.Load())
	assert.NotZero(t, mask&realtime.KnowledgeBase)
	assert.NotZero(t, mask&realtime.Stats)
}
