package dashboard

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
)

// No t.Parallel: the db pools and the default handoff service are package globals.

// handoffAPIEnv is a dashboard handler over a fresh database with a real handoff service
// installed as the default, and one tree seeded through it.
type handoffAPIEnv struct {
	h       http.Handler
	created *handoff.CreateResponse
	child   handoff.SessionID
	result  *handoff.CompleteResponse
}

// setupHandoffAPI seeds a tree the way agents would: create, open, complete. The service's
// context is cancelled before the database closes (cleanups run last-in, first-out).
func setupHandoffAPI(t *testing.T) handoffAPIEnv {
	t.Helper()
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)
	ctx, cancel := context.WithCancel(context.Background())
	svc := handoff.New(ctx, nil)
	handoff.SetDefault(svc)
	t.Cleanup(func() {
		handoff.SetDefault(nil)
		cancel()
	})
	created, err := svc.Create(ctx, handoff.CreateRequest{SessionID: "parent-dash", ProjectPath: "/p/dash", Brief: "map the auth flow", Label: "auth"})
	require.NoError(t, err)
	opened, err := svc.Open(ctx, handoff.OpenRequest{Handoff: created.Ref})
	require.NoError(t, err)
	result, err := svc.Complete(ctx, handoff.CompleteRequest{SessionID: opened.SessionID, Content: "the auth flow goes through middleware", Summary: "via middleware"})
	require.NoError(t, err)
	return handoffAPIEnv{h: NewHandler("127.0.0.1"), created: created, child: opened.SessionID, result: result}
}

// serveHandoff sends a request with an optional Origin and returns the status and raw body.
func serveHandoff(t *testing.T, h http.Handler, method, path, body, origin string) (int, []byte) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "127.0.0.1:7830"
	req.Header.Set("Content-Type", "application/json")
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	return rr.Code, rr.Body.Bytes()
}

func getTrees(t *testing.T, h http.Handler) handoffTreesResponse {
	t.Helper()
	code, body := serveHandoff(t, h, http.MethodGet, "/api/dashboard/handoff-trees?limit=20", "", "")
	require.Equal(t, http.StatusOK, code, string(body))
	var out handoffTreesResponse
	require.NoError(t, json.Unmarshal(body, &out), string(body))
	return out
}

func TestHandoffTreesAPIShape(t *testing.T) {
	env := setupHandoffAPI(t)
	code, body := serveHandoff(t, env.h, http.MethodGet, "/api/dashboard/handoff-trees?limit=20", "", "")
	require.Equal(t, http.StatusOK, code, string(body))
	var raw struct {
		Trees  []map[string]json.RawMessage `json:"trees"`
		Limits map[string]int               `json:"limits"`
		Ratio  *float64                     `json:"repeat_search_ratio_24h"`
	}
	require.NoError(t, json.Unmarshal(body, &raw))
	require.NotNil(t, raw.Ratio, "the aggregate repeat ratio is always present")
	assert.Equal(t, 64000, raw.Limits["tree_max_tokens"])
	require.Len(t, raw.Trees, 1)
	for _, k := range []string{
		"tree_id", "root_session_id", "project_path", "created_at", "last_access_at", "expires_at", "expired",
		"tokens_used", "tokens_max", "entries_used", "entries_max", "active_claims", "queued_claims", "search_calls",
		"repeat_calls", "repeat_rate", "tokens_delivered", "tokens_saved", "handoffs",
	} {
		assert.Contains(t, raw.Trees[0], k, "tree key %s", k)
	}
	var handoffs []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw.Trees[0]["handoffs"], &handoffs))
	require.Len(t, handoffs, 1)
	for _, k := range []string{"handoff", "label", "mode", "depth", "parent_session_id", "created_at", "children"} {
		assert.Contains(t, handoffs[0], k, "handoff key %s", k)
	}
	var children []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(handoffs[0]["children"], &children))
	require.Len(t, children, 1)
	for _, k := range []string{
		"session_id", "label", "status", "depth", "opened_at", "last_activity_at", "result_ref", "summary",
		"search_calls", "repeat_calls", "repeat_rate", "tokens_available", "tokens_delivered", "tokens_saved",
		"active_claims", "queued_claims",
	} {
		assert.Contains(t, children[0], k, "child key %s", k)
	}

	out := getTrees(t, env.h)
	tree := out.Trees[0]
	assert.Equal(t, env.created.TreeID, tree.TreeID)
	assert.Equal(t, handoff.SessionID("parent-dash"), tree.RootSessionID)
	assert.Equal(t, "/p/dash", tree.ProjectPath)
	assert.False(t, tree.Expired)
	assert.NotEmpty(t, tree.ExpiresAt)
	assert.Positive(t, tree.TokensUsed, "the result is charged to the tree")
	h := tree.Handoffs[0]
	assert.Equal(t, env.created.Ref, h.Ref)
	assert.Equal(t, "auth", h.Label)
	assert.Equal(t, handoff.ModeFresh, h.Mode)
	assert.Equal(t, 1, h.Depth)
	c := h.Children[0]
	assert.Equal(t, env.child, c.SessionID)
	assert.Equal(t, handoff.StatusDone, c.Status)
	assert.Equal(t, env.result.ResultRef, c.ResultRef)
	assert.Equal(t, "via middleware", c.Summary)
	assert.Positive(t, c.TokensDelivered, "the open digest was delivered")
}

func TestHandoffTreesAPIRejectsBadLimit(t *testing.T) {
	env := setupHandoffAPI(t)
	for _, limit := range []string{"0", "-1", "many"} {
		code, body := serveHandoff(t, env.h, http.MethodGet, "/api/dashboard/handoff-trees?limit="+limit, "", "")
		assert.Equal(t, http.StatusBadRequest, code, "limit=%s: %s", limit, body)
	}
}

func TestHandoffTreeFlush(t *testing.T) {
	env := setupHandoffAPI(t)
	body := `{"tree_id":"` + string(env.created.TreeID) + `"}`
	code, resp := serveHandoff(t, env.h, http.MethodPost, "/api/dashboard/handoff-trees/flush", body, "http://evil.example")
	assert.Equal(t, http.StatusForbidden, code, "a foreign Origin never reaches the flush: %s", resp)
	require.Len(t, getTrees(t, env.h).Trees, 1, "the tree survives the rejected flush")

	code, resp = serveHandoff(t, env.h, http.MethodPost, "/api/dashboard/handoff-trees/flush", `{"tree_id":"nope"}`, "")
	assert.Equal(t, http.StatusBadRequest, code, string(resp))

	code, resp = serveHandoff(t, env.h, http.MethodPost, "/api/dashboard/handoff-trees/flush", body, "")
	require.Equal(t, http.StatusOK, code, string(resp))
	var out struct {
		Status  string                `json:"status"`
		Flushed handoff.FlushResponse `json:"flushed"`
	}
	require.NoError(t, json.Unmarshal(resp, &out))
	assert.Equal(t, "ok", out.Status)
	assert.Equal(t, env.created.TreeID, out.Flushed.TreeID)
	assert.Equal(t, 1, out.Flushed.Handoffs)
	assert.Equal(t, 1, out.Flushed.Children)
	assert.Empty(t, getTrees(t, env.h).Trees, "the flushed tree is gone")
}

func TestSettingsHandoffKnobs(t *testing.T) {
	h := setupFlagsAPI(t)
	for _, v := range []string{"0", "-3", "soon", ""} {
		code, out := serveFlags(t, h, http.MethodPost, "/api/settings", `{"key":"handoff_max_depth","value":"`+v+`"}`)
		assert.Equal(t, http.StatusBadRequest, code, "value %q", v)
		assert.Equal(t, "handoff_max_depth must be a positive integer", out.Error)
	}
	assert.Empty(t, db.GetSetting(handoff.SettingMaxDepth, ""), "rejected values are not stored")
	code, out := serveFlags(t, h, http.MethodPost, "/api/settings", `{"key":"handoff_max_depth","value":" 5 "}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	assert.Equal(t, "5", db.GetSetting(handoff.SettingMaxDepth, ""))
	assert.Equal(t, 5, handoff.LoadLimits().MaxDepth)

	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var settings map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &settings))
	assert.Equal(t, "5", settings[handoff.SettingMaxDepth])
	for _, ls := range handoff.LimitSettings() {
		if ls.Key != handoff.SettingMaxDepth {
			assert.Equal(t, strconv.Itoa(ls.Default), settings[ls.Key], "default for %s", ls.Key)
		}
	}
}
