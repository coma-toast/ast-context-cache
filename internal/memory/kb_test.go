package memory

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func TestKBStoreNewKindsDefaultScopesAndValidation(t *testing.T) {
	dbtest.Init(t)

	// Preference with no session_id and no project_path -> global scope, defaults subject/predicate.
	prefGlobal, err := Store(StoreInput{
		Kind: KindPreference,
		Rule: "Always format Go files with gofumpt",
	})
	require.NoError(t, err)
	ePref, err := entryByRef(prefGlobal.Ref)
	require.NoError(t, err)
	assert.Equal(t, ScopeGlobal, ePref.Scope)
	assert.Equal(t, "user", ePref.Subject)
	assert.Equal(t, "prefers", ePref.Predicate)
	assert.Equal(t, "Always format Go files with gofumpt", ePref.Object)

	// Event with project_path and no session_id -> project scope, default severity=info, auto occurred_at.
	evProj, err := Store(StoreInput{
		Kind:        KindEvent,
		ProjectPath: "/repo/sauger",
		Title:       "Deployment succeeded",
		Body:        "Rolled out v2.1 cleanly",
		Tags:        []string{"Deploy", "release", "deploy"},
		Actor:       "ci-bot",
		Category:    "release",
	})
	require.NoError(t, err)
	eEv, err := entryByRef(evProj.Ref)
	require.NoError(t, err)
	assert.Equal(t, ScopeProject, eEv.Scope)
	assert.Equal(t, SeverityInfo, eEv.Severity)
	assert.NotEmpty(t, eEv.OccurredAt)
	assert.Equal(t, []string{"deploy", "release"}, eEv.Tags)

	// ConfigChange with no session_id and no project_path -> global scope, auto occurred_at.
	ccRes, err := Store(StoreInput{
		Kind:        KindConfigChange,
		Target:      "db.max_conns",
		BeforeValue: "10",
		AfterValue:  "25",
		Reason:      "handle concurrent mcp workers",
		Actor:       "jason",
	})
	require.NoError(t, err)
	eCC, err := entryByRef(ccRes.Ref)
	require.NoError(t, err)
	assert.Equal(t, ScopeGlobal, eCC.Scope)
	assert.NotEmpty(t, eCC.OccurredAt)

	// Discovery with session_id only -> session scope.
	discRes, err := Store(StoreInput{
		Kind:      KindDiscovery,
		SessionID: "sess-1",
		Title:     "FTS5 bm25 works best with column filter",
		Body:      "Excluding ref prevents mem_ prefix false positives",
		Category:  "sqlite",
		Verified:  true,
	})
	require.NoError(t, err)
	eDisc, err := entryByRef(discRes.Ref)
	require.NoError(t, err)
	assert.Equal(t, ScopeSession, eDisc.Scope)
	assert.True(t, eDisc.Verified)
	assert.NotEmpty(t, eDisc.VerifiedAt)

	// Body > 16 KB rejected.
	_, err = Store(StoreInput{
		Kind:      KindDiscovery,
		SessionID: "sess-1",
		Title:     "Oversized",
		Body:      strings.Repeat("x", maxBodyBytes+1),
	})
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))

	// ConfigChange validation errors.
	_, err = Store(StoreInput{Kind: KindConfigChange, AfterValue: "25"})
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))

	_, err = Store(StoreInput{Kind: KindConfigChange, Target: "k"})
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))

	// Invalid severity rejected.
	_, err = Store(StoreInput{Kind: KindEvent, Title: "oops", Severity: "panic"})
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

func TestKBSupersessionRules(t *testing.T) {
	dbtest.Init(t)

	// Preference supersedes only preference with same (subject, predicate, scope), not fact.
	factRes, err := Store(StoreInput{
		Kind:      KindFact,
		Scope:     ScopeGlobal,
		Subject:   "user",
		Predicate: "prefers",
		Object:    "dark mode fact",
	})
	require.NoError(t, err)

	pref1, err := Store(StoreInput{
		Kind:               KindPreference,
		Scope:              ScopeGlobal,
		Subject:            "user",
		Predicate:          "prefers",
		Object:             "tabs",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)
	assert.Empty(t, pref1.InvalidatedRefs)

	pref2, err := Store(StoreInput{
		Kind:               KindPreference,
		Scope:              ScopeGlobal,
		Subject:            "user",
		Predicate:          "prefers",
		Object:             "spaces",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)
	assert.Equal(t, []string{pref1.Ref}, pref2.InvalidatedRefs)

	eFact, err := entryByRef(factRes.Ref)
	require.NoError(t, err)
	assert.Empty(t, eFact.ValidUntil, "fact with same triple must not be superseded by preference")

	// ConfigChange always chains on target within same scope with ReasonSuperseded.
	cc1, err := Store(StoreInput{
		Kind:        KindConfigChange,
		Scope:       ScopeProject,
		ProjectPath: "/proj/a",
		Target:      "cache.ttl",
		BeforeValue: "5m",
		AfterValue:  "15m",
	})
	require.NoError(t, err)
	cc2, err := Store(StoreInput{
		Kind:        KindConfigChange,
		Scope:       ScopeProject,
		ProjectPath: "/proj/a",
		Target:      "cache.ttl",
		BeforeValue: "15m",
		AfterValue:  "30m",
	})
	require.NoError(t, err)
	assert.Equal(t, []string{cc1.Ref}, cc2.InvalidatedRefs)

	eCC1, err := entryByRef(cc1.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, eCC1.ValidUntil)
	assert.Equal(t, ReasonSuperseded, eCC1.InvalidatedReason)
	assert.Equal(t, cc2.Ref, eCC1.SupersededBy)

	// History follows the config_change chain.
	hist, err := History(HistoryInput{Ref: cc1.Ref})
	require.NoError(t, err)
	require.Len(t, hist.Entries, 2)
	assert.Equal(t, cc1.Ref, hist.Entries[0].Ref)
	assert.Equal(t, cc2.Ref, hist.Entries[1].Ref)

	// Events and Discoveries never auto-supersede even with InvalidatePrevious: true.
	ev1, err := Store(StoreInput{Kind: KindEvent, Title: "Alert 1", InvalidatePrevious: true})
	require.NoError(t, err)
	ev2, err := Store(StoreInput{Kind: KindEvent, Title: "Alert 1", InvalidatePrevious: true})
	require.NoError(t, err)
	assert.Empty(t, ev2.InvalidatedRefs)
	eEv1, err := entryByRef(ev1.Ref)
	require.NoError(t, err)
	assert.Empty(t, eEv1.ValidUntil)

	d1, err := Store(StoreInput{Kind: KindDiscovery, Title: "Finding", InvalidatePrevious: true})
	require.NoError(t, err)
	d2, err := Store(StoreInput{Kind: KindDiscovery, Title: "Finding", InvalidatePrevious: true})
	require.NoError(t, err)
	assert.Empty(t, d2.InvalidatedRefs)
	eD1, err := entryByRef(d1.Ref)
	require.NoError(t, err)
	assert.Empty(t, eD1.ValidUntil)
}

func TestKBSecretRedactionOnStoreAndUpdate(t *testing.T) {
	dbtest.Init(t)

	rawToken := "ghp_1234567890abcdefghij1234567890"
	res, err := Store(StoreInput{
		Kind:       KindConfigChange,
		Target:     "github.token",
		AfterValue: "Bearer " + rawToken,
		Body:       "Configured secret token " + rawToken,
	})
	require.NoError(t, err)
	assert.NotEmpty(t, res.Warnings)
	assert.Contains(t, res.RedactedFields, "after_value")
	assert.Contains(t, res.RedactedFields, "body")

	e, err := entryByRef(res.Ref)
	require.NoError(t, err)
	assert.NotContains(t, e.AfterValue, rawToken)
	assert.NotContains(t, e.Body, rawToken)

	// FTS must not contain raw secret either.
	var ftsCnt int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory_kb_fts WHERE structured_memory_kb_fts MATCH ?`, rawToken).Scan(&ftsCnt))
	assert.Zero(t, ftsCnt)

	// UpdateEntry also redacts secrets.
	newSecret := "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
	updated, warnings, err := UpdateEntry(UpdateInput{
		Ref:    res.Ref,
		Reason: &newSecret,
	}, nil)
	require.NoError(t, err)
	assert.NotEmpty(t, warnings)
	assert.NotContains(t, updated.Reason, newSecret)
	assert.Contains(t, updated.RedactedFields, "reason")
}

func TestKBTwoTierRecallAndTruncation(t *testing.T) {
	dbtest.Init(t)

	res, err := Store(StoreInput{
		Kind:        KindConfigChange,
		Target:      "api.timeout",
		BeforeValue: "5s",
		AfterValue:  "30s",
		Reason:      "upstream latency spikes",
		Body:        strings.Repeat("word ", 600) + "日本語テスト",
		Tags:        []string{"perf"},
		Actor:       "alice",
	})
	require.NoError(t, err)

	// Tier-1: Expand=false, Refs=nil omits Tier-2 fields.
	t1, err := Recall(RecallInput{Query: "upstream", Expand: false}, nil)
	require.NoError(t, err)
	require.Len(t, t1.Entries, 1)
	assert.Empty(t, t1.Entries[0].Body)
	assert.Empty(t, t1.Entries[0].BeforeValue)
	assert.Empty(t, t1.Entries[0].AfterValue)
	assert.Empty(t, t1.Entries[0].Reason)
	require.Len(t, t1.Lines, 1)
	assert.Equal(t, "alice", t1.Lines[0].Actor)
	assert.Equal(t, []string{"perf"}, t1.Lines[0].Tags)

	// Tier-2 via Refs returns full fields and honors raised 8000 cap when v2 enabled.
	t2Refs, err := Recall(RecallInput{Refs: []string{res.Ref}}, nil)
	require.NoError(t, err)
	require.Len(t, t2Refs.Entries, 1)
	assert.Contains(t, t2Refs.Entries[0].Body, "日本語テスト")
	assert.Equal(t, "5s", t2Refs.Entries[0].BeforeValue)
	assert.Equal(t, "30s", t2Refs.Entries[0].AfterValue)
	assert.Equal(t, "upstream latency spikes", t2Refs.Entries[0].Reason)

	// Single oversized entry truncated at valid UTF-8 boundary when budget is tight.
	prevV2 := flags.Enabled(flags.KeyMemoryV2)
	require.NoError(t, flags.Set(flags.KeyMemoryV2, true))
	t.Cleanup(func() { _ = flags.Set(flags.KeyMemoryV2, prevV2) })
	bigBody := strings.Repeat("α", 1500)
	bigRes, err := Store(StoreInput{
		Kind:  KindDiscovery,
		Title: "Unicode truncation check",
		Body:  bigBody,
	})
	require.NoError(t, err)

	trunc, err := Recall(RecallInput{Refs: []string{bigRes.Ref}, TokenBudget: 80}, nil)
	require.NoError(t, err)
	require.Len(t, trunc.Entries, 1)
	assert.True(t, utf8.ValidString(trunc.Entries[0].Body))
	assert.Contains(t, trunc.Entries[0].Body, "truncated; request smaller budget or view in Dashboard")
	assert.LessOrEqual(t, trunc.TokensUsed, 80)

	// Tier-2 raises the v2 TokenBudget cap from 2000 to 8000 (so a ~3500-token entry fits under Refs/Expand).
	largeRes, err := Store(StoreInput{
		Kind:  KindDiscovery,
		Title: "Large tier-2 cap test",
		Body:  strings.Repeat("abcd ", 2800), // ~3500 tokens, < 16 KB
	})
	require.NoError(t, err)
	fullCap, err := Recall(RecallInput{Refs: []string{largeRes.Ref}, TokenBudget: 9000}, nil)
	require.NoError(t, err)
	require.Len(t, fullCap.Entries, 1)
	assert.NotContains(t, fullCap.Entries[0].Body, "truncated; request smaller budget", "Tier-2 8000 cap should allow ~3500-token body without truncating")
}

func TestKBRecallFiltersAndQuarantine(t *testing.T) {
	dbtest.Init(t)

	ev1, err := Store(StoreInput{
		Kind:       KindEvent,
		Title:      "Old deploy",
		Tags:       []string{"go", "deploy"},
		Actor:      "alice",
		OccurredAt: "2026-01-10T10:00:00Z",
	})
	require.NoError(t, err)
	ev2, err := Store(StoreInput{
		Kind:       KindEvent,
		Title:      "Recent golang incident",
		Tags:       []string{"golang", "incident"},
		Actor:      "bob",
		OccurredAt: "2026-03-10T10:00:00Z",
		Verified:   true,
	})
	require.NoError(t, err)

	// Quarantine ev2.
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET quarantined = 1 WHERE ref = ?`, ev2.Ref)
	require.NoError(t, err)

	// Default recall excludes quarantined.
	res, err := Recall(RecallInput{}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{ev1.Ref}, entryRefs(res.Entries))

	// IncludeQuarantined returns both.
	resQ, err := Recall(RecallInput{IncludeQuarantined: true}, nil)
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{ev1.Ref, ev2.Ref}, entryRefs(resQ.Entries))

	// ApproveQuarantined clears quarantine and reindexes.
	require.NoError(t, ApproveQuarantined(ev2.Ref, nil))

	// Tags filter must not match "golang" when filtering for "go".
	resGo, err := Recall(RecallInput{Tags: []string{"go"}}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{ev1.Ref}, entryRefs(resGo.Entries))

	// Actor & VerifiedOnly filter.
	resBob, err := Recall(RecallInput{Actor: "bob", VerifiedOnly: true}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{ev2.Ref}, entryRefs(resBob.Entries))

	// Since/Until time range filter uses occurred_at.
	resSince, err := Recall(RecallInput{Since: "2026-02-01T00:00:00Z"}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{ev2.Ref}, entryRefs(resSince.Entries))

	resUntil, err := Recall(RecallInput{Until: "2026-02-01T00:00:00Z"}, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{ev1.Ref}, entryRefs(resUntil.Entries))

	// IncludeSuperseded returns superseded entries while still excluding user-forgotten entries.
	cc1, err := Store(StoreInput{Kind: KindConfigChange, Target: "log.level", AfterValue: "info"})
	require.NoError(t, err)
	cc2, err := Store(StoreInput{Kind: KindConfigChange, Target: "log.level", AfterValue: "debug"})
	require.NoError(t, err)
	forgotten, err := Store(StoreInput{Kind: KindDiscovery, Title: "Forgotten finding"})
	require.NoError(t, err)
	_, err = Forget(ForgetInput{Refs: []string{forgotten.Ref}})
	require.NoError(t, err)

	resSup, err := Recall(RecallInput{Kinds: []Kind{KindConfigChange, KindDiscovery}, IncludeSuperseded: true}, nil)
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{cc1.Ref, cc2.Ref}, entryRefs(resSup.Entries))
}

func TestKBDecayExemptionsVerifiedBoostAndStaleAnchor(t *testing.T) {
	dbtest.Init(t)

	old := time.Now().UTC().Add(-180 * 24 * time.Hour).Format("2006-01-02 15:04:05")
	for _, k := range []Kind{KindPreference, KindEvent, KindConfigChange, KindDiscovery} {
		e := Entry{Kind: k, Scope: ScopeSession, AccessCount: 1, LastAccessedAt: old, CreatedAt: old}
		assert.Equal(t, 1.0, decayFactor(e, RecallInput{}), "kind %s should be exempt from decay", k)

		e.Verified = true
		assert.InDelta(t, 1.25, decayFactor(e, RecallInput{}), 1e-9, "verified %s should get 1.25x", k)
	}

	// Verified cap at 1.75 even when base decay factor hits 1.5.
	now := time.Now().UTC().Format("2006-01-02 15:04:05")
	highFact := Entry{
		Kind:           KindFact,
		Scope:          ScopeSession,
		AccessCount:    1000,
		LastAccessedAt: now,
		CreatedAt:      now,
		Verified:       true,
	}
	assert.InDelta(t, 1.75, decayFactor(highFact, RecallInput{}), 1e-9)

	// RunStaleAnchorJob skips verified facts.
	past := time.Now().Add(-2 * time.Hour).Format(time.RFC3339)
	unverified, err := Store(StoreInput{
		Kind:           KindFact,
		Scope:          ScopeGlobal,
		Subject:        "host.ip",
		Object:         "10.0.0.1",
		WorldInvalidAt: past,
	})
	require.NoError(t, err)
	verified, err := Store(StoreInput{
		Kind:           KindFact,
		Scope:          ScopeGlobal,
		Subject:        "host.dns",
		Object:         "10.0.0.2",
		WorldInvalidAt: past,
		Verified:       true,
	})
	require.NoError(t, err)

	RunStaleAnchorJob()

	eUnv, err := entryByRef(unverified.Ref)
	require.NoError(t, err)
	assert.Equal(t, ReasonStaleAnchor, eUnv.InvalidatedReason)

	eVer, err := entryByRef(verified.Ref)
	require.NoError(t, err)
	assert.Empty(t, eVer.ValidUntil, "verified fact must not be invalidated by stale anchor job")
}

func TestKBCurationUpdateSetVerifiedQueryAndExport(t *testing.T) {
	dbtest.Init(t)
	search.Cache.Unload()
	t.Cleanup(search.Cache.Unload)

	// 1. In-place update on discovery.
	disc, err := Store(StoreInput{
		Kind:        KindDiscovery,
		Scope:       ScopeProject,
		ProjectPath: "/workspace/sauger",
		Title:       "Initial title",
		Body:        "Original notes",
		Tags:        []string{"alpha"},
	})
	require.NoError(t, err)

	newTitle := "Refined title"
	newTags := []string{"Beta", "alpha"}
	updated, _, err := UpdateEntry(UpdateInput{
		Ref:   disc.Ref,
		Title: &newTitle,
		Tags:  &newTags,
	}, nil)
	require.NoError(t, err)
	assert.Equal(t, disc.Ref, updated.Ref)
	assert.Equal(t, "Refined title", updated.Title)
	assert.Equal(t, []string{"beta", "alpha"}, updated.Tags)

	// 2. Substantive edit on preference creates new version and supersedes old.
	pref, err := Store(StoreInput{
		Kind:      KindPreference,
		Scope:     ScopeGlobal,
		Subject:   "user",
		Predicate: "prefers",
		Object:    "json",
	})
	require.NoError(t, err)

	newObj := "yaml"
	v2, _, err := UpdateEntry(UpdateInput{
		Ref:    pref.Ref,
		Object: &newObj,
	}, nil)
	require.NoError(t, err)
	assert.NotEqual(t, pref.Ref, v2.Ref)
	assert.Equal(t, "yaml", v2.Object)

	oldPref, err := entryByRef(pref.Ref)
	require.NoError(t, err)
	assert.NotEmpty(t, oldPref.ValidUntil)
	assert.Equal(t, ReasonSuperseded, oldPref.InvalidatedReason)
	assert.Equal(t, v2.Ref, oldPref.SupersededBy)

	// 3. SetVerified toggle.
	require.NoError(t, SetVerified(disc.Ref, true))
	eDisc, err := entryByRef(disc.Ref)
	require.NoError(t, err)
	assert.True(t, eDisc.Verified)
	assert.NotEmpty(t, eDisc.VerifiedAt)

	require.NoError(t, SetVerified(disc.Ref, false))
	eDisc, err = entryByRef(disc.Ref)
	require.NoError(t, err)
	assert.False(t, eDisc.Verified)
	assert.Empty(t, eDisc.VerifiedAt)

	err = SetVerified("mem_missing", true)
	assert.True(t, errs.HasCode(err, errs.CodeNotFound))

	// 4. QueryKB with project + global + WTG sibling + quarantine + pagination + search.
	_, err = db.DB.Exec(`INSERT OR IGNORE INTO project_links (parent_path, child_path, auto_linked) VALUES (?, ?, 0)`,
		"/workspace/sauger", "/workspace/sauger-child")
	require.NoError(t, err)

	_, err = Store(StoreInput{
		Kind:        KindEvent,
		Scope:       ScopeProject,
		ProjectPath: "/workspace/sauger-child",
		Title:       "Child repo migration",
		OccurredAt:  "2026-04-01T12:00:00Z",
	})
	require.NoError(t, err)

	qRes, err := QueryKB(KBQueryInput{
		ProjectPath:              "/workspace/sauger",
		IncludeGlobalWithProject: true,
		Sort:                     "occurred_desc",
		Page:                     1,
		PerPage:                  2,
	})
	require.NoError(t, err)
	assert.Equal(t, 3, qRes.Total) // disc + v2 (global) + child event
	assert.Equal(t, 2, qRes.TotalPages)
	assert.Len(t, qRes.Entries, 2)
	assert.Equal(t, 3, qRes.Counts.Total)
	assert.Equal(t, 1, qRes.Counts.Discovery)
	assert.Equal(t, 1, qRes.Counts.Preference)
	assert.Equal(t, 1, qRes.Counts.Event)

	// Search with Query + embedder.
	emb := stubEmbedder{"Child repo migration": unitVector(0.95), "migration": unitVector(1.0)}
	sRes, err := QueryKB(KBQueryInput{
		Query:                    "migration",
		ProjectPath:              "/workspace/sauger",
		IncludeGlobalWithProject: true,
		Embedder:                 emb,
	})
	require.NoError(t, err)
	require.Equal(t, 1, sRes.Total)
	assert.Equal(t, "Child repo migration", sRes.Entries[0].Title)

	// ExportEntries markdown and json.
	mdBytes, err := ExportEntries(KBQueryInput{ProjectPath: "/workspace/sauger", IncludeGlobalWithProject: true}, "markdown")
	require.NoError(t, err)
	assert.Contains(t, string(mdBytes), "# Knowledge Base Export")
	assert.Contains(t, string(mdBytes), "Child repo migration")

	jsonBytes, err := ExportEntries(KBQueryInput{ProjectPath: "/workspace/sauger", IncludeGlobalWithProject: true}, "json")
	require.NoError(t, err)
	var exported []Entry
	require.NoError(t, json.Unmarshal(jsonBytes, &exported))
	assert.Len(t, exported, 3)

	_, err = ExportEntries(KBQueryInput{}, "xml")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

func TestKBStoreExtractedAllKindsAndConflictDiscovery(t *testing.T) {
	dbtest.Init(t)

	note := `
FACT: db.engine | is | sqlite
RULE: Always run gofumpt before commit
PREFERENCE: user | prefers | concise diffs
EVENT: Deploy v3 | error | release | Outage in staging
CONFIG_CHANGE: cache.size | 64MB | 128MB | reduce eviction
DISCOVERY: FTS5 column filters | Ignore ref column during BM25
`
	ex := ExtractFromText(note)
	stored, err := StoreExtracted("s-all", "/proj/x", "ctx_abc", ex, ScopeProject)
	require.NoError(t, err)
	assert.Len(t, stored, 6)

	// ConflictCandidates supports KindDiscovery and skips quarantined.
	cands, method := ConflictCandidates(Entry{
		Kind:        KindDiscovery,
		Scope:       ScopeProject,
		ProjectPath: "/proj/x",
		Title:       "FTS5 column filters tuning",
		Ref:         "mem_probe",
	}, nil, 0)
	assert.Equal(t, "fts", method)
	require.NotEmpty(t, cands)
	assert.Contains(t, cands[0].Line, "FTS5 column filters")
}

func TestKBRestoreCompletesWithoutSQLiteLockWait(t *testing.T) {
	dbtest.Init(t)

	a, err := Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "lock.test",
		Predicate:          "is",
		Object:             "v1",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)
	_, err = Store(StoreInput{
		Kind:               KindFact,
		Scope:              ScopeGlobal,
		Subject:            "lock.test",
		Predicate:          "is",
		Object:             "v2",
		InvalidatePrevious: true,
	})
	require.NoError(t, err)

	start := time.Now()
	res, err := Restore(RestoreInput{Refs: []string{a.Ref}})
	elapsed := time.Since(start)
	require.NoError(t, err)
	assert.Equal(t, []string{a.Ref}, res.RestoredRefs)
	assert.Less(t, elapsed, 2*time.Second, "Restore must not stall on SQLite busy_timeout")
}
