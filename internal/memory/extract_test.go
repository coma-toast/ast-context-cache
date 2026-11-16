package memory

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestExtractFromTextAllMarkers(t *testing.T) {
	t.Parallel()

	note := `# Release & KB Notes

Regular prose line should be ignored.

` + "```" + `
FACT: inside | code | fence
PREFERENCE: inside fence
EVENT: [error] inside fence
CONFIG_CHANGE: k | a -> b | fence
DISCOVERY: inside fence
` + "```" + `

~~~
RULE: inside tilde fence
~~~

FACT: db.engine | uses | SQLite WAL
- rule: Always run gofumpt before commit
* preference: editor | uses | neovim
• Preference: jason prefers compact diffs
+ PREFERENCE: dark-mode
EVENT: [ERROR] Pool exhausted | Connections hit 100% on primary
- event: Deploy v2.1 | release | Shipped zero-downtime migration
EVENT: [warning] Disk watermark crossed | infra | 85% full on /data
* event: Service restarted
CONFIG_CHANGE: max_connections | 100 -> 200 | traffic spike
- config_change: log_level | debug | investigating timeout
DISCOVERY: FTS5 supports column filters | Prefix queries with {col1 col2} : query
* discovery: Single-line discovery title
`

	ex := ExtractFromText(note)
	assert.Empty(t, ex.Skipped)

	require.Len(t, ex.Facts, 1)
	assert.Equal(t, FactInput{Subject: "db.engine", Predicate: "uses", Object: "SQLite WAL"}, ex.Facts[0])

	require.Len(t, ex.Procedures, 1)
	assert.Equal(t, ProcedureInput{Rule: "Always run gofumpt before commit"}, ex.Procedures[0])

	require.Len(t, ex.Preferences, 3)
	assert.Equal(t, PreferenceInput{Subject: "editor", Predicate: "uses", Object: "neovim"}, ex.Preferences[0])
	assert.Equal(t, PreferenceInput{Subject: "jason", Predicate: "prefers", Object: "compact diffs"}, ex.Preferences[1])
	assert.Equal(t, PreferenceInput{Subject: "user", Predicate: "preference", Object: "dark-mode"}, ex.Preferences[2])

	require.Len(t, ex.Events, 4)
	assert.Equal(t, EventInput{Title: "Pool exhausted", Body: "Connections hit 100% on primary", Severity: SeverityError}, ex.Events[0])
	assert.Equal(t, EventInput{Title: "Deploy v2.1", Category: "release", Body: "Shipped zero-downtime migration"}, ex.Events[1])
	assert.Equal(t, EventInput{Title: "Disk watermark crossed", Category: "infra", Body: "85% full on /data", Severity: SeverityWarning}, ex.Events[2])
	assert.Equal(t, EventInput{Title: "Service restarted"}, ex.Events[3])

	require.Len(t, ex.ConfigChanges, 2)
	assert.Equal(t, ConfigChangeInput{Target: "max_connections", BeforeValue: "100", AfterValue: "200", Reason: "traffic spike"}, ex.ConfigChanges[0])
	assert.Equal(t, ConfigChangeInput{Target: "log_level", AfterValue: "debug", Reason: "investigating timeout"}, ex.ConfigChanges[1])

	require.Len(t, ex.Discoveries, 2)
	assert.Equal(t, DiscoveryInput{Title: "FTS5 supports column filters", Body: "Prefix queries with {col1 col2} : query"}, ex.Discoveries[0])
	assert.Equal(t, DiscoveryInput{Title: "Single-line discovery title"}, ex.Discoveries[1])
}

func TestExtractFromTextInvalidAndEmptySkipped(t *testing.T) {
	t.Parallel()

	note := `
FACT: too-short
RULE:
PREFERENCE:
PREFERENCE: | |
EVENT:
EVENT: [critical]
EVENT: | missing title
CONFIG_CHANGE:
CONFIG_CHANGE: missing_pipe
CONFIG_CHANGE: target | before ->
DISCOVERY:
DISCOVERY: | missing title
`
	ex := ExtractFromText(note)
	assert.Empty(t, ex.Facts)
	assert.Empty(t, ex.Procedures)
	assert.Empty(t, ex.Preferences)
	assert.Empty(t, ex.Events)
	assert.Empty(t, ex.ConfigChanges)
	assert.Empty(t, ex.Discoveries)
	assert.Len(t, ex.Skipped, 12)
}

func TestTypesValidationAndTags(t *testing.T) {
	t.Parallel()

	for _, k := range []Kind{KindFact, KindProcedure, KindPreference, KindEvent, KindConfigChange, KindDiscovery} {
		assert.True(t, IsValidKind(k), "expected valid kind %s", k)
	}
	assert.False(t, IsValidKind(KindPinned))
	assert.False(t, IsValidKind(Kind("unknown")))

	for _, s := range []string{"", SeverityInfo, SeverityWarning, SeverityError, SeverityCritical} {
		assert.True(t, IsValidSeverity(s), "expected valid severity %q", s)
	}
	assert.False(t, IsValidSeverity("fatal"))

	encoded := EncodeTags([]string{" db ", "fts5", "db", ""})
	assert.Equal(t, `["db","fts5"]`, encoded)
	assert.Equal(t, []string{"db", "fts5"}, DecodeTags(encoded))
	assert.Equal(t, []string{"db", "fts5"}, DecodeTags(" db , fts5 , db, "))
	assert.Empty(t, EncodeTags(nil))
	assert.Empty(t, DecodeTags(""))
}

func TestFormatLineAndEmbedText(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "user.lang prefers Go", FormatLine(Entry{Kind: KindFact, Subject: "user.lang", Predicate: "prefers", Object: "Go"}))
	assert.Equal(t, "PROC: run tests", FormatLine(Entry{Kind: KindProcedure, Rule: "run tests"}))
	assert.Equal(t, "[preference] editor uses: neovim", FormatLine(Entry{Kind: KindPreference, Subject: "editor", Predicate: "uses", Object: "neovim"}))
	assert.Equal(t, "[preference] concise answers", FormatLine(Entry{Kind: KindPreference, Rule: "concise answers"}))
	assert.Equal(t, "[event:error] Outage (api-gw, 2026-11-15T10:00:00Z)", FormatLine(Entry{
		Kind:       KindEvent,
		Severity:   SeverityError,
		Title:      "Outage",
		Target:     "api-gw",
		OccurredAt: "2026-11-15T10:00:00Z",
	}))
	assert.Equal(t, "[event] Deployed", FormatLine(Entry{Kind: KindEvent, Title: "Deployed"}))
	assert.Equal(t, "[config_change] pool: 10 -> 50 (load)", FormatLine(Entry{
		Kind:        KindConfigChange,
		Target:      "pool",
		BeforeValue: "10",
		AfterValue:  "50",
		Reason:      "load",
	}))
	assert.Equal(t, "[config_change] pool: 50", FormatLine(Entry{
		Kind:       KindConfigChange,
		Target:     "pool",
		AfterValue: "50",
	}))
	assert.Equal(t, "[discovery] WAL speeds reads (sqlite)", FormatLine(Entry{
		Kind:     KindDiscovery,
		Title:    "WAL speeds reads",
		Category: "sqlite",
	}))

	// Multi-byte UTF-8 truncation at 1,200 bytes must never split a rune.
	longBody := strings.Repeat("こんにちは世界 ", 120)
	embed := EmbedText(Entry{
		Kind:     KindDiscovery,
		Title:    "Multibyte test",
		Category: "i18n",
		Tags:     []string{"utf8", "truncation"},
		Body:     longBody,
	})
	assert.LessOrEqual(t, len(embed), maxEmbedTextBytes)
	assert.True(t, utf8.ValidString(embed))
	assert.Contains(t, embed, "Multibyte test")
}

func TestDualFTSIndexAndDelete(t *testing.T) {
	dbtest.Init(t)

	e := Entry{
		Ref:         "mem_kb_test01",
		Kind:        KindConfigChange,
		Scope:       ScopeProject,
		ProjectPath: "/repo",
		Title:       "Bump pool size",
		Target:      "db.pool_max",
		BeforeValue: "10",
		AfterValue:  "64",
		Reason:      "concurrency",
		Category:    "database",
		Actor:       "deploy-bot",
		Tags:        []string{"sqlite", "tuning"},
		Body:        "Increased connection pool under heavy load",
	}
	require.NoError(t, indexEntryFTS(e))
	// Re-index with updated body must replace without leaving duplicate rows.
	e.Body = "Updated connection pool explanation"
	require.NoError(t, indexEntryFTS(e))

	var kbCount int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory_kb_fts WHERE ref = ?`, e.Ref).Scan(&kbCount))
	assert.Equal(t, 1, kbCount)

	var matchedRef string
	require.NoError(t, db.ContextDB.QueryRow(`SELECT ref FROM structured_memory_kb_fts WHERE structured_memory_kb_fts MATCH 'sqlite'`).Scan(&matchedRef))
	assert.Equal(t, e.Ref, matchedRef)

	deleteFTS(e.Ref)
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory_kb_fts WHERE ref = ?`, e.Ref).Scan(&kbCount))
	assert.Equal(t, 0, kbCount)
	var ftsCount int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory_fts WHERE ref = ?`, e.Ref).Scan(&ftsCount))
	assert.Equal(t, 0, ftsCount)
}
