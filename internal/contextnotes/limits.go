package contextnotes

import (
	"os"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	defaultMaxNotesSession  = 50
	defaultMaxTokensSession = 32000
	defaultMaxNotesGlobal   = 500
	defaultMaxTokensGlobal  = 200000
	defaultLimitPolicy      = "reject"
	// Offload notes (self-offloaded tool results) have their own global budget and TTL.
	defaultMaxOffloadTokensGlobal = 500000
	defaultOffloadTTLHours        = 24
)

// Limits holds virtual context storage caps.
type Limits struct {
	MaxNotesSession  int
	MaxTokensSession int
	MaxNotesGlobal   int
	MaxTokensGlobal  int
	Policy           string // reject | lru_session
	// MaxOffloadTokensGlobal caps offload notes, which the counts above exclude.
	MaxOffloadTokensGlobal int
}

func LoadLimits() Limits {
	l := Limits{
		MaxNotesSession:  db.SettingInt("context_max_notes_session", "AST_CONTEXT_MAX_NOTES_SESSION", defaultMaxNotesSession),
		MaxTokensSession: db.SettingInt("context_max_tokens_session", "AST_CONTEXT_MAX_TOKENS_SESSION", defaultMaxTokensSession),
		MaxNotesGlobal:   db.SettingInt("context_max_notes_global", "AST_CONTEXT_MAX_NOTES_GLOBAL", defaultMaxNotesGlobal),
		MaxTokensGlobal:  db.SettingInt("context_max_tokens_global", "AST_CONTEXT_MAX_TOKENS_GLOBAL", defaultMaxTokensGlobal),
		Policy:           defaultLimitPolicy,
		MaxOffloadTokensGlobal: db.SettingInt("context_offload_max_tokens_global", "AST_CONTEXT_OFFLOAD_MAX_TOKENS_GLOBAL",
			defaultMaxOffloadTokensGlobal),
	}
	if v := envOrSetting("AST_CONTEXT_LIMIT_POLICY", "context_limit_policy"); v != "" {
		p := strings.ToLower(strings.TrimSpace(v))
		if p == "lru_session" || p == "reject" {
			l.Policy = p
		}
	}
	return l
}

func envOrSetting(envKey, settingKey string) string {
	if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
		return v
	}
	return strings.TrimSpace(db.GetSetting(settingKey, ""))
}

func (l Limits) AsMap() map[string]interface{} {
	return map[string]interface{}{
		"max_notes_session":         l.MaxNotesSession,
		"max_tokens_session":        l.MaxTokensSession,
		"max_notes_global":          l.MaxNotesGlobal,
		"max_tokens_global":         l.MaxTokensGlobal,
		"policy":                    l.Policy,
		"max_offload_tokens_global": l.MaxOffloadTokensGlobal,
	}
}

// OffloadTTL is how long an offload note lives after its last access (setting offload_ttl_hours).
func OffloadTTL() time.Duration {
	return time.Duration(db.SettingInt("offload_ttl_hours", "AST_OFFLOAD_TTL_HOURS", defaultOffloadTTLHours)) * time.Hour
}
