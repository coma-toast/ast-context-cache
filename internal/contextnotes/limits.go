package contextnotes

import (
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	defaultMaxNotesSession  = 50
	defaultMaxTokensSession = 32000
	defaultMaxNotesGlobal   = 500
	defaultMaxTokensGlobal  = 200000
	defaultLimitPolicy      = "reject"
)

// Limits holds virtual context storage caps.
type Limits struct {
	MaxNotesSession  int
	MaxTokensSession int
	MaxNotesGlobal   int
	MaxTokensGlobal  int
	Policy           string // reject | lru_session
}

func LoadLimits() Limits {
	l := Limits{
		MaxNotesSession:  db.SettingInt("context_max_notes_session", "AST_CONTEXT_MAX_NOTES_SESSION", defaultMaxNotesSession),
		MaxTokensSession: db.SettingInt("context_max_tokens_session", "AST_CONTEXT_MAX_TOKENS_SESSION", defaultMaxTokensSession),
		MaxNotesGlobal:   db.SettingInt("context_max_notes_global", "AST_CONTEXT_MAX_NOTES_GLOBAL", defaultMaxNotesGlobal),
		MaxTokensGlobal:  db.SettingInt("context_max_tokens_global", "AST_CONTEXT_MAX_TOKENS_GLOBAL", defaultMaxTokensGlobal),
		Policy:           defaultLimitPolicy,
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
		"max_notes_session":  l.MaxNotesSession,
		"max_tokens_session": l.MaxTokensSession,
		"max_notes_global":   l.MaxNotesGlobal,
		"max_tokens_global":  l.MaxTokensGlobal,
		"policy":             l.Policy,
	}
}
