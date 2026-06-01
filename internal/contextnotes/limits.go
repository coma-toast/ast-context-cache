package contextnotes

import (
	"os"
	"strconv"
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
		MaxNotesSession:  defaultMaxNotesSession,
		MaxTokensSession: defaultMaxTokensSession,
		MaxNotesGlobal:   defaultMaxNotesGlobal,
		MaxTokensGlobal:  defaultMaxTokensGlobal,
		Policy:           defaultLimitPolicy,
	}
	if v := envOrSetting("AST_CONTEXT_MAX_NOTES_SESSION", "context_max_notes_session"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			l.MaxNotesSession = n
		}
	}
	if v := envOrSetting("AST_CONTEXT_MAX_TOKENS_SESSION", "context_max_tokens_session"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			l.MaxTokensSession = n
		}
	}
	if v := envOrSetting("AST_CONTEXT_MAX_NOTES_GLOBAL", "context_max_notes_global"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			l.MaxNotesGlobal = n
		}
	}
	if v := envOrSetting("AST_CONTEXT_MAX_TOKENS_GLOBAL", "context_max_tokens_global"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			l.MaxTokensGlobal = n
		}
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
