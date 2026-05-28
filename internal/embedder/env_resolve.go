package embedder

import (
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// EffectiveEnv returns a non-empty os.Getenv(k), otherwise the trimmed value from
// the settings table (same key name as the env var). Process environment always wins
// when set to a non-empty string so CI and one-off runs stay predictable.
func EffectiveEnv(k string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return strings.TrimSpace(db.GetSetting(k, ""))
}
