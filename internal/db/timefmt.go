package db

import (
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// sqlTimeInputLayouts are the timestamp shapes NormalizeSQLTime accepts. RFC3339 parsing
// also accepts fractional seconds, so it covers RFC3339Nano.
var sqlTimeInputLayouts = []string{time.RFC3339, time.DateTime, time.DateOnly}

// SQLTime formats t the way SQLite's datetime('now') does (UTC "YYYY-MM-DD HH:MM:SS"), so
// stored timestamps compare correctly as text against each other and SQLite's clock.
func SQLTime(t time.Time) string {
	return t.UTC().Format(time.DateTime)
}

// NormalizeSQLTime parses an RFC3339, "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD" timestamp and
// returns it in SQLTime form.
func NormalizeSQLTime(s string) (string, error) {
	for _, layout := range sqlTimeInputLayouts {
		if t, err := time.Parse(layout, s); err == nil {
			return SQLTime(t), nil
		}
	}
	return "", errs.NewCode(errs.CodeInvalidInput, "invalid timestamp: want RFC3339, YYYY-MM-DD HH:MM:SS or YYYY-MM-DD", "value", s)
}
