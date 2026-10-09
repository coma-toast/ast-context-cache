package db

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func TestSQLTimeFormatsUTC(t *testing.T) {
	ts := time.Date(2026, 10, 9, 15, 4, 5, 0, time.FixedZone("CDT", -5*3600))
	assert.Equal(t, "2026-10-09 20:04:05", SQLTime(ts))
}

func TestNormalizeSQLTime(t *testing.T) {
	cases := map[string]string{
		"2026-10-09T15:04:05Z":                "2026-10-09 15:04:05",
		"2026-10-09T15:04:05-05:00":           "2026-10-09 20:04:05",
		"2026-10-09T15:04:05.123456789+02:00": "2026-10-09 13:04:05",
		"2026-10-09 15:04:05":                 "2026-10-09 15:04:05",
		"2026-10-09":                          "2026-10-09 00:00:00",
	}
	for in, want := range cases {
		got, err := NormalizeSQLTime(in)
		require.NoError(t, err, in)
		assert.Equal(t, want, got, in)
	}
}

func TestNormalizeSQLTimeRejectsGarbage(t *testing.T) {
	for _, in := range []string{"", "yesterday", "2026/10/09", "10-09-2026"} {
		_, err := NormalizeSQLTime(in)
		require.Error(t, err, in)
		assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), in)
	}
}
