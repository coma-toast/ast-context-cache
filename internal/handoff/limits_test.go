package handoff

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestLoadLimitsDefaults(t *testing.T) {
	dbtest.Init(t)
	assert.Equal(t, Limits{
		TTLDays: 7, SummaryMaxTokens: 300, ChildInactiveMinutes: 30, TreeMaxTokens: 64000,
		TreeMaxEntries: 300, MaxDepth: 3, MaxChildren: 16, OpenBudgetTokens: 1500,
	}, LoadLimits())
	assert.Equal(t, 7*24*time.Hour, LoadLimits().TTL())
	assert.Equal(t, 30*time.Minute, LoadLimits().ChildInactive())
}

func TestLoadLimitsResolution(t *testing.T) {
	dbtest.Init(t)
	tests := []struct {
		name, setting, env string
		want               int
	}{
		{name: "setting", setting: "5", want: 5},
		{name: "env beats setting", setting: "5", env: "9", want: 9},
		{name: "invalid env uses default", setting: "5", env: "soon", want: 16},
		{name: "non-positive setting uses default", setting: "0", want: 16},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.NoError(t, db.SetSetting(SettingMaxChildren, tt.setting))
			t.Setenv("AST_HANDOFF_MAX_CHILDREN", tt.env)
			assert.Equal(t, tt.want, LoadLimits().MaxChildren)
		})
	}
	assert.Equal(t, "AST_HANDOFF_TTL_DAYS", EnvKey(SettingTTLDays))
}

func TestLimitSettings(t *testing.T) {
	dbtest.Init(t)
	l := LoadLimits()
	want := map[string]int{
		SettingTTLDays: l.TTLDays, SettingSummaryMaxTokens: l.SummaryMaxTokens, SettingChildInactiveMinutes: l.ChildInactiveMinutes,
		SettingTreeMaxTokens: l.TreeMaxTokens, SettingTreeMaxEntries: l.TreeMaxEntries, SettingMaxDepth: l.MaxDepth,
		SettingMaxChildren: l.MaxChildren, SettingOpenBudgetTokens: l.OpenBudgetTokens,
	}
	got := map[string]int{}
	for _, ls := range LimitSettings() {
		got[ls.Key] = ls.Default
		assert.True(t, IsLimitSetting(ls.Key), ls.Key)
	}
	assert.Equal(t, want, got, "every limit is listed with the default LoadLimits resolves to")
	assert.False(t, IsLimitSetting("handoff_unknown"))
}
