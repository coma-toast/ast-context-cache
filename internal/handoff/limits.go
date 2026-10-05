package handoff

import (
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Settings keys for the handoff limits. Each can be overridden by the environment variable
// AST_HANDOFF_<suffix in upper snake case>, e.g. AST_HANDOFF_TTL_DAYS.
const (
	SettingTTLDays              = "handoff_ttl_days"
	SettingSummaryMaxTokens     = "handoff_summary_max_tokens"
	SettingChildInactiveMinutes = "handoff_child_inactive_minutes"
	SettingTreeMaxTokens        = "handoff_tree_max_tokens"
	SettingTreeMaxEntries       = "handoff_tree_max_entries"
	SettingMaxDepth             = "handoff_max_depth"
	SettingMaxChildren          = "handoff_max_children"
	SettingOpenBudgetTokens     = "handoff_open_budget_tokens"
)

const (
	defaultTTLDays              = 7
	defaultSummaryMaxTokens     = 300
	defaultChildInactiveMinutes = 30
	defaultTreeMaxTokens        = 64000
	defaultTreeMaxEntries       = 300
	defaultMaxDepth             = 3
	defaultMaxChildren          = 16
	defaultOpenBudgetTokens     = 1500
)

// Limits holds the handoff retention windows and caps (RQ-1, RQ-4, RT-2, FI-3, OP-3).
type Limits struct {
	TTLDays              int `json:"ttl_days"`
	SummaryMaxTokens     int `json:"summary_max_tokens"`
	ChildInactiveMinutes int `json:"child_inactive_minutes"`
	TreeMaxTokens        int `json:"tree_max_tokens"`
	TreeMaxEntries       int `json:"tree_max_entries"`
	MaxDepth             int `json:"max_depth"`
	MaxChildren          int `json:"max_children"`
	OpenBudgetTokens     int `json:"open_budget_tokens"`
}

// LimitSetting is one handoff limit's settings key and its default.
type LimitSetting struct {
	Key     string
	Default int
}

// limitSettings lists every limit setting, in Limits field order.
var limitSettings = []LimitSetting{
	{SettingTTLDays, defaultTTLDays},
	{SettingSummaryMaxTokens, defaultSummaryMaxTokens},
	{SettingChildInactiveMinutes, defaultChildInactiveMinutes},
	{SettingTreeMaxTokens, defaultTreeMaxTokens},
	{SettingTreeMaxEntries, defaultTreeMaxEntries},
	{SettingMaxDepth, defaultMaxDepth},
	{SettingMaxChildren, defaultMaxChildren},
	{SettingOpenBudgetTokens, defaultOpenBudgetTokens},
}

// LimitSettings returns every handoff limit's settings key and default, for the dashboard's
// settings defaults and validation.
func LimitSettings() []LimitSetting {
	return append([]LimitSetting(nil), limitSettings...)
}

// IsLimitSetting reports whether key is a handoff limit's settings key.
func IsLimitSetting(key string) bool {
	for _, ls := range limitSettings {
		if ls.Key == key {
			return true
		}
	}
	return false
}

// LoadLimits resolves every limit as env > setting > default. Limits are read per call, so a
// settings change applies to the next operation without a restart.
func LoadLimits() Limits {
	return Limits{
		TTLDays:              settingInt(SettingTTLDays, defaultTTLDays),
		SummaryMaxTokens:     settingInt(SettingSummaryMaxTokens, defaultSummaryMaxTokens),
		ChildInactiveMinutes: settingInt(SettingChildInactiveMinutes, defaultChildInactiveMinutes),
		TreeMaxTokens:        settingInt(SettingTreeMaxTokens, defaultTreeMaxTokens),
		TreeMaxEntries:       settingInt(SettingTreeMaxEntries, defaultTreeMaxEntries),
		MaxDepth:             settingInt(SettingMaxDepth, defaultMaxDepth),
		MaxChildren:          settingInt(SettingMaxChildren, defaultMaxChildren),
		OpenBudgetTokens:     settingInt(SettingOpenBudgetTokens, defaultOpenBudgetTokens),
	}
}

// TTL is how long a tree lives after its last access.
func (l Limits) TTL() time.Duration {
	return time.Duration(l.TTLDays) * 24 * time.Hour
}

// ChildInactive is how long a child may go without activity before it is abandoned.
func (l Limits) ChildInactive() time.Duration {
	return time.Duration(l.ChildInactiveMinutes) * time.Minute
}

// EnvKey returns the environment variable that overrides setting key.
func EnvKey(key string) string {
	return "AST_" + strings.ToUpper(key)
}

func settingInt(key string, def int) int {
	return db.SettingInt(key, EnvKey(key), def)
}
