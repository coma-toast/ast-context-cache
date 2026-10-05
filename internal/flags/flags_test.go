package flags

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// No t.Parallel: the db pools, the snapshot, and the subscriber list are package globals.

// setup clears every flag env var (one left set in the developer's shell would lock a flag),
// opens a fresh database, and resets the snapshot and subscribers before and after the test.
func setup(t *testing.T) {
	t.Helper()
	for _, f := range registry {
		t.Setenv(f.Env, "")
	}
	dbtest.Init(t)
	reset()
	t.Cleanup(reset)
}

func reset() {
	mu.Lock()
	defer mu.Unlock()
	current.Store(nil)
	subscribers = nil
}

func stateOf(t *testing.T, key string) FlagState {
	t.Helper()
	for _, st := range State() {
		if st.Key == key {
			return st
		}
	}
	t.Fatalf("no state for %s", key)
	return FlagState{}
}

func TestDefaults(t *testing.T) {
	setup(t)
	states := State()
	require.Len(t, states, len(registry))
	for i, f := range registry {
		assert.Equal(t, f.Key, states[i].Key, "registry order")
		assert.Equal(t, f.Default, Enabled(f.Key), f.Key)
		assert.Equal(t, SourceDefault, states[i].Source, f.Key)
		assert.False(t, states[i].Locked, f.Key)
		assert.Equal(t, f.Env, states[i].Env, f.Key)
		assert.NotEmpty(t, states[i].Description, f.Key)
	}
	assert.False(t, Enabled(KeyHandoffHooks))
	assert.True(t, ToolEnabled("scratchpad"))
	assert.True(t, ActionEnabled("scratchpad", "claim"))
}

func TestResolution(t *testing.T) {
	tests := []struct {
		name       string
		setting    string
		env        string
		want       bool
		wantSource string
		wantLocked bool
	}{
		{name: "default", want: false, wantSource: SourceDefault},
		{name: "setting overrides default", setting: "true", want: true, wantSource: SourceSetting},
		{name: "invalid setting falls back to default", setting: "maybe", want: false, wantSource: SourceDefault},
		{name: "env overrides setting and locks", setting: "true", env: "false", want: false, wantSource: SourceEnv, wantLocked: true},
		{name: "env on locks", env: "1", want: true, wantSource: SourceEnv, wantLocked: true},
		{name: "invalid env is ignored", setting: "true", env: "sometimes", want: true, wantSource: SourceSetting},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setup(t)
			if tt.setting != "" {
				require.NoError(t, db.SetSetting(KeyHandoffHooks, tt.setting))
			}
			t.Setenv("AST_FEATURE_HANDOFF_HOOKS", tt.env)
			Reload()
			assert.Equal(t, tt.want, Enabled(KeyHandoffHooks))
			st := stateOf(t, KeyHandoffHooks)
			assert.Equal(t, tt.wantSource, st.Source)
			assert.Equal(t, tt.wantLocked, st.Locked)
			assert.Equal(t, tt.want, st.Enabled)
		})
	}
}

func TestSetPersistsAndApplies(t *testing.T) {
	setup(t)
	require.NoError(t, Set(KeySharedQueryCache, false))
	assert.False(t, Enabled(KeySharedQueryCache))
	assert.Equal(t, "false", db.GetSetting(KeySharedQueryCache, ""))
	assert.Equal(t, SourceSetting, stateOf(t, KeySharedQueryCache).Source)
	reset()
	assert.False(t, Enabled(KeySharedQueryCache), "value survives a rebuild from the database")
}

func TestSnapshotIsCached(t *testing.T) {
	setup(t)
	require.True(t, Enabled(KeySharedQueryCache))
	require.NoError(t, db.SetSetting(KeySharedQueryCache, "false"))
	assert.True(t, Enabled(KeySharedQueryCache), "a direct settings write is not seen until Reload")
	Reload()
	assert.False(t, Enabled(KeySharedQueryCache))
}

func TestSetEnvLockedReturnsConflict(t *testing.T) {
	setup(t)
	require.NoError(t, Set(KeyHandoffClaims, false))
	t.Setenv("AST_FEATURE_HANDOFF_CLAIMS", "true")
	Reload()
	assert.True(t, Enabled(KeyHandoffClaims))
	err := Set(KeyHandoffClaims, false)
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeConflict))
	assert.Equal(t, "false", db.GetSetting(KeyHandoffClaims, ""), "a rejected Set leaves the stored setting alone")
	assert.True(t, Enabled(KeyHandoffClaims))
}

func TestUnknownKey(t *testing.T) {
	setup(t)
	assert.False(t, Enabled("feature_nope"))
	assert.False(t, AffectsTools("feature_nope"))
	err := Set("feature_nope", true)
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeNotFound))
	assert.Empty(t, db.GetSetting("feature_nope", ""))
}

func TestMasterSwitchImpliesChildren(t *testing.T) {
	setup(t)
	require.NoError(t, Set(KeyHandoffClaims, false))
	require.NoError(t, Set(KeyHandoff, false))
	for _, key := range []string{KeyHandoff, KeyHandoffScratchpad, KeyHandoffClaims, KeyHandoffLiveTrail, KeyHandoffHooks} {
		assert.False(t, Enabled(key), key)
	}
	assert.True(t, Enabled(KeySharedQueryCache), "non-handoff flags are not children")
	st := stateOf(t, KeyHandoffScratchpad)
	assert.False(t, st.Enabled)
	assert.Equal(t, SourceDefault, st.Source, "source names where the flag's own value came from")
	for _, tool := range []string{"handoff", "open_handoff", "scratchpad"} {
		assert.False(t, ToolEnabled(tool), tool)
		assert.Equal(t, KeyHandoff, ToolDisabledBy(tool), tool)
	}
	assert.True(t, ToolEnabled("get_context_capsule"))
	require.NoError(t, Set(KeyHandoff, true))
	assert.True(t, Enabled(KeyHandoffScratchpad), "children come back with the master switch")
	assert.False(t, Enabled(KeyHandoffClaims), "a child's own setting survives the master switch")
}

func TestChildOffHidesOnlyItsTool(t *testing.T) {
	setup(t)
	require.NoError(t, Set(KeyHandoffScratchpad, false))
	assert.False(t, ToolEnabled("scratchpad"))
	assert.Equal(t, KeyHandoffScratchpad, ToolDisabledBy("scratchpad"))
	assert.True(t, ToolEnabled("handoff"))
	assert.True(t, ToolEnabled("open_handoff"))
	assert.Empty(t, ToolDisabledBy("handoff"))
}

func TestOnChange(t *testing.T) {
	setup(t)
	got := map[string]bool{}
	calls := 0
	OnChange(func(key string, on bool) {
		got[key] = on
		calls++
	})
	require.NoError(t, Set(KeyHandoff, false))
	want := map[string]bool{KeyHandoff: false, KeyHandoffScratchpad: false, KeyHandoffClaims: false, KeyHandoffLiveTrail: false}
	assert.Equal(t, want, got, "implied children fire; hooks was already off and shared cache is unaffected")
	calls = 0
	require.NoError(t, Set(KeyHandoff, false))
	assert.Zero(t, calls, "no effective change, no callback")
	require.NoError(t, Set(KeyHandoffHooks, true))
	assert.Zero(t, calls, "a child set while the master is off has no effective change")
	clear(got)
	require.NoError(t, Set(KeyHandoff, true))
	assert.Equal(t, map[string]bool{KeyHandoff: true, KeyHandoffScratchpad: true, KeyHandoffClaims: true, KeyHandoffLiveTrail: true, KeyHandoffHooks: true}, got)
}

func TestOnChangeCallbackMayReadFlags(t *testing.T) {
	setup(t)
	var seen bool
	OnChange(func(key string, on bool) {
		if key == KeySharedQueryCache {
			seen = Enabled(key) == on
		}
	})
	require.NoError(t, Set(KeySharedQueryCache, false))
	assert.True(t, seen, "subscribers run after locks are released and see the new snapshot")
}

func TestAffectsTools(t *testing.T) {
	tests := []struct {
		key  string
		want bool
	}{
		{KeyHandoff, true},
		{KeyHandoffScratchpad, true},
		{KeyHandoffClaims, true},
		{KeyHandoffLiveTrail, false},
		{KeyHandoffHooks, false},
		{KeySharedQueryCache, false},
		{"feature_nope", false},
	}
	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			assert.Equal(t, tt.want, AffectsTools(tt.key))
		})
	}
}

func TestActionEnabled(t *testing.T) {
	setup(t)
	require.NoError(t, Set(KeyHandoffClaims, false))
	assert.True(t, ToolEnabled("scratchpad"), "claims gates actions, not the tool")
	assert.False(t, ActionEnabled("scratchpad", "claim"))
	assert.False(t, ActionEnabled("scratchpad", "release"))
	assert.True(t, ActionEnabled("scratchpad", "append"))
	assert.True(t, ActionEnabled("get_context_capsule", "claim"), "actions are per tool")
	require.NoError(t, Set(KeyHandoffClaims, true))
	require.NoError(t, Set(KeyHandoffScratchpad, false))
	assert.False(t, ActionEnabled("scratchpad", "append"), "a disabled tool disables all its actions")
}

func TestAllReturnsCopy(t *testing.T) {
	all := All()
	require.Len(t, all, len(registry))
	all[0].Tools[0] = "mutated"
	all[2].Actions["scratchpad"][0] = "mutated"
	assert.Equal(t, "handoff", registry[0].Tools[0])
	assert.Equal(t, "claim", registry[2].Actions["scratchpad"][0])
}
