package db_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestSettingFloat(t *testing.T) {
	dbtest.Init(t)
	const key, env = "test_setting_float", "AST_TEST_SETTING_FLOAT"
	tests := []struct {
		name, env, setting string
		want               float64
	}{
		{name: "default", want: 0.5},
		{name: "setting", setting: "0.25", want: 0.25},
		{name: "zero setting", setting: "0", want: 0},
		{name: "env beats setting", env: "0.75", setting: "0.25", want: 0.75},
		{name: "bad env uses default", env: "abc", setting: "0.25", want: 0.5},
		{name: "negative setting uses default", setting: "-1", want: 0.5},
		{name: "nan setting uses default", setting: "NaN", want: 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(env, tt.env)
			require.NoError(t, db.SetSetting(key, tt.setting))
			assert.InDelta(t, tt.want, db.SettingFloat(key, env, 0.5), 1e-9)
		})
	}
}
