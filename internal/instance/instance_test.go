package instance

import (
	"os"
	"strings"
	"testing"
)

func withEnv(t *testing.T, key, value string) {
	t.Helper()
	old, had := os.LookupEnv(key)
	if value == "" {
		os.Unsetenv(key)
	} else {
		os.Setenv(key, value)
	}
	t.Cleanup(func() {
		if had {
			os.Setenv(key, old)
		} else {
			os.Unsetenv(key)
		}
	})
	Init("")
}

func TestSlugify(t *testing.T) {
	cases := map[string]string{
		"jd":            "jd",
		"Slide":         "slide",
		"My Instance 1": "my-instance-1",
		"a__b..c  d":    "a-b-c-d",
		"  padded  ":    "padded",
		"---":           "",
		"":              "",
		"UPPER lower":   "upper-lower",
	}
	for in, want := range cases {
		if got := Slugify(in); got != want {
			t.Errorf("Slugify(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestValidateName(t *testing.T) {
	valid := []string{"", "jd", "slide", "My Instance 1", "a-b_c.d", strings.Repeat("x", 40)}
	for _, v := range valid {
		if ok, err := ValidateName(v); !ok {
			t.Errorf("ValidateName(%q) = false (%v), want true", v, err)
		}
	}
	invalid := []string{"has/slash", "semi;colon", "quote\"", "back\\slash", "emoji😀", strings.Repeat("x", 41)}
	for _, v := range invalid {
		if ok, _ := ValidateName(v); ok {
			t.Errorf("ValidateName(%q) = true, want false", v)
		}
	}
}

func TestPrecedenceFlagEnvSettingEmpty(t *testing.T) {
	dir := t.TempDir()

	// Empty: everything unset.
	withEnv(t, "AST_INSTANCE_NAME", "")
	if n := Resolve("", false, dir); n.Value != "" || n.Source != SourceDefault {
		t.Errorf("empty resolve = %+v, want empty value with default source", n)
	}

	// Setting only.
	withEnv(t, "AST_INSTANCE_NAME", "")
	Init("")
	if n := Resolve("fromdb", false, dir); n.Value != "fromdb" || n.Source != SourceSettings {
		t.Errorf("setting resolve = %+v, want fromdb/settings", n)
	}

	// Env beats setting.
	withEnv(t, "AST_INSTANCE_NAME", "fromenv")
	Init("")
	if n := Resolve("fromdb", false, dir); n.Value != "fromenv" || n.Source != SourceEnv {
		t.Errorf("env resolve = %+v, want fromenv/env", n)
	}

	// Flag beats env.
	withEnv(t, "AST_INSTANCE_NAME", "fromenv")
	Init("fromflag")
	if n := Resolve("fromdb", false, dir); n.Value != "fromflag" || n.Source != SourceFlag {
		t.Errorf("flag resolve = %+v, want fromflag/flag", n)
	}
	if !IsLocked() {
		t.Error("IsLocked() = false with flag set, want true")
	}

	// Env alone locks too.
	Init("")
	if !IsLocked() {
		t.Error("IsLocked() = false with env set, want true")
	}

	// Nothing set: unlocked.
	withEnv(t, "AST_INSTANCE_NAME", "")
	Init("")
	if IsLocked() {
		t.Error("IsLocked() = true with nothing set, want false")
	}
}

func TestCookieScopesDiffer(t *testing.T) {
	withEnv(t, "AST_INSTANCE_NAME", "")
	Init("")
	a := Resolve("jd", false, t.TempDir())
	b := Resolve("slide", false, t.TempDir())
	if CookieName(a.Scope) == CookieName(b.Scope) {
		t.Errorf("same cookie name %q for two different names", CookieName(a.Scope))
	}
	if !strings.HasPrefix(CookieName(a.Scope), "astcache_session_") {
		t.Errorf("cookie name %q missing scoped prefix", CookieName(a.Scope))
	}
}

func TestFallbackScopesDifferPerDataDir(t *testing.T) {
	withEnv(t, "AST_INSTANCE_NAME", "")
	Init("")
	a := Resolve("", false, t.TempDir())
	b := Resolve("", false, t.TempDir())
	if a.Scope == "" || b.Scope == "" {
		t.Fatal("fallback scope must not be empty")
	}
	if a.Scope == b.Scope {
		t.Errorf("same fallback scope %q for two data dirs", a.Scope)
	}
	if len(a.Scope) != 10 {
		t.Errorf("fallback scope %q length = %d, want 10", a.Scope, len(a.Scope))
	}
}

func TestTabTitle(t *testing.T) {
	if got := TabTitle(""); got != "AST Context Cache" {
		t.Errorf("empty TabTitle = %q", got)
	}
	if got := TabTitle("jd"); got != "jd · AST Context Cache" {
		t.Errorf("named TabTitle = %q", got)
	}
}

func TestUpdateAndCurrent(t *testing.T) {
	withEnv(t, "AST_INSTANCE_NAME", "")
	Init("")
	n := Update("hello", false, t.TempDir())
	if n.Value != "hello" || n.Source != SourceSettings {
		t.Errorf("Update = %+v, want hello/settings", n)
	}
	if cur := Current(); cur.Value != "hello" {
		t.Errorf("Current = %+v, want hello", cur)
	}
}

func TestResolveOverride(t *testing.T) {
	cases := []struct {
		name, flag, env, setting string
		override                 bool
		wantValue, wantSource    string
	}{
		{name: "settings only", setting: "dash", wantValue: "dash", wantSource: SourceSettings},
		{name: "settings only, override on", setting: "dash", override: true, wantValue: "dash", wantSource: SourceSettings},
		{name: "nothing set, override on", override: true, wantSource: SourceDefault},
		{name: "env wins without override", env: "fromenv", setting: "dash", wantValue: "fromenv", wantSource: SourceEnv},
		{name: "override beats env", env: "fromenv", setting: "dash", override: true, wantValue: "dash", wantSource: SourceOverride},
		{name: "override with empty setting keeps env", env: "fromenv", override: true, wantValue: "fromenv", wantSource: SourceEnv},
		{name: "flag wins without override", flag: "fromflag", env: "fromenv", setting: "dash", wantValue: "fromflag", wantSource: SourceFlag},
		{name: "override beats flag", flag: "fromflag", setting: "dash", override: true, wantValue: "dash", wantSource: SourceOverride},
		{name: "override with empty setting keeps flag", flag: "fromflag", env: "fromenv", override: true, wantValue: "fromflag", wantSource: SourceFlag},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			withEnv(t, "AST_INSTANCE_NAME", tc.env)
			Init(tc.flag)
			t.Cleanup(func() { Init("") })
			n := Resolve(tc.setting, tc.override, t.TempDir())
			if n.Value != tc.wantValue || n.Source != tc.wantSource {
				t.Errorf("Resolve = %+v, want %q/%s", n, tc.wantValue, tc.wantSource)
			}
			if want := tc.flag != "" || tc.env != ""; IsLocked() != want {
				t.Errorf("IsLocked() = %v, want %v", IsLocked(), want)
			}
		})
	}
}

func TestOverrideSurvivesRestart(t *testing.T) {
	withEnv(t, "AST_INSTANCE_NAME", "fromenv")
	dir := t.TempDir()
	Init("")
	first := Update("dash", true, dir)
	// Restart: package state re-initialized with the env still set, then the
	// stored setting and override are applied again (as main does after db.Init).
	Init("")
	if n := Update("", false, dir); n.Value != "fromenv" {
		t.Fatalf("pre-DB resolve = %+v, want fromenv", n)
	}
	n := Update("dash", true, dir)
	if n != first || n.Value != "dash" || n.Source != SourceOverride || Current() != n {
		t.Errorf("after restart = %+v (current %+v), want %+v", n, Current(), first)
	}
	if v, src := External(); v != "fromenv" || src != SourceEnv {
		t.Errorf("External = %q/%s, want fromenv/env", v, src)
	}
}
