package config

import "testing"

func TestMergeDefaults(t *testing.T) {
	cfg := &Config{}
	mergeDefaults(cfg)
	if cfg.ListenAddr != ":8080" {
		t.Fatalf("listen addr = %q", cfg.ListenAddr)
	}
}

func TestConfigValidateRejectsNegativeRate(t *testing.T) {
	cfg := &Config{ListenAddr: ":1", RateLimitRPS: -1}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected an error")
	}
}

func TestLoaderValidateWithoutStore(t *testing.T) {
	l := NewLoader(nil)
	if err := l.Validate(&Config{}); err != nil {
		t.Fatal(err)
	}
}
