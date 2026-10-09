package config

import (
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Loader reads configuration files and applies environment overrides.
type Loader struct {
	EnvPrefix string
	store     Store
}

// NewLoader returns a loader that reads overrides from SHOP_* variables.
func NewLoader(store Store) *Loader {
	return &Loader{EnvPrefix: "SHOP_", store: store}
}

// Load reads the YAML file at path, merges defaults, and applies env overrides.
func (l *Loader) Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	cfg := &Config{}
	if err := yaml.Unmarshal(raw, cfg); err != nil {
		return nil, err
	}
	mergeDefaults(cfg)
	l.applyEnvOverrides(cfg)
	return cfg, l.Validate(cfg)
}

// Validate checks a loaded config and that the loader's store is reachable.
func (l *Loader) Validate(cfg *Config) error {
	if l.store == nil {
		return nil
	}
	return cfg.Validate()
}

func (l *Loader) applyEnvOverrides(cfg *Config) {
	for _, kv := range os.Environ() {
		key, val, ok := strings.Cut(kv, "=")
		if !ok || !strings.HasPrefix(key, l.EnvPrefix) {
			continue
		}
		switch strings.TrimPrefix(key, l.EnvPrefix) {
		case "LISTEN_ADDR":
			cfg.ListenAddr = val
		case "RATE_LIMIT_RPS":
			if n, err := strconv.Atoi(val); err == nil {
				cfg.RateLimitRPS = n
			}
		}
	}
}

// mergeDefaults fills settings the file left empty.
func mergeDefaults(cfg *Config) {
	if cfg.ListenAddr == "" {
		cfg.ListenAddr = ":8080"
	}
	if cfg.CacheTTL == 0 {
		cfg.CacheTTL = 5 * time.Minute
	}
	if cfg.RateLimitRPS == 0 {
		cfg.RateLimitRPS = 50
	}
}
