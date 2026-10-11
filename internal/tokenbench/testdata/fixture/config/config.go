package config

import (
	"fmt"
	"time"
)

// Config is the service configuration after defaults and overrides are applied.
type Config struct {
	ListenAddr   string
	DatabaseURL  string
	CacheTTL     time.Duration
	RateLimitRPS int
	FeatureFlags map[string]bool
}

// Validate reports the first invalid setting in the configuration.
func (c *Config) Validate() error {
	if c.ListenAddr == "" {
		return fmt.Errorf("listen address is required")
	}
	if c.RateLimitRPS < 0 {
		return fmt.Errorf("rate limit must not be negative")
	}
	return nil
}

// String renders the configuration with the database URL redacted.
func (c *Config) String() string {
	return fmt.Sprintf("listen=%s ttl=%s rps=%d", c.ListenAddr, c.CacheTTL, c.RateLimitRPS)
}

// FlagEnabled reports whether a named feature flag is switched on.
func (c *Config) FlagEnabled(name string) bool {
	return c.FeatureFlags[name]
}
