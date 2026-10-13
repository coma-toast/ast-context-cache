package db

import (
	"math"
	"os"
	"strconv"
	"strings"
)

// SettingInt resolves a positive integer setting: the envKey environment variable wins when it
// is set, then the settings table row for key, then def. A value that is set but unparseable or
// not positive yields def rather than falling through, so a bad env value never silently
// defers to a stored setting.
func SettingInt(key, envKey string, def int) int {
	if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
		return positiveIntOr(v, def)
	}
	if v := strings.TrimSpace(GetSetting(key, "")); v != "" {
		return positiveIntOr(v, def)
	}
	return def
}

func positiveIntOr(v string, def int) int {
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return def
	}
	return n
}

// SettingFloat resolves a non-negative float setting with the same precedence as SettingInt:
// the envKey environment variable, then the settings table row for key, then def. A value
// that is set but unparseable or negative yields def.
func SettingFloat(key, envKey string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
		return nonNegativeFloatOr(v, def)
	}
	if v := strings.TrimSpace(GetSetting(key, "")); v != "" {
		return nonNegativeFloatOr(v, def)
	}
	return def
}

func nonNegativeFloatOr(v string, def float64) float64 {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f < 0 || math.IsNaN(f) || math.IsInf(f, 0) {
		return def
	}
	return f
}
