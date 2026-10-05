package db

import (
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
