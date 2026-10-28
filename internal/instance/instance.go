package instance

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

const (
	SourceFlag     = "flag"
	SourceEnv      = "env"
	SourceSettings = "settings"
	SourceOverride = "override"
	SourceDefault  = "default"
)

const (
	// SettingKey is the settings key holding the dashboard-set name.
	SettingKey = "instance_name"
	// OverrideKey is the settings key that, when "true", lets the dashboard-set
	// name win over -name and AST_INSTANCE_NAME.
	OverrideKey = "instance_name_override"
)

// Name holds the resolved instance name, its scope (used for cookie/localStorage
// isolation), and the source from which it was resolved.
type Name struct {
	Value  string
	Scope  string
	Source string
}

var (
	flagName string

	mu      sync.RWMutex
	current Name
)

// Init registers the flag name value for later lookups by IsLocked and Resolve.
func Init(flagNameValue string) {
	flagName = strings.TrimSpace(flagNameValue)
}

// External returns the name set by the -name flag (source flag) or
// AST_INSTANCE_NAME (source env), flag first, or empty strings when neither is set.
func External() (value, source string) {
	if flagName != "" {
		return flagName, SourceFlag
	}
	if v := strings.TrimSpace(os.Getenv("AST_INSTANCE_NAME")); v != "" {
		return v, SourceEnv
	}
	return "", ""
}

// IsLocked returns true when an external name source (env var or flag) is set.
// Without the override the instance_name setting cannot be changed from the dashboard.
func IsLocked() bool {
	v, _ := External()
	return v != ""
}

// Slugify turns a name into a safe scope identifier: lowercase [a-z0-9-],
// other runs become '-', trimmed.
func Slugify(s string) string {
	var sb strings.Builder
	lastDash := true // start trimmed
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			sb.WriteRune(r)
			lastDash = false
		} else {
			if !lastDash {
				sb.WriteRune('-')
				lastDash = true
			}
		}
	}
	res := strings.Trim(sb.String(), "-")
	return res
}

// FallbackScope generates a scope from the absolute data directory path using
// the first 10 hex characters of sha256(abs(data dir)).
func FallbackScope(dataDir string) string {
	if dataDir == "" {
		dataDir = "."
	}
	abs, err := filepath.Abs(dataDir)
	if err != nil {
		abs = dataDir
	}
	hash := sha256.Sum256([]byte(abs))
	return fmt.Sprintf("%x", hash[:5]) // 5 bytes = 10 hex chars
}

// Resolve creates a Name from the instance_name setting, computing a scope for
// cookie/localStorage isolation. Order: flag > env > setting, except that with
// override on a non-empty setting beats flag and env. If the result is empty, the
// scope falls back to a hash of the data directory.
func Resolve(name string, override bool, dataDir string) Name {
	src, val := SourceSettings, strings.TrimSpace(name)
	if ext, extSrc := External(); ext != "" {
		if override && val != "" {
			src = SourceOverride
		} else {
			src, val = extSrc, ext
		}
	}

	scope := Slugify(val)
	if scope == "" {
		src = SourceDefault
		scope = FallbackScope(dataDir)
	}

	return Name{
		Value:  val,
		Scope:  scope,
		Source: src,
	}
}

// Update resolves the name from the instance_name setting and the override
// setting (see Resolve), stores it as the process-wide current name, and returns
// it. dataDir scopes the fallback hash when no name is set.
func Update(dbSetting string, override bool, dataDir string) Name {
	n := Resolve(strings.TrimSpace(dbSetting), override, dataDir)
	mu.Lock()
	current = n
	mu.Unlock()
	return n
}

// Current returns the last name stored by Update, or the zero Name if Update
// has never been called.
func Current() Name {
	mu.RLock()
	defer mu.RUnlock()
	return current
}

// CookieName returns the session cookie name for a scope:
// "astcache_session_<scope>", or the legacy bare name when scope is empty.
func CookieName(scope string) string {
	if scope == "" {
		return "astcache_session"
	}
	return "astcache_session_" + scope
}

// TabTitle returns the browser tab title for a name: "<name> · AST Context
// Cache" when set, "AST Context Cache" otherwise.
func TabTitle(name string) string {
	if strings.TrimSpace(name) == "" {
		return "AST Context Cache"
	}
	return strings.TrimSpace(name) + " · AST Context Cache"
}

// ValidateName checks that a value is a valid instance_name setting value.
// Empty strings are allowed (for clearing the setting). Otherwise the name must be
// 1–40 characters of letters, digits, spaces, '-', '_' and '.', trimmed.
func ValidateName(value string) (bool, error) {
	if value == "" {
		return true, nil
	}
	if len(value) > 40 {
		return false, fmt.Errorf("name must be 40 characters or fewer (got %d)", len(value))
	}
	for i, r := range value {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == ' ' || r == '-' || r == '_' || r == '.') {
			return false, fmt.Errorf("invalid character %q at position %d", r, i)
		}
	}
	return true, nil
}
