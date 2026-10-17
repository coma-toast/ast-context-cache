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
	SourceFlag    = "flag"
	SourceEnv     = "env"
	SourceDB      = "db"
	SourceDefault = "default"
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

// IsLocked returns true when an external name source (env var or flag) is set,
// meaning the instance_name setting cannot be changed from the dashboard.
func IsLocked() bool {
	if flagName != "" {
		return true
	}
	if v := strings.TrimSpace(os.Getenv("AST_INSTANCE_NAME")); v != "" {
		return true
	}
	return false
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

// Resolve creates a Name from a raw string, computing a scope for cookie/localStorage
// isolation. If name is empty, it falls back to a hash of the data directory.
func Resolve(name string, dataDir string) Name {
	src := SourceDB
	val := strings.TrimSpace(name)
	if flagName != "" {
		src = SourceFlag
		val = flagName
	} else if env := strings.TrimSpace(os.Getenv("AST_INSTANCE_NAME")); env != "" {
		src = SourceEnv
		val = env
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

// Update resolves the name from the given DB setting value (flag > env >
// setting > empty), stores it as the process-wide current name, and returns it.
// dataDir scopes the fallback hash when no name is set.
func Update(dbSetting, dataDir string) Name {
	n := Resolve(strings.TrimSpace(dbSetting), dataDir)
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
