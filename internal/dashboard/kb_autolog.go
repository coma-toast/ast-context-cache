package dashboard

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/netlisten"
)

const defaultConfigChangeReason = "Dashboard configuration updated"

type settingSnapshot struct {
	val       string
	isSecret  bool
	hadBefore bool
}

func isSecretSettingKey(key string) bool {
	return secretSettingKeys[key] || key == instance.FaviconUploadKey
}

// secretChangeSummary returns safe before/after placeholders for secret settings
// and reports whether a change occurred (skipping unset -> cleared).
func secretChangeSummary(hadBefore bool, newVal string) (before, after string, changed bool) {
	hasNew := strings.TrimSpace(newVal) != ""
	if !hadBefore && !hasNew {
		return "(unset)", "(cleared)", false
	}
	if hadBefore {
		before = "(set)"
	} else {
		before = "(unset)"
	}
	if hasNew {
		after = "(updated)"
	} else {
		after = "(cleared)"
	}
	return before, after, true
}

// snapshotSettingBefore inspects the current persisted value of key before
// applySettings mutates DB or subsystem state. For secret keys it never returns
// the raw secret value.
func snapshotSettingBefore(key string) (val string, isSecret bool, hadBefore bool) {
	if isSecretSettingKey(key) {
		return "", true, strings.TrimSpace(db.GetSetting(key, "")) != ""
	}
	switch {
	case netlisten.IsKey(key), embedqueue.IsQueueCapSetting(key), isInstanceSettingKey(key), isFaviconSettingKey(key):
		val = strings.TrimSpace(db.GetSetting(key, ""))
		return val, false, val != ""
	default:
		val = db.GetSetting(key, "")
		return val, false, val != ""
	}
}

// recordSettingsConfigChange persists a global config_change entry in the
// knowledge base and logs a warning on failure without failing the caller.
func recordSettingsConfigChange(target, beforeVal, afterVal, reason string) bool {
	if db.ContextDB == nil {
		return false
	}
	target = strings.TrimSpace(target)
	if target == "" {
		return false
	}
	if strings.TrimSpace(reason) == "" {
		reason = defaultConfigChangeReason
	}
	body := ""
	if strings.TrimSpace(afterVal) == "" {
		body = "Cleared setting " + target
	}
	_, err := memory.Store(memory.StoreInput{
		Kind:        memory.KindConfigChange,
		Scope:       memory.ScopeGlobal,
		Actor:       "dashboard",
		Target:      target,
		BeforeValue: beforeVal,
		AfterValue:  afterVal,
		Reason:      reason,
		Body:        body,
	})
	if err != nil {
		logger.Warn("Failed to record settings config change", "target", target, "error", err)
		return false
	}
	return true
}
