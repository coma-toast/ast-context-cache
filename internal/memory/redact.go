package memory

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

var (
	rePEMBlock = regexp.MustCompile(`-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----`)
	rePEMLine  = regexp.MustCompile(`-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----`)
	reURLCreds = regexp.MustCompile(`([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@"'` + "`" + `<>]+:[^/\s@"'` + "`" + `<>]+)(@)`)
	reJWT      = regexp.MustCompile(`\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+`)
	reOpenAI   = regexp.MustCompile(`\bsk-(?:proj-|live-|test-|ant-api[0-9]{2}-)?[A-Za-z0-9_-]{20,}`)
	reGitHub   = regexp.MustCompile(`\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})`)
	reAWS      = regexp.MustCompile(`\bAKIA[0-9A-Z]{16}\b`)
	reSlack    = regexp.MustCompile(`\bxox[baprs]-[A-Za-z0-9-]{10,}`)
	reBearer   = regexp.MustCompile(`(?i)(\bBearer\s+)([A-Za-z0-9\-._~+/]{8,}=*)`)
	reKeyValue = regexp.MustCompile(`(?i)(\b(password|passwd|pwd|api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key|client[_-]?secret)["']?\s*[:=]\s*["']?)([^\s"';,]{6,})(["']?)`)
)

// RedactText scans text for secret patterns and replaces each matched secret
// with [REDACTED:<type>]. It returns the redacted text and the deduplicated
// list of secret types matched (in order of first detection).
func RedactText(text string) (string, []string) {
	if text == "" {
		return "", nil
	}

	var matched []string
	seen := make(map[string]struct{})
	record := func(t string) {
		if _, ok := seen[t]; ok {
			return
		}
		seen[t] = struct{}{}
		matched = append(matched, t)
	}

	replaceDirect := func(re *regexp.Regexp, secretType string) {
		text = re.ReplaceAllStringFunc(text, func(m string) string {
			record(secretType)
			return "[REDACTED:" + secretType + "]"
		})
	}

	// 1. Multi-line and single-line PEM private keys first so interior lines
	// are not partially re-matched by key=value rules.
	replaceDirect(rePEMBlock, "private_key")
	replaceDirect(rePEMLine, "private_key")

	// 2. URL embedded credentials (user:pass@): preserve protocol and host.
	text = reURLCreds.ReplaceAllStringFunc(text, func(m string) string {
		sub := reURLCreds.FindStringSubmatch(m)
		if len(sub) != 4 {
			return m
		}
		if strings.HasPrefix(sub[2], "[REDACTED:") {
			return m
		}
		record("url_credentials")
		return sub[1] + "[REDACTED:url_credentials]" + sub[3]
	})

	// 3. Specific token prefixes and JWTs before generic Bearer / key=value.
	replaceDirect(reJWT, "jwt")
	replaceDirect(reOpenAI, "api_key")
	replaceDirect(reGitHub, "github_token")
	replaceDirect(reAWS, "aws_access_key")
	replaceDirect(reSlack, "slack_token")

	// 4. Generic Bearer tokens (keeping "Bearer " prefix).
	text = reBearer.ReplaceAllStringFunc(text, func(m string) string {
		sub := reBearer.FindStringSubmatch(m)
		if len(sub) != 3 {
			return m
		}
		if strings.HasPrefix(sub[2], "[REDACTED:") {
			return m
		}
		record("bearer_token")
		return sub[1] + "[REDACTED:bearer_token]"
	})

	// 5. Explicit key=value / key:value secret assignments (keeping key name).
	text = reKeyValue.ReplaceAllStringFunc(text, func(m string) string {
		sub := reKeyValue.FindStringSubmatch(m)
		if len(sub) != 5 {
			return m
		}
		val := sub[3]
		if strings.HasPrefix(val, "[REDACTED:") && strings.HasSuffix(val, "]") {
			return m
		}
		secretType := normalizeSecretKeyType(sub[2])
		record(secretType)
		return sub[1] + "[REDACTED:" + secretType + "]" + sub[4]
	})

	return text, matched
}

func normalizeSecretKeyType(key string) string {
	k := strings.ToLower(strings.ReplaceAll(key, "-", "_"))
	switch k {
	case "passwd", "pwd":
		return "password"
	default:
		return k
	}
}

// RedactFields redacts secrets across named string pointers in place. It returns
// the deduplicated slice of field names that were modified and human-readable
// warnings (e.g. "redacted api_key in after_value").
func RedactFields(fields map[string]*string) (redactedFields []string, warnings []string) {
	if len(fields) == 0 {
		return nil, nil
	}
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		ptr := fields[name]
		if ptr == nil || *ptr == "" {
			continue
		}
		redacted, types := RedactText(*ptr)
		if len(types) == 0 {
			continue
		}
		*ptr = redacted
		redactedFields = append(redactedFields, name)
		for _, t := range types {
			warnings = append(warnings, fmt.Sprintf("redacted %s in %s", t, name))
		}
	}
	return redactedFields, warnings
}

// RedactStoreInput redacts any secrets from StoreInput text fields in place.
func RedactStoreInput(in *StoreInput) (redactedFields []string, warnings []string) {
	if in == nil {
		return nil, nil
	}
	return RedactFields(map[string]*string{
		"subject":   &in.Subject,
		"predicate": &in.Predicate,
		"object":    &in.Object,
		"rule":      &in.Rule,
	})
}

// RedactEntry redacts any secrets from Entry text fields in place and updates
// e.RedactedFields with the modified field names.
func RedactEntry(e *Entry) (redactedFields []string, warnings []string) {
	if e == nil {
		return nil, nil
	}
	redactedFields, warnings = RedactFields(map[string]*string{
		"title":        &e.Title,
		"subject":      &e.Subject,
		"predicate":    &e.Predicate,
		"object":       &e.Object,
		"rule":         &e.Rule,
		"body":         &e.Body,
		"target":       &e.Target,
		"before_value": &e.BeforeValue,
		"after_value":  &e.AfterValue,
		"reason":       &e.Reason,
		"category":     &e.Category,
		"actor":        &e.Actor,
	})
	if len(redactedFields) > 0 {
		e.RedactedFields = mergeUniqueStrings(e.RedactedFields, redactedFields)
	}
	return redactedFields, warnings
}

func mergeUniqueStrings(existing, added []string) []string {
	seen := make(map[string]struct{}, len(existing)+len(added))
	out := make([]string, 0, len(existing)+len(added))
	for _, s := range existing {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	for _, s := range added {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	return out
}
