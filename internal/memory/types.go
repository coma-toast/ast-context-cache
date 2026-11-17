package memory

import (
	"encoding/json"
	"strings"
)

// Kind distinguishes temporal facts, procedural rules, and KB/audit memory types.
type Kind string

const (
	KindFact         Kind = "fact"
	KindProcedure    Kind = "procedure"
	KindPreference   Kind = "preference"
	KindEvent        Kind = "event"
	KindConfigChange Kind = "config_change"
	KindDiscovery    Kind = "discovery"
	KindPinned       Kind = "pinned"
)

// IsValidKind reports whether k is one of the 6 storable structured_memory kinds.
func IsValidKind(k Kind) bool {
	switch k {
	case KindFact, KindProcedure, KindPreference, KindEvent, KindConfigChange, KindDiscovery:
		return true
	default:
		return false
	}
}

// Severity levels for event and audit memory entries.
const (
	SeverityInfo     = "info"
	SeverityWarning  = "warning"
	SeverityError    = "error"
	SeverityCritical = "critical"
)

// IsValidSeverity reports whether s is empty (default) or a recognized severity level.
func IsValidSeverity(s string) bool {
	switch s {
	case "", SeverityInfo, SeverityWarning, SeverityError, SeverityCritical:
		return true
	default:
		return false
	}
}

// ReasonSuperseded marks an entry superseded by a newer revision.
const ReasonSuperseded = "superseded"

// Scope controls conflict resolution (Mem0-style user/session/project tiers).
type Scope string

const (
	ScopeSession Scope = "session"
	ScopeProject Scope = "project"
	ScopeGlobal  Scope = "global"
)

// Entry is a stored structured memory item.
type Entry struct {
	Ref               string   `json:"ref"`
	Kind              Kind     `json:"kind"`
	Scope             Scope    `json:"scope"`
	SessionID         string   `json:"session_id,omitempty"`
	ProjectPath       string   `json:"project_path,omitempty"`
	Subject           string   `json:"subject,omitempty"`
	Predicate         string   `json:"predicate,omitempty"`
	Object            string   `json:"object,omitempty"`
	Rule              string   `json:"rule,omitempty"`
	ValidFrom         string   `json:"valid_from,omitempty"`
	ValidUntil        string   `json:"valid_until,omitempty"`
	SupersededBy      string   `json:"superseded_by,omitempty"`
	SourceRef         string   `json:"source_ref,omitempty"`
	TokenEst          int      `json:"token_est,omitempty"`
	AccessCount       int      `json:"access_count,omitempty"`
	LastAccessedAt    string   `json:"last_accessed_at,omitempty"`
	CreatedAt         string   `json:"created_at,omitempty"`
	InvalidatedReason string   `json:"invalidated_reason,omitempty"`
	WorldValidAt      string   `json:"world_valid_at,omitempty"`
	WorldInvalidAt    string   `json:"world_invalid_at,omitempty"`
	ExpiresAt         string   `json:"expires_at,omitempty"`
	Quarantined       int      `json:"quarantined,omitempty"`
	Title             string   `json:"title,omitempty"`
	Body              string   `json:"body,omitempty"`
	Tags              []string `json:"tags,omitempty"`
	Actor             string   `json:"actor,omitempty"`
	Category          string   `json:"category,omitempty"`
	Severity          string   `json:"severity,omitempty"`
	Target            string   `json:"target,omitempty"`
	BeforeValue       string   `json:"before_value,omitempty"`
	AfterValue        string   `json:"after_value,omitempty"`
	Reason            string   `json:"reason,omitempty"`
	OccurredAt        string   `json:"occurred_at,omitempty"`
	Verified          bool     `json:"verified,omitempty"`
	VerifiedAt        string   `json:"verified_at,omitempty"`
	RedactedFields    []string `json:"redacted_fields,omitempty"`
}

// CompactLine is a token-efficient representation for agents.
type CompactLine struct {
	Ref         string   `json:"ref"`
	Kind        Kind     `json:"kind"`
	Line        string   `json:"line"`
	Score       float64  `json:"score,omitempty"`
	Title       string   `json:"title,omitempty"`
	Scope       Scope    `json:"scope,omitempty"`
	ProjectPath string   `json:"project_path,omitempty"`
	OccurredAt  string   `json:"occurred_at,omitempty"`
	CreatedAt   string   `json:"created_at,omitempty"`
	Severity    string   `json:"severity,omitempty"`
	Category    string   `json:"category,omitempty"`
	Actor       string   `json:"actor,omitempty"`
	Target      string   `json:"target,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Verified    bool     `json:"verified,omitempty"`
}

// normalizeTagList trims whitespace, drops empty elements, and deduplicates preserving order.
func normalizeTagList(items []string) []string {
	if len(items) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(items))
	var out []string
	for _, item := range items {
		t := strings.TrimSpace(item)
		if t == "" {
			continue
		}
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		out = append(out, t)
	}
	return out
}

// EncodeTags normalizes tags and encodes them as a JSON array string for storage.
// Returns "" when there are no non-empty tags.
func EncodeTags(tags []string) string {
	norm := normalizeTagList(tags)
	if len(norm) == 0 {
		return ""
	}
	b, err := json.Marshal(norm)
	if err != nil {
		return ""
	}
	return string(b)
}

// DecodeTags parses a stored tags column value, handling both JSON arrays and
// legacy comma-separated strings.
func DecodeTags(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	if strings.HasPrefix(raw, "[") {
		var arr []string
		if json.Unmarshal([]byte(raw), &arr) == nil {
			return normalizeTagList(arr)
		}
	}
	return normalizeTagList(strings.Split(raw, ","))
}
