package memory

// Kind distinguishes temporal facts from procedural rules.
type Kind string

const (
	KindFact      Kind = "fact"
	KindProcedure Kind = "procedure"
)

// Scope controls conflict resolution (Mem0-style user/session/project tiers).
type Scope string

const (
	ScopeSession Scope = "session"
	ScopeProject Scope = "project"
	ScopeGlobal  Scope = "global"
)

// Entry is a stored structured memory item.
type Entry struct {
	Ref            string `json:"ref"`
	Kind           Kind   `json:"kind"`
	Scope          Scope  `json:"scope"`
	SessionID      string `json:"session_id,omitempty"`
	ProjectPath    string `json:"project_path,omitempty"`
	Subject        string `json:"subject,omitempty"`
	Predicate      string `json:"predicate,omitempty"`
	Object         string `json:"object,omitempty"`
	Rule           string `json:"rule,omitempty"`
	ValidFrom      string `json:"valid_from,omitempty"`
	ValidUntil     string `json:"valid_until,omitempty"`
	SupersededBy   string `json:"superseded_by,omitempty"`
	SourceRef      string `json:"source_ref,omitempty"`
	TokenEst       int    `json:"token_est,omitempty"`
	AccessCount    int    `json:"access_count,omitempty"`
	LastAccessedAt string `json:"last_accessed_at,omitempty"`
	CreatedAt      string `json:"created_at,omitempty"`
}

// CompactLine is a token-efficient representation for agents.
type CompactLine struct {
	Ref   string `json:"ref"`
	Kind  Kind   `json:"kind"`
	Line  string `json:"line"`
	Score float64 `json:"score,omitempty"`
}
