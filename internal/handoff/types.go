package handoff

import (
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"regexp"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	handoffRefPrefix  = "hof_"
	treeIDPrefix      = "hft_"
	childSessionInfix = ".c"
	// idRandomBytes gives refs and tree ids 64 bits of randomness (HO-5).
	idRandomBytes = 8
)

var (
	handoffRefRegex = regexp.MustCompile(`^hof_[0-9a-f]{16}$`)
	treeIDRegex     = regexp.MustCompile(`^hft_[0-9a-f]{16}$`)
)

// HandoffRef identifies one handoff: "hof_" followed by 16 hex characters.
type HandoffRef string

// TreeID identifies a handoff tree: "hft_" followed by 16 hex characters.
type TreeID string

// SessionID is an agent session id: a root session's own free-form id, or a child id minted by
// Open as "<handoff ref>.c<N>".
type SessionID string

// Mode says whether a child inherited its parent's context window (OP-5).
type Mode string

// Mode values.
const (
	// ModeFresh children start with empty dedup state; the explored manifest is informational.
	ModeFresh Mode = "fresh"
	// ModeFork children inherit the parent's window, so their dedup state is seeded with it.
	ModeFork Mode = "fork"
)

// Status is a child session's state. Done, partial, and failed are completion statuses.
type Status string

// Status values.
const (
	StatusOpen      Status = "open"
	StatusDone      Status = "done"
	StatusPartial   Status = "partial"
	StatusFailed    Status = "failed"
	StatusAbandoned Status = "abandoned"
)

// Section is one part of a handoff snapshot (handoff_snapshot_items.section).
type Section string

// Section values.
const (
	SectionManifest Section = "manifest"
	SectionTrail    Section = "trail"
	SectionNote     Section = "note"
	SectionMemory   Section = "memory"
	SectionPointer  Section = "pointer"
)

// EntryType is a scratchpad entry's type (SP-2).
type EntryType string

// EntryType values.
const (
	EntryTypeFinding EntryType = "finding"
	EntryTypeDeadEnd EntryType = "dead_end"
	EntryTypeClaim   EntryType = "claim"
	EntryTypeTrail   EntryType = "trail"
)

// Change describes how an expanded pointer differs from its snapshot (OP-7).
type Change string

// Change values.
const (
	ChangeFresh       Change = "fresh"
	ChangeModified    Change = "modified"
	ChangeMoved       Change = "moved"
	ChangeDeleted     Change = "deleted"
	ChangeFileMissing Change = "file_missing"
)

// SummarySource says who wrote a completion summary (RT-3).
type SummarySource string

// SummarySource values.
const (
	SummarySourceChild   SummarySource = "child"
	SummarySourceDerived SummarySource = "derived"
)

// ClaimOutcome is the result of a claim request (CL-2).
type ClaimOutcome string

// ClaimOutcome values.
const (
	ClaimGranted ClaimOutcome = "granted"
	ClaimQueued  ClaimOutcome = "queued"
	// ClaimHeld means the caller already holds the key.
	ClaimHeld ClaimOutcome = "held"
)

// PointerInput is one pointer a parent hands to its children: a symbol key ("file|name" or an
// fqn) or a project-relative file path, with an optional note.
type PointerInput struct {
	Key  string `json:"key"`
	Note string `json:"note,omitempty"`
}

// Breakdown is a snapshot's token estimate per section (HO-5, HO-7).
type Breakdown struct {
	Manifest int `json:"manifest"`
	Trail    int `json:"trail"`
	Notes    int `json:"notes"`
	Memory   int `json:"memory"`
	Pointers int `json:"pointers"`
	Total    int `json:"total"`
}

// CreateRequest is a parent's handoff creation (HO-1, HO-4).
type CreateRequest struct {
	SessionID   SessionID      `json:"session_id"`
	ProjectPath string         `json:"project_path,omitempty"`
	Brief       string         `json:"brief"`
	Label       string         `json:"label,omitempty"`
	Pointers    []PointerInput `json:"pointers,omitempty"`
	CtxRefs     []string       `json:"ctx_refs,omitempty"`
	MemRefs     []string       `json:"mem_refs,omitempty"`
	Mode        Mode           `json:"mode,omitempty"`
	// ExcludeTrail drops trail entries by index; ExcludeAllTrail (exclude_trail="all") drops them all.
	ExcludeTrail      []int  `json:"exclude_trail,omitempty"`
	ExcludeAllTrail   bool   `json:"exclude_all_trail,omitempty"`
	ExcludeTrailQuery string `json:"exclude_trail_query,omitempty"`
	// IncludeManifest defaults to true when nil.
	IncludeManifest *bool `json:"include_manifest,omitempty"`
}

// CreateResponse carries the new ref, its prompt stub, and the snapshot's size (HO-5).
type CreateResponse struct {
	Ref       HandoffRef `json:"handoff"`
	TreeID    TreeID     `json:"tree_id"`
	Depth     int        `json:"depth"`
	Stub      string     `json:"stub"`
	Breakdown Breakdown  `json:"breakdown"`
}

// PageCursor says where a truncated open or expand response continues (OP-3).
type PageCursor struct {
	Section Section `json:"section"`
	Offset  int     `json:"offset"`
}

// OpenRequest opens a handoff as a new child, or resumes the child named by SessionID (OP-1, OP-2).
type OpenRequest struct {
	Handoff     HandoffRef `json:"handoff"`
	SessionID   SessionID  `json:"session_id,omitempty"`
	ProjectPath string     `json:"project_path,omitempty"`
	TokenBudget int        `json:"token_budget,omitempty"`
	// Next continues a truncated digest.
	Next *PageCursor `json:"next,omitempty"`
}

// PointerDigest is a pointer as the open digest lists it: key and note, no source.
type PointerDigest struct {
	ID   int64  `json:"id"`
	Key  string `json:"key"`
	Note string `json:"note,omitempty"`
	Kind string `json:"kind,omitempty"`
}

// ItemDigest is an included note or memory entry as the open digest lists it.
type ItemDigest struct {
	ID       int64  `json:"id"`
	Ref      string `json:"ref"`
	Label    string `json:"label,omitempty"`
	TokenEst int    `json:"tokens"`
}

// TrailDigest is one snapshot trail entry as the open digest lists it, newest first.
type TrailDigest struct {
	ID      int64  `json:"id"`
	Tool    string `json:"tool"`
	Query   string `json:"query"`
	Hits    int    `json:"hits"`
	ZeroHit bool   `json:"zero_hit,omitempty"`
}

// EntryHeadline is a scratchpad entry shortened for digests.
type EntryHeadline struct {
	ID       int64     `json:"id"`
	Type     EntryType `json:"type"`
	Author   SessionID `json:"author"`
	Headline string    `json:"headline"`
}

// ScratchpadDigest summarizes a tree's scratchpad for the open digest (OP-3, SP-7, CL-8).
type ScratchpadDigest struct {
	Counts   map[EntryType]int `json:"counts,omitempty"`
	Latest   []EntryHeadline   `json:"latest,omitempty"`
	DeadEnds []EntryHeadline   `json:"dead_ends,omitempty"`
	Claims   []ClaimView       `json:"claims,omitempty"`
}

// OpenResponse is the compact open digest (OP-3).
type OpenResponse struct {
	Handoff    HandoffRef        `json:"handoff"`
	SessionID  SessionID         `json:"session_id"`
	TreeID     TreeID            `json:"tree_id"`
	Mode       Mode              `json:"mode"`
	Resumed    bool              `json:"resumed,omitempty"`
	Label      string            `json:"label,omitempty"`
	Brief      string            `json:"brief"`
	Pointers   []PointerDigest   `json:"pointers,omitempty"`
	Notes      []ItemDigest      `json:"notes,omitempty"`
	Memory     []ItemDigest      `json:"memory,omitempty"`
	Trail      []TrailDigest     `json:"trail,omitempty"`
	Scratchpad *ScratchpadDigest `json:"scratchpad,omitempty"`
	TokensUsed int               `json:"tokens_used"`
	Truncated  bool              `json:"truncated,omitempty"`
	Next       *PageCursor       `json:"next,omitempty"`
}

// ExpandRequest expands snapshot items on demand (OP-4). Items are snapshot item ids; All
// (items="all") expands the whole section. Mode applies to pointers: skeleton, auto, or full.
type ExpandRequest struct {
	Handoff     HandoffRef  `json:"handoff"`
	SessionID   SessionID   `json:"session_id"`
	ProjectPath string      `json:"project_path,omitempty"`
	Section     Section     `json:"section"`
	Items       []int64     `json:"items,omitempty"`
	All         bool        `json:"all,omitempty"`
	Mode        string      `json:"mode,omitempty"`
	TokenBudget int         `json:"token_budget,omitempty"`
	Next        *PageCursor `json:"next,omitempty"`
}

// LineRange is an inclusive 1-based line range.
type LineRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// ExpandedItem is one expanded snapshot item. For pointers, Content is the current code and
// Stale/Change/OldLines/NewLines describe drift from the snapshot fingerprint (OP-7).
type ExpandedItem struct {
	ID        int64      `json:"id"`
	Section   Section    `json:"section"`
	Key       string     `json:"key,omitempty"`
	Label     string     `json:"label,omitempty"`
	Content   string     `json:"content,omitempty"`
	FileRel   string     `json:"file,omitempty"`
	FQN       string     `json:"fqn,omitempty"`
	Kind      string     `json:"kind,omitempty"`
	TokenEst  int        `json:"tokens,omitempty"`
	Stale     bool       `json:"stale,omitempty"`
	Change    Change     `json:"change,omitempty"`
	OldLines  *LineRange `json:"old_lines,omitempty"`
	NewLines  *LineRange `json:"new_lines,omitempty"`
	Truncated bool       `json:"truncated,omitempty"`
}

// ExpandResponse returns expanded items within the token budget.
type ExpandResponse struct {
	Handoff    HandoffRef     `json:"handoff"`
	Section    Section        `json:"section"`
	Items      []ExpandedItem `json:"items"`
	TokensUsed int            `json:"tokens_used"`
	Truncated  bool           `json:"truncated,omitempty"`
	Next       *PageCursor    `json:"next,omitempty"`
}

// CompleteRequest is a child's completion (RT-1, RT-8).
type CompleteRequest struct {
	SessionID     SessionID `json:"session_id"`
	ProjectPath   string    `json:"project_path,omitempty"`
	Content       string    `json:"content"`
	Summary       string    `json:"summary,omitempty"`
	Status        Status    `json:"status,omitempty"`
	ChangedFiles  []string  `json:"changed_files,omitempty"`
	OpenQuestions []string  `json:"open_questions,omitempty"`
}

// CompleteResponse carries the stored result's ref and the return stub (RT-2–RT-7).
type CompleteResponse struct {
	ResultRef        string        `json:"result_ref"`
	Handoff          HandoffRef    `json:"handoff"`
	Status           Status        `json:"status"`
	Summary          string        `json:"summary"`
	SummarySource    SummarySource `json:"summary_source"`
	SummaryTruncated bool          `json:"summary_truncated,omitempty"`
	Stub             string        `json:"stub"`
	PromotedMemory   []string      `json:"promoted_memory,omitempty"`
	ReleasedClaims   []string      `json:"released_claims,omitempty"`
	SupersededRef    string        `json:"superseded_ref,omitempty"`
}

// CollectRequest fans in results for one handoff, or every handoff of the parent session
// (FI-1, FI-5, FI-6). WaitSeconds (≤60) long-polls for a child status change.
type CollectRequest struct {
	SessionID   SessionID  `json:"session_id,omitempty"`
	Handoff     HandoffRef `json:"handoff,omitempty"`
	Recursive   bool       `json:"recursive,omitempty"`
	WaitSeconds int        `json:"wait_seconds,omitempty"`
	TokenBudget int        `json:"token_budget,omitempty"`
}

// ChildResult is one child's state in a collect response (FI-1).
type ChildResult struct {
	SessionID        SessionID  `json:"session_id"`
	Handoff          HandoffRef `json:"handoff"`
	Label            string     `json:"label,omitempty"`
	Depth            int        `json:"depth"`
	Status           Status     `json:"status"`
	ResultRef        string     `json:"result_ref,omitempty"`
	Summary          string     `json:"summary,omitempty"`
	SummaryTruncated bool       `json:"summary_truncated,omitempty"`
	LastActivityAt   string     `json:"last_activity_at"`
	ActiveClaims     int        `json:"active_claims"`
	NoteCount        int        `json:"note_count"`
	ChangedFiles     []string   `json:"changed_files,omitempty"`
	OpenQuestions    []string   `json:"open_questions,omitempty"`
}

// CollectResponse lists children within the token budget.
type CollectResponse struct {
	Children   []ChildResult `json:"children"`
	TokensUsed int           `json:"tokens_used"`
	Truncated  bool          `json:"truncated,omitempty"`
	// Waited reports that the call long-polled before answering.
	Waited bool `json:"waited,omitempty"`
}

// ListRequest lists the handoffs a parent session created (FI-2).
type ListRequest struct {
	SessionID SessionID `json:"session_id"`
}

// HandoffSummary is one handoff with per-status child counts.
type HandoffSummary struct {
	Ref          HandoffRef     `json:"handoff"`
	TreeID       TreeID         `json:"tree_id"`
	Label        string         `json:"label,omitempty"`
	Mode         Mode           `json:"mode"`
	Depth        int            `json:"depth"`
	CreatedAt    string         `json:"created_at"`
	Children     int            `json:"children"`
	StatusCounts map[Status]int `json:"status_counts,omitempty"`
}

// ListResponse is the parent's handoffs, newest first.
type ListResponse struct {
	Handoffs []HandoffSummary `json:"handoffs"`
}

// StatusRequest names a handoff or tree; with only SessionID it means that session's tree.
type StatusRequest struct {
	SessionID SessionID  `json:"session_id,omitempty"`
	Handoff   HandoffRef `json:"handoff,omitempty"`
	TreeID    TreeID     `json:"tree_id,omitempty"`
}

// StatusResponse is a compact view of a tree: usage against its caps, expiry, and handoffs.
type StatusResponse struct {
	TreeID        TreeID           `json:"tree_id"`
	RootSessionID SessionID        `json:"root_session_id"`
	ProjectPath   string           `json:"project_path,omitempty"`
	CreatedAt     string           `json:"created_at"`
	LastAccessAt  string           `json:"last_access_at"`
	ExpiresAt     string           `json:"expires_at"`
	TokensUsed    int              `json:"tokens_used"`
	TokensMax     int              `json:"tokens_max"`
	EntriesUsed   int              `json:"entries_used"`
	EntriesMax    int              `json:"entries_max"`
	Handoffs      []HandoffSummary `json:"handoffs"`
	ActiveClaims  int              `json:"active_claims"`
	QueuedClaims  int              `json:"queued_claims"`
}

// FlushRequest names the tree to delete: by tree id, by any of its handoff refs, or by its
// root session (RQ-3).
type FlushRequest struct {
	SessionID SessionID  `json:"session_id,omitempty"`
	Handoff   HandoffRef `json:"handoff,omitempty"`
	TreeID    TreeID     `json:"tree_id,omitempty"`
}

// FlushResponse reports what a tree flush deleted.
type FlushResponse struct {
	TreeID        TreeID `json:"tree_id"`
	Handoffs      int    `json:"handoffs"`
	Children      int    `json:"children"`
	NotesDeleted  int    `json:"notes_deleted"`
	MemoryDeleted int    `json:"memory_deleted"`
}

// PostRequest appends a scratchpad entry (SP-1, SP-2).
type PostRequest struct {
	SessionID SessionID `json:"session_id"`
	Type      EntryType `json:"type"`
	Text      string    `json:"text"`
	Refs      []string  `json:"refs,omitempty"`
}

// PostResponse carries the new entry's id (a read cursor) and the tree's usage.
type PostResponse struct {
	ID          int64  `json:"id"`
	TreeID      TreeID `json:"tree_id"`
	TokenEst    int    `json:"tokens"`
	TokensUsed  int    `json:"tree_tokens_used"`
	EntriesUsed int    `json:"tree_entries_used"`
}

// ReadRequest reads the tree's scratchpad after the Since cursor (SP-4). The caller's own
// entries are excluded unless IncludeOwn is set; retracted entries unless IncludeRetracted.
type ReadRequest struct {
	SessionID        SessionID   `json:"session_id"`
	Since            int64       `json:"since,omitempty"`
	Types            []EntryType `json:"types,omitempty"`
	Author           SessionID   `json:"author,omitempty"`
	IncludeOwn       bool        `json:"include_own,omitempty"`
	IncludeRetracted bool        `json:"include_retracted,omitempty"`
	TokenBudget      int         `json:"token_budget,omitempty"`
}

// ScratchpadEntry is one scratchpad entry.
type ScratchpadEntry struct {
	ID        int64     `json:"id"`
	Author    SessionID `json:"author"`
	Type      EntryType `json:"type"`
	Text      string    `json:"text"`
	Refs      []string  `json:"refs,omitempty"`
	TokenEst  int       `json:"tokens"`
	CreatedAt string    `json:"created_at"`
	Retracted bool      `json:"retracted,omitempty"`
}

// ReadResponse is a page of entries plus the dead-ends view and active claims (SP-7, CL-8).
type ReadResponse struct {
	Entries    []ScratchpadEntry `json:"entries"`
	DeadEnds   []ScratchpadEntry `json:"dead_ends,omitempty"`
	Claims     []ClaimView       `json:"claims,omitempty"`
	NextCursor int64             `json:"next_cursor"`
	TokensUsed int               `json:"tokens_used"`
	Truncated  bool              `json:"truncated,omitempty"`
}

// RetractRequest hides one of the caller's own entries (SP-3).
type RetractRequest struct {
	SessionID SessionID `json:"session_id"`
	Entry     int64     `json:"entry"`
}

// RetractResponse confirms a retraction.
type RetractResponse struct {
	Entry     int64 `json:"entry"`
	Retracted bool  `json:"retracted"`
}

// ClaimRequest claims a resource key in the caller's tree (CL-1). Claims are advisory.
type ClaimRequest struct {
	SessionID SessionID `json:"session_id"`
	Key       string    `json:"key"`
	Reason    string    `json:"reason,omitempty"`
}

// ClaimResponse says whether the claim was granted or queued, and behind whom (CL-2).
type ClaimResponse struct {
	Key         string       `json:"key"`
	Outcome     ClaimOutcome `json:"outcome"`
	Holder      SessionID    `json:"holder,omitempty"`
	HolderLabel string       `json:"holder_label,omitempty"`
	Position    int          `json:"position,omitempty"`
}

// ReleaseRequest releases one of the caller's claims, or leaves its queue position.
type ReleaseRequest struct {
	SessionID SessionID `json:"session_id"`
	Key       string    `json:"key"`
}

// ReleaseResponse reports the release and who was granted the key next (CL-4).
type ReleaseResponse struct {
	Key       string    `json:"key"`
	Released  bool      `json:"released"`
	GrantedTo SessionID `json:"granted_to,omitempty"`
}

// QueuedClaim is one waiter in a key's FIFO queue.
type QueuedClaim struct {
	SessionID  SessionID `json:"session_id"`
	Reason     string    `json:"reason,omitempty"`
	EnqueuedAt string    `json:"enqueued_at"`
	Position   int       `json:"position"`
}

// ClaimView is an active claim and its queue (CL-8).
type ClaimView struct {
	Key         string        `json:"key"`
	Holder      SessionID     `json:"holder"`
	HolderLabel string        `json:"holder_label,omitempty"`
	Reason      string        `json:"reason,omitempty"`
	GrantedAt   string        `json:"granted_at"`
	Queue       []QueuedClaim `json:"queue,omitempty"`
}

// Grant is a claim granted to a session from a queue, reported once (CL-4, CL-5).
type Grant struct {
	TreeID    TreeID `json:"tree_id"`
	Key       string `json:"key"`
	GrantedAt string `json:"granted_at"`
}

// SearchEvent is the search a tree session just ran, as Annotate matches it against the parent
// and sibling trails (OP-9, SP-6). MatchKey is the trail's match key: tool, normalized query,
// and filters.
type SearchEvent struct {
	Tool     string   `json:"tool"`
	Query    string   `json:"query"`
	MatchKey string   `json:"match_key"`
	Hits     int      `json:"hits"`
	TopKeys  []string `json:"top_keys,omitempty"`
	ZeroHit  bool     `json:"zero_hit,omitempty"`
}

// NewHandoffRef returns a random handoff ref.
func NewHandoffRef() (HandoffRef, error) {
	h, err := randomHex()
	if err != nil {
		return "", err
	}
	return HandoffRef(handoffRefPrefix + h), nil
}

// NewTreeID returns a random tree id.
func NewTreeID() (TreeID, error) {
	h, err := randomHex()
	if err != nil {
		return "", err
	}
	return TreeID(treeIDPrefix + h), nil
}

// ParseHandoffRef validates s as a handoff ref.
func ParseHandoffRef(s string) (HandoffRef, error) {
	if !handoffRefRegex.MatchString(s) {
		return "", errs.NewCode(errs.CodeInvalidInput, "invalid handoff ref", "handoff", s)
	}
	return HandoffRef(s), nil
}

// ParseTreeID validates s as a tree id.
func ParseTreeID(s string) (TreeID, error) {
	if !treeIDRegex.MatchString(s) {
		return "", errs.NewCode(errs.CodeInvalidInput, "invalid tree id", "tree", s)
	}
	return TreeID(s), nil
}

// ChildSessionID returns the session id of ref's nth child: "<ref>.c<n>".
func ChildSessionID(ref HandoffRef, n int) SessionID {
	return SessionID(string(ref) + childSessionInfix + strconv.Itoa(n))
}

// LogValue expands the ref as handoff=<ref>.
func (r HandoffRef) LogValue() slog.Value {
	return slog.GroupValue(slog.String("handoff", string(r)))
}

// Attr returns the ref as a log attribute that renders as handoff=<ref>; go vet rejects ID
// types passed as bare slog arguments.
func (r HandoffRef) Attr() slog.Attr {
	return slog.Any("", r)
}

// LogValue expands the id as tree=<id>.
func (t TreeID) LogValue() slog.Value {
	return slog.GroupValue(slog.String("tree", string(t)))
}

// Attr returns the id as a log attribute that renders as tree=<id>.
func (t TreeID) Attr() slog.Attr {
	return slog.Any("", t)
}

// LogValue expands the id as session=<id>.
func (s SessionID) LogValue() slog.Value {
	return slog.GroupValue(slog.String("session", string(s)))
}

// Attr returns the id as a log attribute that renders as session=<id>.
func (s SessionID) Attr() slog.Attr {
	return slog.Any("", s)
}

// Valid reports whether m is a known mode.
func (m Mode) Valid() bool {
	return m == ModeFresh || m == ModeFork
}

// Valid reports whether s is a known status.
func (s Status) Valid() bool {
	switch s {
	case StatusOpen, StatusDone, StatusPartial, StatusFailed, StatusAbandoned:
		return true
	}
	return false
}

// Completed reports whether s is one a child can complete with (RT-1).
func (s Status) Completed() bool {
	return s == StatusDone || s == StatusPartial || s == StatusFailed
}

// Valid reports whether s is a known snapshot section.
func (s Section) Valid() bool {
	switch s {
	case SectionManifest, SectionTrail, SectionNote, SectionMemory, SectionPointer:
		return true
	}
	return false
}

// Valid reports whether t is a known scratchpad entry type.
func (t EntryType) Valid() bool {
	switch t {
	case EntryTypeFinding, EntryTypeDeadEnd, EntryTypeClaim, EntryTypeTrail:
		return true
	}
	return false
}

func randomHex() (string, error) {
	b := make([]byte, idRandomBytes)
	if _, err := rand.Read(b); err != nil {
		return "", errs.WrapMessage("failed to read random bytes", err)
	}
	return hex.EncodeToString(b), nil
}
