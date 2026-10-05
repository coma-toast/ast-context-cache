// Package handoff lets a parent agent session hand a subagent a compact, snapshot-backed brief
// and collect a capped summary back. Sessions linked by handoffs form a tree that shares a
// scratchpad and advisory claims, and the whole tree expires together.
//
// Every handoff table lives in context.db, and every write goes through db.HandoffTx, the
// one-connection IMMEDIATE pool, so tree mutations are linearized.
package handoff

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/logging"
)

// nowFunc is the service clock; tests replace it to age rows without sleeping.
var nowFunc = time.Now

// defaultService backs Default for callers that can't take the service as a dependency (the
// MCP tool router).
var defaultService atomic.Pointer[Service]

// Service creates, opens, and completes handoffs and runs a tree's scratchpad and claims.
type Service interface {
	// Create snapshots the parent session and returns a new handoff ref (HO-1–HO-10).
	Create(ctx context.Context, req CreateRequest) (*CreateResponse, error)
	// Open mints (or resumes) a child session and returns the compact digest (OP-1–OP-3).
	Open(ctx context.Context, req OpenRequest) (*OpenResponse, error)
	// Expand returns snapshot items in full, with pointer staleness (OP-4, OP-7, OP-8).
	Expand(ctx context.Context, req ExpandRequest) (*ExpandResponse, error)
	// Complete stores a child's result and returns its stub (RT-1–RT-8).
	Complete(ctx context.Context, req CompleteRequest) (*CompleteResponse, error)
	// Collect fans in children's results (FI-1, FI-4–FI-6).
	Collect(ctx context.Context, req CollectRequest) (*CollectResponse, error)
	// List returns the handoffs a parent created (FI-2).
	List(ctx context.Context, req ListRequest) (*ListResponse, error)
	// Status returns a compact view of one tree.
	Status(ctx context.Context, req StatusRequest) (*StatusResponse, error)
	// Flush deletes a whole tree (RQ-3).
	Flush(ctx context.Context, req FlushRequest) (*FlushResponse, error)
	// Post appends a scratchpad entry (SP-1, SP-2).
	Post(ctx context.Context, req PostRequest) (*PostResponse, error)
	// Read pages through the scratchpad (SP-4, SP-7).
	Read(ctx context.Context, req ReadRequest) (*ReadResponse, error)
	// Retract hides one of the caller's own entries (SP-3).
	Retract(ctx context.Context, req RetractRequest) (*RetractResponse, error)
	// Claim takes or queues for an advisory claim (CL-1, CL-2, CL-6).
	Claim(ctx context.Context, req ClaimRequest) (*ClaimResponse, error)
	// Release gives up a claim and grants it to the next waiter (CL-3, CL-4).
	Release(ctx context.Context, req ReleaseRequest) (*ReleaseResponse, error)
	// Annotate returns the response-level fields to merge into a tree session's search
	// response and marks parent-explored results in place; nil outside a tree (OP-9, OP-10, SP-6).
	Annotate(sid SessionID, ev SearchEvent, results []map[string]any) map[string]any
	// Touch records MCP activity for sid, reviving an abandoned child (FI-3).
	Touch(sid SessionID)
	// PendingGrants returns claims granted to sid since it was last told, marking them notified (CL-5).
	PendingGrants(sid SessionID) ([]Grant, error)
	// IsTreeSession reports whether sid is a root or child session of any tree. It is
	// answered from memory after the first lookup, so non-tree sessions pay nothing (NFR-2).
	IsTreeSession(sid SessionID) bool
}

type realService struct {
	emb     embedder.Interface
	logger  *slog.Logger
	trees   *treeIndex
	waiters *waitHub
}

// New returns the handoff service and starts its expiry sweeper and abandonment loops, which
// stop when ctx is done.
func New(ctx context.Context, emb embedder.Interface) Service {
	s := newService(emb)
	go s.sweepLoop(ctx)
	go s.abandonLoop(ctx)
	return s
}

// Start builds the service with New and installs it as the Default.
func Start(ctx context.Context, emb embedder.Interface) Service {
	s := New(ctx, emb)
	SetDefault(s)
	return s
}

// Default returns the service installed by Start or SetDefault, or nil before either.
func Default() Service {
	if p := defaultService.Load(); p != nil {
		return *p
	}
	return nil
}

// SetDefault installs s as the Default; nil clears it.
func SetDefault(s Service) {
	if s == nil {
		defaultService.Store(nil)
		return
	}
	defaultService.Store(&s)
}

// Create is implemented in Phase 6.4.
func (s *realService) Create(ctx context.Context, req CreateRequest) (*CreateResponse, error) {
	return nil, notImplemented("create")
}

// Open is implemented in Phase 6.5.
func (s *realService) Open(ctx context.Context, req OpenRequest) (*OpenResponse, error) {
	return nil, notImplemented("open")
}

// Expand is implemented in Phase 6.5.
func (s *realService) Expand(ctx context.Context, req ExpandRequest) (*ExpandResponse, error) {
	return nil, notImplemented("expand")
}

// Complete is implemented in Phase 6.7.
func (s *realService) Complete(ctx context.Context, req CompleteRequest) (*CompleteResponse, error) {
	return nil, notImplemented("complete")
}

// Collect is implemented in Phase 6.8.
func (s *realService) Collect(ctx context.Context, req CollectRequest) (*CollectResponse, error) {
	return nil, notImplemented("collect")
}

// List is implemented in Phase 6.8.
func (s *realService) List(ctx context.Context, req ListRequest) (*ListResponse, error) {
	return nil, notImplemented("list")
}

// Status is implemented in Phase 6.8.
func (s *realService) Status(ctx context.Context, req StatusRequest) (*StatusResponse, error) {
	return nil, notImplemented("status")
}

// Flush is implemented in Phase 6.8 on top of flushTree.
func (s *realService) Flush(ctx context.Context, req FlushRequest) (*FlushResponse, error) {
	return nil, notImplemented("flush")
}

// Post is implemented in Phase 7.1.
func (s *realService) Post(ctx context.Context, req PostRequest) (*PostResponse, error) {
	return nil, notImplemented("post")
}

// Read is implemented in Phase 7.1.
func (s *realService) Read(ctx context.Context, req ReadRequest) (*ReadResponse, error) {
	return nil, notImplemented("read")
}

// Retract is implemented in Phase 7.1.
func (s *realService) Retract(ctx context.Context, req RetractRequest) (*RetractResponse, error) {
	return nil, notImplemented("retract")
}

// Claim is implemented in Phase 7.3.
func (s *realService) Claim(ctx context.Context, req ClaimRequest) (*ClaimResponse, error) {
	return nil, notImplemented("claim")
}

// Release is implemented in Phase 7.3.
func (s *realService) Release(ctx context.Context, req ReleaseRequest) (*ReleaseResponse, error) {
	return nil, notImplemented("release")
}

// Annotate is implemented in Phase 6.6; until then no response is annotated.
func (s *realService) Annotate(sid SessionID, ev SearchEvent, results []map[string]any) map[string]any {
	return nil
}

// Touch is implemented in Phase 6.9; until then activity is not recorded.
func (s *realService) Touch(sid SessionID) {}

// PendingGrants is implemented in Phase 7.3.
func (s *realService) PendingGrants(sid SessionID) ([]Grant, error) {
	return nil, notImplemented("pending_grants")
}

// IsTreeSession reports whether sid belongs to a tree.
func (s *realService) IsTreeSession(sid SessionID) bool {
	_, ok := s.trees.lookup(sid)
	return ok
}

func newService(emb embedder.Interface) *realService {
	return &realService{
		emb:     emb,
		logger:  logging.Tagged("handoff"),
		trees:   newTreeIndex(),
		waiters: newWaitHub(),
	}
}

func notImplemented(method string) error {
	return errs.NewCode(errs.CodeUnsupported, "not implemented yet", "method", method)
}

// sqlTime formats t like SQLite's datetime('now'), in UTC, so stored times compare as text.
func sqlTime(t time.Time) string {
	return t.UTC().Format(time.DateTime)
}
