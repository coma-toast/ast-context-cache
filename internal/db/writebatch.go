package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/tokens"
)

const (
	insertQueryLogQuery = `INSERT INTO queries (
		timestamp, tool_name, arguments, result_chars, input_tokens, output_tokens,
		tokens_saved, file_baseline_tokens, full_baseline_tokens,
		tokens_used, symbol_baseline_tokens, dedup_tokens_saved, savings_vs_files,
		deduped_count, mode, cache_hit,
		duration_ms, cpu_ms, interface, session_id, error, project_path, estimate_method
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
	insertSessionLogQuery = `INSERT INTO sessions (session_id, symbol_id, symbol_name, start_line, file_path, mode, token_count) VALUES (?, ?, ?, ?, ?, ?, ?)`
	insertTrailQuery      = `INSERT INTO search_trail (
		session_id, tool, query, query_norm, filters_key, mode, doc_type, project_path,
		hit_count, zero_hit, top_hits_json, created_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
)

// EstimateMethod names the token estimator recorded in queries.estimate_method for rows
// whose QueryLogMetrics.EstimateMethod is empty.
var EstimateMethod = tokens.Method

// Execer matches *sql.DB and *sql.Tx for Exec.
type Execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}

const (
	queryLogFlushInterval = 4 * time.Second
	queryLogFlushSize     = 120
	sessionFlushInterval  = 3 * time.Second
	sessionFlushSize      = 200
	// The search trail is flushed on the session buffer's cadence and limits.
	trailFlushInterval = sessionFlushInterval
	trailFlushSize     = sessionFlushSize
)

// QueryLogMetrics holds analytics fields for a logged MCP tool call.
type QueryLogMetrics struct {
	ResultChars      int
	InputTokens      int
	OutputTokens     int
	TokensUsed       int
	TokensSaved      int
	SymbolBaseline   int
	FileBaseline     int
	DedupTokensSaved int
	SavingsVsFiles   int
	DedupedCount     int
	Mode             string
	CacheHit         bool
	DurationMs       float64
	CpuMs            float64
	// EstimateMethod is the token estimator behind these counts; empty means EstimateMethod().
	EstimateMethod string
}

type queryLogRow struct {
	toolName         string
	argsJSON         string
	metrics          QueryLogMetrics
	sessionID        string
	errMsg           string
	projectPath      string
	timestampRFC3339 string
}

type sessionLogRow struct {
	sessionID  string
	symbolID   int
	symbolName string
	startLine  int
	filePath   string
	mode       string
	tokenCount int
}

// TrailRow is one search_trail row buffered for a batched insert. CreatedAt is the
// caller's formatted timestamp, so the row sorts and dedupes the same as its in-memory copy.
type TrailRow struct {
	SessionID, Tool, Query, QueryNorm, FiltersKey, Mode, DocType, ProjectPath string
	HitCount                                                                  int
	ZeroHit                                                                   bool
	TopHitsJSON, CreatedAt                                                    string
}

// QueryLogSnapshot is a flushed analytics row for dashboard toasts / live updates.
type QueryLogSnapshot struct {
	Timestamp   string
	ToolName    string
	ArgsJSON    string
	ProjectPath string
	TokensSaved int
	DurationMs  float64
	CpuMs       float64
}

// AfterQueryLogFlush is set by the dashboard to push toasts and WS partials (avoids db importing dashboard).
var AfterQueryLogFlush func(rows []QueryLogSnapshot)

var (
	queryBufMu sync.Mutex
	queryBuf   []queryLogRow
	sessBufMu  sync.Mutex
	sessBuf    []sessionLogRow
	trailBufMu sync.Mutex
	trailBuf   []TrailRow
	// trailFlushMu is held for a whole flush, so FlushWriteBuffers returns only after a
	// batch the batcher already took has committed.
	trailFlushMu sync.Mutex

	// A full buffer kicks its batcher rather than spawning a flush goroutine of
	// its own, so every background flush runs on a batcher that Init/Close stop.
	queryFlushKick = make(chan struct{}, 1)
	sessFlushKick  = make(chan struct{}, 1)
	trailFlushKick = make(chan struct{}, 1)

	batcherMu   sync.Mutex
	batcherStop chan struct{}
	batcherDone sync.WaitGroup
)

// StartWriteBatchers starts periodic flush of buffered query/session analytics
// and search trail rows. It is a no-op while they are already running.
func StartWriteBatchers() {
	batcherMu.Lock()
	defer batcherMu.Unlock()
	if batcherStop != nil {
		return
	}
	stop := make(chan struct{})
	batcherStop = stop
	batcherDone.Add(3)
	go runWriteBatcher(stop, queryLogFlushInterval, queryFlushKick, flushQueryLogBuffer)
	go runWriteBatcher(stop, sessionFlushInterval, sessFlushKick, flushSessionLogBuffer)
	go runWriteBatcher(stop, trailFlushInterval, trailFlushKick, flushTrailBuffer)
}

func runWriteBatcher(stop <-chan struct{}, every time.Duration, kick <-chan struct{}, flush func()) {
	defer batcherDone.Done()
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
		case <-kick:
		}
		flush()
	}
}

// stopWriteBatchers stops the batchers and waits out a flush in progress. Init
// and Close call it before reassigning or closing the pools: the batchers read
// DB, and used to outlive the Init that started them — every Init added two
// more, each racing the next Init's write of DB.
func stopWriteBatchers() {
	batcherMu.Lock()
	defer batcherMu.Unlock()
	if batcherStop == nil {
		return
	}
	close(batcherStop)
	batcherStop = nil
	batcherDone.Wait()
}

func kickFlush(kick chan<- struct{}) {
	select {
	case kick <- struct{}{}:
	default:
	}
}

func flushQueryLogBuffer() {
	queryBufMu.Lock()
	if len(queryBuf) == 0 {
		queryBufMu.Unlock()
		return
	}
	batch := queryBuf
	queryBuf = nil
	queryBufMu.Unlock()
	tx, err := DB.Begin()
	if err != nil {
		logger.Warn("Failed to begin query log batch", "error", err)
		return
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(insertQueryLogQuery)
	if err != nil {
		logger.Warn("Failed to prepare query log batch", "error", err)
		return
	}
	defer stmt.Close()
	for _, r := range batch {
		m := r.metrics
		cacheHit := 0
		if m.CacheHit {
			cacheHit = 1
		}
		// full_baseline_tokens has always received SymbolBaseline (the same value as
		// symbol_baseline_tokens). Nothing reads it for the savings ledgers; kept as is.
		if _, err := stmt.Exec(
			r.timestampRFC3339, r.toolName, r.argsJSON, m.ResultChars, m.InputTokens, m.OutputTokens,
			m.TokensSaved, m.FileBaseline, m.SymbolBaseline,
			m.TokensUsed, m.SymbolBaseline, m.DedupTokensSaved, m.SavingsVsFiles,
			m.DedupedCount, m.Mode, cacheHit,
			m.DurationMs, m.CpuMs, "http", r.sessionID, r.errMsg, r.projectPath, m.EstimateMethod,
		); err != nil {
			logger.Warn("Failed to insert query log row", "error", err)
		}
	}
	if err := tx.Commit(); err != nil {
		logger.Warn("Failed to commit query log batch", "error", err)
		return
	}
	if AfterQueryLogFlush != nil {
		snap := make([]QueryLogSnapshot, len(batch))
		for i, r := range batch {
			snap[i] = QueryLogSnapshot{
				Timestamp:   r.timestampRFC3339,
				ToolName:    r.toolName,
				ArgsJSON:    r.argsJSON,
				ProjectPath: r.projectPath,
				TokensSaved: r.metrics.TokensSaved,
				DurationMs:  r.metrics.DurationMs,
				CpuMs:       r.metrics.CpuMs,
			}
		}
		AfterQueryLogFlush(snap)
	}
}

func flushSessionLogBuffer() {
	sessBufMu.Lock()
	if len(sessBuf) == 0 {
		sessBufMu.Unlock()
		return
	}
	batch := sessBuf
	sessBuf = nil
	sessBufMu.Unlock()
	tx, err := DB.Begin()
	if err != nil {
		logger.Warn("Failed to begin session log batch", "error", err)
		return
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(insertSessionLogQuery)
	if err != nil {
		logger.Warn("Failed to prepare session log batch", "error", err)
		return
	}
	defer stmt.Close()
	for _, r := range batch {
		if _, err := stmt.Exec(r.sessionID, r.symbolID, r.symbolName, r.startLine, r.filePath, r.mode, r.tokenCount); err != nil {
			logger.Warn("Failed to insert session log row", "error", err)
		}
	}
	if err := tx.Commit(); err != nil {
		logger.Warn("Failed to commit session log batch", "error", err)
	}
}

func flushTrailBuffer() {
	trailFlushMu.Lock()
	defer trailFlushMu.Unlock()
	trailBufMu.Lock()
	if len(trailBuf) == 0 {
		trailBufMu.Unlock()
		return
	}
	batch := trailBuf
	trailBuf = nil
	trailBufMu.Unlock()
	tx, err := DB.Begin()
	if err != nil {
		logger.Warn("Failed to begin search trail batch", "error", err)
		return
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(insertTrailQuery)
	if err != nil {
		logger.Warn("Failed to prepare search trail batch", "error", err)
		return
	}
	defer stmt.Close()
	for _, r := range batch {
		zeroHit := 0
		if r.ZeroHit {
			zeroHit = 1
		}
		if _, err := stmt.Exec(r.SessionID, r.Tool, r.Query, r.QueryNorm, r.FiltersKey, r.Mode, r.DocType, r.ProjectPath, r.HitCount, zeroHit, r.TopHitsJSON, r.CreatedAt); err != nil {
			logger.Warn("Failed to insert search trail row", "error", err, "session", r.SessionID)
		}
	}
	if err := tx.Commit(); err != nil {
		logger.Warn("Failed to commit search trail batch", "error", err, "rows", len(batch))
	}
}

// EnqueueTrail buffers a search trail row; flushed periodically in batches.
func EnqueueTrail(r TrailRow) {
	if r.SessionID == "" {
		return
	}
	trailBufMu.Lock()
	trailBuf = append(trailBuf, r)
	n := len(trailBuf)
	trailBufMu.Unlock()
	if n >= trailFlushSize {
		kickFlush(trailFlushKick)
	}
}

// EnqueueSessionReturned buffers a session dedup row; flushed periodically in batches.
func EnqueueSessionReturned(sessionID string, symbolID int, symbolName string, startLine int, filePath, mode string, tokenCount int) {
	if sessionID == "" {
		return
	}
	sessBufMu.Lock()
	sessBuf = append(sessBuf, sessionLogRow{
		sessionID: sessionID, symbolID: symbolID, symbolName: symbolName, startLine: startLine,
		filePath: filePath, mode: mode, tokenCount: tokenCount,
	})
	n := len(sessBuf)
	sessBufMu.Unlock()
	if n >= sessionFlushSize {
		kickFlush(sessFlushKick)
	}
}

func extractSessionID(args map[string]interface{}) string {
	if args == nil {
		return ""
	}
	if inner, ok := args["arguments"].(map[string]interface{}); ok {
		if sid, ok := inner["session_id"].(string); ok && sid != "" {
			return sid
		}
	}
	if sid, ok := args["session_id"].(string); ok && sid != "" {
		return sid
	}
	return fmt.Sprintf("session-%d", time.Now().Unix()/3600)
}

func enqueueQueryLog(toolName string, args map[string]interface{}, m QueryLogMetrics, projectPath, errMsg string) {
	argsJSON, _ := json.Marshal(args)
	if m.EstimateMethod == "" {
		m.EstimateMethod = EstimateMethod()
	}
	r := queryLogRow{
		toolName:         toolName,
		argsJSON:         string(argsJSON),
		metrics:          m,
		sessionID:        extractSessionID(args),
		errMsg:           errMsg,
		projectPath:      projectPath,
		timestampRFC3339: time.Now().Format(time.RFC3339),
	}
	queryBufMu.Lock()
	queryBuf = append(queryBuf, r)
	n := len(queryBuf)
	queryBufMu.Unlock()
	if n >= queryLogFlushSize {
		kickFlush(queryFlushKick)
	}
}

// FlushWriteBuffers commits buffered analytics (for tests or graceful shutdown).
func FlushWriteBuffers() {
	flushQueryLogBuffer()
	flushSessionLogBuffer()
	flushTrailBuffer()
}
