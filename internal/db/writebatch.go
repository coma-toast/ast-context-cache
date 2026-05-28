package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// Execer matches *sql.DB and *sql.Tx for Exec.
type Execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}

const (
	queryLogFlushInterval = 4 * time.Second
	queryLogFlushSize     = 120
	sessionFlushInterval  = 3 * time.Second
	sessionFlushSize      = 200
)

type queryLogRow struct {
	toolName           string
	argsJSON           string
	resultChars        int
	inputTokens        int
	outputTokens       int
	tokensSaved        int
	fileBaselineTokens int
	fullBaselineTokens int
	durationMs         float64
	cpuMs              float64
	sessionID          string
	errMsg             string
	projectPath        string
	timestampRFC3339   string
}

type sessionLogRow struct {
	sessionID  string
	symbolID   int
	filePath   string
	mode       string
	tokenCount int
}

// QueryLogSnapshot is a flushed analytics row for dashboard toasts / live updates.
type QueryLogSnapshot struct {
	Timestamp   string
	ToolName    string
	ArgsJSON    string
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
)

// StartWriteBatchers starts periodic flush of buffered query/session analytics rows.
func StartWriteBatchers() {
	go func() {
		t := time.NewTicker(queryLogFlushInterval)
		defer t.Stop()
		for range t.C {
			flushQueryLogBuffer()
		}
	}()
	go func() {
		t := time.NewTicker(sessionFlushInterval)
		defer t.Stop()
		for range t.C {
			flushSessionLogBuffer()
		}
	}()
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
		log.Printf("query log batch: begin: %v", err)
		return
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(`INSERT INTO queries (timestamp, tool_name, arguments, result_chars, input_tokens, output_tokens, tokens_saved, file_baseline_tokens, full_baseline_tokens, duration_ms, cpu_ms, interface, session_id, error, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		log.Printf("query log batch: prepare: %v", err)
		return
	}
	defer stmt.Close()
	for _, r := range batch {
		if _, err := stmt.Exec(r.timestampRFC3339, r.toolName, r.argsJSON, r.resultChars, r.inputTokens, r.outputTokens, r.tokensSaved, r.fileBaselineTokens, r.fullBaselineTokens, r.durationMs, r.cpuMs, "http", r.sessionID, r.errMsg, r.projectPath); err != nil {
			log.Printf("query log batch: insert: %v", err)
		}
	}
	if err := tx.Commit(); err != nil {
		log.Printf("query log batch: commit: %v", err)
		return
	}
	if AfterQueryLogFlush != nil {
		snap := make([]QueryLogSnapshot, len(batch))
		for i, r := range batch {
			snap[i] = QueryLogSnapshot{
				Timestamp:   r.timestampRFC3339,
				ToolName:    r.toolName,
				ArgsJSON:    r.argsJSON,
				TokensSaved: r.tokensSaved,
				DurationMs:  r.durationMs,
				CpuMs:       r.cpuMs,
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
		log.Printf("session log batch: begin: %v", err)
		return
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(`INSERT INTO sessions (session_id, symbol_id, file_path, mode, token_count) VALUES (?, ?, ?, ?, ?)`)
	if err != nil {
		log.Printf("session log batch: prepare: %v", err)
		return
	}
	defer stmt.Close()
	for _, r := range batch {
		if _, err := stmt.Exec(r.sessionID, r.symbolID, r.filePath, r.mode, r.tokenCount); err != nil {
			log.Printf("session log batch: insert: %v", err)
		}
	}
	if err := tx.Commit(); err != nil {
		log.Printf("session log batch: commit: %v", err)
	}
}

// EnqueueSessionReturned buffers a session dedup row; flushed periodically in batches.
func EnqueueSessionReturned(sessionID string, symbolID int, filePath, mode string, tokenCount int) {
	if sessionID == "" {
		return
	}
	sessBufMu.Lock()
	sessBuf = append(sessBuf, sessionLogRow{sessionID: sessionID, symbolID: symbolID, filePath: filePath, mode: mode, tokenCount: tokenCount})
	n := len(sessBuf)
	sessBufMu.Unlock()
	if n >= sessionFlushSize {
		go flushSessionLogBuffer()
	}
}

func enqueueQueryLog(toolName string, args map[string]interface{}, resultChars, inputTokens, outputTokens, tokensSaved, fileBaselineTokens, fullBaselineTokens int, durationMs, cpuMs float64, projectPath, errMsg string) {
	argsJSON, _ := json.Marshal(args)
	sessionID := fmt.Sprintf("session-%d", time.Now().Unix()/3600)
	r := queryLogRow{
		toolName:           toolName,
		argsJSON:           string(argsJSON),
		resultChars:        resultChars,
		inputTokens:        inputTokens,
		outputTokens:       outputTokens,
		tokensSaved:        tokensSaved,
		fileBaselineTokens: fileBaselineTokens,
		fullBaselineTokens: fullBaselineTokens,
		durationMs:         durationMs,
		cpuMs:              cpuMs,
		sessionID:          sessionID,
		errMsg:             errMsg,
		projectPath:        projectPath,
		timestampRFC3339:   time.Now().Format(time.RFC3339),
	}
	queryBufMu.Lock()
	queryBuf = append(queryBuf, r)
	n := len(queryBuf)
	queryBufMu.Unlock()
	if n >= queryLogFlushSize {
		go flushQueryLogBuffer()
	}
}

// FlushWriteBuffers commits buffered analytics (for tests or graceful shutdown).
func FlushWriteBuffers() {
	flushQueryLogBuffer()
	flushSessionLogBuffer()
}
