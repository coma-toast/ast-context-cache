package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

var DB *sql.DB

const defaultDBPath = ".astcache/usage.db"

func dbPath() string {
	home := os.Getenv("HOME")
	if home == "" {
		return defaultDBPath
	}
	return filepath.Join(home, defaultDBPath)
}

func Init() error {
	p := dbPath()
	os.MkdirAll(filepath.Dir(p), 0755)
	var err error
	DB, err = sql.Open("sqlite3", p+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return err
	}

	DB.Exec(`PRAGMA journal_mode=WAL`)
	DB.Exec(`PRAGMA busy_timeout=5000`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS queries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			tool_name TEXT NOT NULL,
			arguments TEXT,
			result_chars INTEGER,
			input_tokens INTEGER DEFAULT 0,
			output_tokens INTEGER DEFAULT 0,
			tokens_saved INTEGER DEFAULT 0,
			duration_ms REAL,
			interface TEXT DEFAULT 'http',
			session_id TEXT,
			error TEXT,
			project_path TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_queries_project ON queries(project_path);
		CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
		CREATE TABLE IF NOT EXISTS symbols (
			id INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			kind TEXT NOT NULL,
			file TEXT NOT NULL,
			start_line INTEGER,
			end_line INTEGER,
			code TEXT,
			fqn TEXT,
			project_path TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
		CREATE INDEX IF NOT EXISTS idx_symbols_project ON symbols(project_path);
	`)

	// Migration: add skeleton column if missing
	DB.Exec(`ALTER TABLE symbols ADD COLUMN skeleton TEXT`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN file_baseline_tokens INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN full_baseline_tokens INTEGER DEFAULT 0`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS edges (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			source_file TEXT NOT NULL,
			source_symbol TEXT,
			target TEXT NOT NULL,
			kind TEXT NOT NULL,
			project_path TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
		CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_file);
		CREATE INDEX IF NOT EXISTS idx_edges_project ON edges(project_path);
	`)

	DB.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(name, fqn, code, content='symbols', content_rowid='id')`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS sessions (
			id INTEGER PRIMARY KEY,
			session_id TEXT NOT NULL,
			symbol_id INTEGER,
			file_path TEXT,
			returned_at TEXT DEFAULT (datetime('now')),
			mode TEXT,
			token_count INTEGER
		);
		CREATE INDEX IF NOT EXISTS idx_sessions_sid ON sessions(session_id);
	`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS summaries (
			id INTEGER PRIMARY KEY,
			file_path TEXT NOT NULL,
			symbol_name TEXT,
			summary_text TEXT NOT NULL,
			content_hash TEXT NOT NULL,
			project_path TEXT,
			created_at TEXT DEFAULT (datetime('now')),
			UNIQUE(file_path, symbol_name, project_path)
		);
	`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS vectors (
			id INTEGER PRIMARY KEY,
			symbol_id INTEGER,
			content_hash TEXT NOT NULL,
			vector BLOB NOT NULL,
			doc_type TEXT DEFAULT 'code',
			source_file TEXT,
			name TEXT,
			kind TEXT,
			project_path TEXT,
			created_at TEXT DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_vectors_project ON vectors(project_path);
		CREATE INDEX IF NOT EXISTS idx_vectors_file ON vectors(source_file);
		CREATE UNIQUE INDEX IF NOT EXISTS idx_vectors_hash ON vectors(content_hash, project_path);
	`)

	EnsureFTSTriggers()
	go DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)

	return nil
}

func EnsureFTSTriggers() {
	DB.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_ins AFTER INSERT ON symbols BEGIN
		INSERT INTO symbols_fts(rowid, name, fqn, code) VALUES (new.id, new.name, new.fqn, new.code);
	END`)
	DB.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_del AFTER DELETE ON symbols BEGIN
		INSERT INTO symbols_fts(symbols_fts, rowid, name, fqn, code) VALUES('delete', old.id, old.name, old.fqn, old.code);
	END`)
}

func LogQuery(toolName string, args map[string]interface{}, resultChars, inputTokens, outputTokens, tokensSaved, fileBaselineTokens, fullBaselineTokens int, durationMs float64, projectPath, errMsg string) {
	sessionID := fmt.Sprintf("session-%d", time.Now().Unix()/3600)
	argsJSON, _ := json.Marshal(args)
	DB.Exec("INSERT INTO queries (timestamp, tool_name, arguments, result_chars, input_tokens, output_tokens, tokens_saved, file_baseline_tokens, full_baseline_tokens, duration_ms, interface, session_id, error, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		time.Now().Format(time.RFC3339), toolName, string(argsJSON), resultChars, inputTokens, outputTokens, tokensSaved, fileBaselineTokens, fullBaselineTokens, durationMs, "http", sessionID, errMsg, projectPath)
}

func EstimateTokens(text string) int {
	return len(text) / 4
}

// RelPath strips the projectPath prefix from an absolute file path, returning
// a relative path for more compact (token-efficient) results.
func RelPath(file, projectPath string) string {
	if projectPath != "" && strings.HasPrefix(file, projectPath+"/") {
		return strings.TrimPrefix(file, projectPath+"/")
	}
	return file
}

func StartWALCheckpoint() {
	for range time.NewTicker(5 * time.Minute).C {
		DB.Exec(`PRAGMA wal_checkpoint(TRUNCATE)`)
	}
}
