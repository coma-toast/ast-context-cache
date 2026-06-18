package db

import (
	"database/sql"
	"log"
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

func GetDBPath() string {
	return dbPath()
}

// DefaultLogPath is the default ast-mcp server log file (ast-mcp start / dashboard Logs tab).
func DefaultLogPath() string {
	home := os.Getenv("HOME")
	if home == "" {
		return filepath.Join(".astcache", "ast-mcp.log")
	}
	return filepath.Join(home, ".astcache", "ast-mcp.log")
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
	DB.Exec(`PRAGMA synchronous=NORMAL`)
	DB.Exec(`PRAGMA cache_size=-32000`)

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
	DB.Exec(`ALTER TABLE symbols ADD COLUMN embed_hash TEXT`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN file_baseline_tokens INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN full_baseline_tokens INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN cpu_ms REAL DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN tokens_used INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN symbol_baseline_tokens INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN dedup_tokens_saved INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN savings_vs_files INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN deduped_count INTEGER DEFAULT 0`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN mode TEXT DEFAULT ''`)
	DB.Exec(`ALTER TABLE queries ADD COLUMN cache_hit INTEGER DEFAULT 0`)

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

	DB.Exec(`CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS agent_configs (
			id INTEGER PRIMARY KEY,
			agent_type TEXT NOT NULL,
			install_path TEXT NOT NULL,
			is_global INTEGER DEFAULT 0,
			instructions_hash TEXT,
			installed_at TEXT DEFAULT (datetime('now')),
			UNIQUE(agent_type, install_path)
		);
	`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS indexed_files (
			file TEXT NOT NULL,
			project_path TEXT NOT NULL,
			indexed_at DATETIME NOT NULL,
			PRIMARY KEY (file, project_path)
		);
	`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS doc_sources (
			id INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			type TEXT NOT NULL,
			url TEXT NOT NULL,
			version TEXT,
			last_updated TEXT,
			created_at TEXT DEFAULT (datetime('now')),
			UNIQUE(name, type, url)
		);
	`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS doc_content (
			id INTEGER PRIMARY KEY,
			source_id INTEGER NOT NULL,
			title TEXT NOT NULL,
			content TEXT NOT NULL,
			path TEXT,
			content_hash TEXT,
			updated_at TEXT DEFAULT (datetime('now')),
			FOREIGN KEY (source_id) REFERENCES doc_sources(id)
		);
		CREATE INDEX IF NOT EXISTS idx_doc_content_source ON doc_content(source_id);
		CREATE INDEX IF NOT EXISTS idx_doc_content_title ON doc_content(title);
	`)

	DB.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, content, content='doc_content', content_rowid='id')`)

	DB.Exec(`CREATE TRIGGER IF NOT EXISTS docs_fts_ins AFTER INSERT ON doc_content BEGIN
		INSERT INTO docs_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
	END`)
	DB.Exec(`CREATE TRIGGER IF NOT EXISTS docs_fts_del AFTER DELETE ON doc_content BEGIN
		INSERT INTO docs_fts(docs_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
	END`)

	DB.Exec(`
		CREATE TABLE IF NOT EXISTS context_notes (
			ref TEXT PRIMARY KEY,
			session_id TEXT NOT NULL,
			project_path TEXT,
			label TEXT,
			content TEXT NOT NULL,
			content_hash TEXT NOT NULL,
			tags TEXT,
			token_est INTEGER DEFAULT 0,
			access_count INTEGER DEFAULT 0,
			tokens_fetched INTEGER DEFAULT 0,
			last_accessed_at TEXT,
			created_at TEXT DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_context_notes_session ON context_notes(session_id);
		CREATE INDEX IF NOT EXISTS idx_context_notes_project ON context_notes(project_path);
	`)
	DB.Exec(`
		CREATE TABLE IF NOT EXISTS context_note_access (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ref TEXT NOT NULL,
			session_id TEXT,
			project_path TEXT,
			tool_name TEXT NOT NULL,
			virtual_tokens INTEGER NOT NULL,
			accessed_at TEXT DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_context_note_access_at ON context_note_access(accessed_at);
		CREATE INDEX IF NOT EXISTS idx_context_note_access_ref ON context_note_access(ref);
	`)
	DB.Exec(`
		CREATE TABLE IF NOT EXISTS context_session_stats (
			session_id TEXT PRIMARY KEY,
			project_path TEXT,
			notes_count INTEGER DEFAULT 0,
			virtual_tokens_stored INTEGER DEFAULT 0,
			virtual_tokens_accessed INTEGER DEFAULT 0,
			last_store_at TEXT,
			last_access_at TEXT
		);
	`)
	DB.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS context_notes_fts USING fts5(ref, session_id, label, content)`)
	DB.Exec(`
		CREATE TABLE IF NOT EXISTS structured_memory (
			ref TEXT PRIMARY KEY,
			kind TEXT NOT NULL,
			scope TEXT NOT NULL DEFAULT 'session',
			session_id TEXT,
			project_path TEXT,
			subject TEXT,
			predicate TEXT,
			object TEXT,
			rule TEXT,
			valid_from TEXT NOT NULL DEFAULT (datetime('now')),
			valid_until TEXT,
			superseded_by TEXT,
			source_ref TEXT,
			token_est INTEGER NOT NULL DEFAULT 0,
			access_count INTEGER DEFAULT 0,
			last_accessed_at TEXT,
			created_at TEXT DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_struct_mem_session ON structured_memory(session_id);
		CREATE INDEX IF NOT EXISTS idx_struct_mem_project ON structured_memory(project_path);
		CREATE INDEX IF NOT EXISTS idx_struct_mem_fact ON structured_memory(kind, subject, predicate, valid_until);
	`)
	DB.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS structured_memory_fts USING fts5(ref, subject, predicate, object, rule)`)
	DB.Exec(`
		CREATE TABLE IF NOT EXISTS memory_access (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ref TEXT NOT NULL,
			session_id TEXT,
			project_path TEXT,
			tool_name TEXT NOT NULL,
			tokens_returned INTEGER NOT NULL,
			accessed_at TEXT DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_memory_access_at ON memory_access(accessed_at);
	`)
	DB.Exec(`
		CREATE TABLE IF NOT EXISTS embed_pending (
			file TEXT NOT NULL,
			project_path TEXT NOT NULL,
			reason TEXT NOT NULL DEFAULT 'failed',
			updated_at INTEGER NOT NULL,
			PRIMARY KEY (file, project_path)
		);
		CREATE INDEX IF NOT EXISTS idx_embed_pending_project ON embed_pending(project_path);
	`)

	EnsureFTSTriggers()
	go DB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	StartWriteBatchers()
	return nil
}

func GetSetting(key, defaultValue string) string {
	var val string
	err := DB.QueryRow("SELECT value FROM settings WHERE key = ?", key).Scan(&val)
	if err != nil {
		return defaultValue
	}
	return val
}

func SetSetting(key, value string) error {
	_, err := DB.Exec("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", key, value)
	return err
}

func GetAllSettings() map[string]string {
	result := map[string]string{}
	rows, err := DB.Query("SELECT key, value FROM settings")
	if err != nil {
		return result
	}
	defer rows.Close()
	for rows.Next() {
		var k, v string
		rows.Scan(&k, &v)
		result[k] = v
	}
	return result
}

type AgentConfig struct {
	ID               int    `json:"id"`
	AgentType        string `json:"agent_type"`
	InstallPath      string `json:"install_path"`
	IsGlobal         bool   `json:"is_global"`
	InstructionsHash string `json:"instructions_hash"`
	InstalledAt      string `json:"installed_at"`
}

func GetAgentConfigs() ([]AgentConfig, error) {
	rows, err := DB.Query("SELECT id, agent_type, install_path, is_global, instructions_hash, installed_at FROM agent_configs ORDER BY agent_type")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var configs []AgentConfig
	for rows.Next() {
		var c AgentConfig
		var isGlobal int
		rows.Scan(&c.ID, &c.AgentType, &c.InstallPath, &isGlobal, &c.InstructionsHash, &c.InstalledAt)
		c.IsGlobal = isGlobal == 1
		configs = append(configs, c)
	}
	return configs, nil
}

func AddAgentConfig(agentType, installPath string, isGlobal bool, hash string) error {
	_, err := DB.Exec(`INSERT INTO agent_configs (agent_type, install_path, is_global, instructions_hash) VALUES (?, ?, ?, ?)
		ON CONFLICT(agent_type, install_path) DO UPDATE SET instructions_hash = excluded.instructions_hash, installed_at = datetime('now')`,
		agentType, installPath, map[bool]int{true: 1, false: 0}[isGlobal], hash)
	return err
}

func RemoveAgentConfig(agentType, installPath string) error {
	_, err := DB.Exec("DELETE FROM agent_configs WHERE agent_type = ? AND install_path = ?", agentType, installPath)
	return err
}

func EnsureFTSTriggers() {
	DB.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_ins AFTER INSERT ON symbols BEGIN
		INSERT INTO symbols_fts(rowid, name, fqn, code) VALUES (new.id, new.name, new.fqn, new.code);
	END`)
	DB.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_del AFTER DELETE ON symbols BEGIN
		INSERT INTO symbols_fts(symbols_fts, rowid, name, fqn, code) VALUES('delete', old.id, old.name, old.fqn, old.code);
	END`)
}

func LogQuery(toolName string, args map[string]interface{}, m QueryLogMetrics, projectPath, errMsg string) {
	enqueueQueryLog(toolName, args, m, projectPath, errMsg)
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

func UpsertIndexedFile(file, projectPath string, indexedAt time.Time) {
	_ = UpsertIndexedFileWith(DB, file, projectPath, indexedAt)
}

// UpsertIndexedFileWith writes indexed_files using the given executor (e.g. within a transaction).
func UpsertIndexedFileWith(e Execer, file, projectPath string, indexedAt time.Time) error {
	_, err := e.Exec(`INSERT INTO indexed_files (file, project_path, indexed_at) VALUES (?, ?, ?)
		ON CONFLICT(file, project_path) DO UPDATE SET indexed_at = excluded.indexed_at`,
		file, projectPath, indexedAt.Format(time.RFC3339))
	return err
}

func GetIndexedFiles(projectPath string) map[string]time.Time {
	result := map[string]time.Time{}
	rows, err := DB.Query("SELECT file, indexed_at FROM indexed_files WHERE project_path = ?", projectPath)
	if err != nil {
		return result
	}
	defer rows.Close()
	for rows.Next() {
		var file, ts string
		rows.Scan(&file, &ts)
		if t, err := time.Parse(time.RFC3339, ts); err == nil {
			result[file] = t
		}
	}
	return result
}

func DeleteIndexedFile(file, projectPath string) {
	DB.Exec("DELETE FROM indexed_files WHERE file = ? AND project_path = ?", file, projectPath)
}

func walFileBytes() int64 {
	fi, err := os.Stat(dbPath() + "-wal")
	if err != nil {
		return 0
	}
	return fi.Size()
}

// CheckpointWAL flushes the WAL. truncate=true forces pages back into the main db file.
func CheckpointWAL(truncate bool) (busy, walFrames, checkpointed int, err error) {
	mode := "PASSIVE"
	if truncate {
		mode = "TRUNCATE"
	}
	row := DB.QueryRow("PRAGMA wal_checkpoint(" + mode + ")")
	err = row.Scan(&busy, &walFrames, &checkpointed)
	return
}

func StartWALCheckpoint() {
	if wal := walFileBytes(); wal > 100*1024*1024 {
		log.Printf("Large WAL at startup (%d MB) — truncating", wal/(1024*1024))
		if busy, frames, ckpt, err := CheckpointWAL(true); err == nil {
			log.Printf("Startup WAL checkpoint: busy=%d log=%d checkpointed=%d", busy, frames, ckpt)
		}
	}
	walTicker := time.NewTicker(2 * time.Minute)
	truncateTicker := time.NewTicker(30 * time.Minute)
	vacuumTicker := time.NewTicker(24 * time.Hour)
	for {
		select {
		case <-walTicker.C:
			if walFileBytes() > 256*1024*1024 {
				if busy, frames, ckpt, err := CheckpointWAL(true); err == nil && (busy > 0 || frames > 0) {
					log.Printf("WAL checkpoint(TRUNCATE): busy=%d log=%d checkpointed=%d wal=%dMB", busy, frames, ckpt, walFileBytes()/(1024*1024))
				}
			} else {
				CheckpointWAL(false)
			}
		case <-truncateTicker.C:
			if busy, frames, ckpt, err := CheckpointWAL(true); err == nil && frames > 0 {
				log.Printf("Scheduled WAL checkpoint: busy=%d log=%d checkpointed=%d", busy, frames, ckpt)
			}
		case <-vacuumTicker.C:
			Compact()
		}
	}
}

func Compact() {
	log.Println("Running VACUUM...")
	start := time.Now()
	DB.Exec(`VACUUM`)
	log.Printf("VACUUM completed in %v", time.Since(start))
}
