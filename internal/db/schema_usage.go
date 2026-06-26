package db

import "database/sql"

func initUsageSchema(conn *sql.DB) {
	conn.Exec(`
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
	`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN file_baseline_tokens INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN full_baseline_tokens INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN cpu_ms REAL DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN tokens_used INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN symbol_baseline_tokens INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN dedup_tokens_saved INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN savings_vs_files INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN deduped_count INTEGER DEFAULT 0`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN mode TEXT DEFAULT ''`)
	conn.Exec(`ALTER TABLE queries ADD COLUMN cache_hit INTEGER DEFAULT 0`)

	conn.Exec(`
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
	conn.Exec(`ALTER TABLE sessions ADD COLUMN symbol_name TEXT DEFAULT ''`)
	conn.Exec(`ALTER TABLE sessions ADD COLUMN start_line INTEGER DEFAULT 0`)

	conn.Exec(`CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)`)

	conn.Exec(`
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

	conn.Exec(`
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
	conn.Exec(`ALTER TABLE context_note_access ADD COLUMN repair_reason TEXT DEFAULT ''`)
	conn.Exec(`ALTER TABLE context_note_access ADD COLUMN metadata_json TEXT DEFAULT ''`)

	conn.Exec(`
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

	conn.Exec(`
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
}
