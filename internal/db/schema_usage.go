package db

import "database/sql"

const (
	createQueriesTable = `
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
	`
	addQueriesFileBaselineTokensColumn   = `ALTER TABLE queries ADD COLUMN file_baseline_tokens INTEGER DEFAULT 0`
	addQueriesFullBaselineTokensColumn   = `ALTER TABLE queries ADD COLUMN full_baseline_tokens INTEGER DEFAULT 0`
	addQueriesCPUMsColumn                = `ALTER TABLE queries ADD COLUMN cpu_ms REAL DEFAULT 0`
	addQueriesTokensUsedColumn           = `ALTER TABLE queries ADD COLUMN tokens_used INTEGER DEFAULT 0`
	addQueriesSymbolBaselineTokensColumn = `ALTER TABLE queries ADD COLUMN symbol_baseline_tokens INTEGER DEFAULT 0`
	addQueriesDedupTokensSavedColumn     = `ALTER TABLE queries ADD COLUMN dedup_tokens_saved INTEGER DEFAULT 0`
	addQueriesSavingsVsFilesColumn       = `ALTER TABLE queries ADD COLUMN savings_vs_files INTEGER DEFAULT 0`
	addQueriesDedupedCountColumn         = `ALTER TABLE queries ADD COLUMN deduped_count INTEGER DEFAULT 0`
	addQueriesModeColumn                 = `ALTER TABLE queries ADD COLUMN mode TEXT DEFAULT ''`
	addQueriesCacheHitColumn             = `ALTER TABLE queries ADD COLUMN cache_hit INTEGER DEFAULT 0`
	createSessionsTable                  = `
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
	`
	addSessionsSymbolNameColumn = `ALTER TABLE sessions ADD COLUMN symbol_name TEXT DEFAULT ''`
	addSessionsStartLineColumn  = `ALTER TABLE sessions ADD COLUMN start_line INTEGER DEFAULT 0`
	createSettingsTable         = `CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)`
	createAgentConfigsTable     = `
		CREATE TABLE IF NOT EXISTS agent_configs (
			id INTEGER PRIMARY KEY,
			agent_type TEXT NOT NULL,
			install_path TEXT NOT NULL,
			is_global INTEGER DEFAULT 0,
			instructions_hash TEXT,
			installed_at TEXT DEFAULT (datetime('now')),
			UNIQUE(agent_type, install_path)
		);
	`
	createContextNoteAccessTable = `
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
	`
	addContextNoteAccessRepairReasonColumn = `ALTER TABLE context_note_access ADD COLUMN repair_reason TEXT DEFAULT ''`
	addContextNoteAccessMetadataJSONColumn = `ALTER TABLE context_note_access ADD COLUMN metadata_json TEXT DEFAULT ''`
	createMemoryAccessTable                = `
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
	`
	createContextSessionStatsTable = `
		CREATE TABLE IF NOT EXISTS context_session_stats (
			session_id TEXT PRIMARY KEY,
			project_path TEXT,
			notes_count INTEGER DEFAULT 0,
			virtual_tokens_stored INTEGER DEFAULT 0,
			virtual_tokens_accessed INTEGER DEFAULT 0,
			last_store_at TEXT,
			last_access_at TEXT
		);
	`
	createProjectLinksTable = `
		CREATE TABLE IF NOT EXISTS project_links (
			parent_path TEXT NOT NULL,
			child_path TEXT NOT NULL,
			auto_linked INTEGER NOT NULL DEFAULT 1,
			created_at TEXT DEFAULT (datetime('now')),
			PRIMARY KEY (parent_path, child_path)
		);
		CREATE INDEX IF NOT EXISTS idx_project_links_parent ON project_links(parent_path);
		CREATE INDEX IF NOT EXISTS idx_project_links_child ON project_links(child_path);
	`
	// installer_state records what the v4 installer wrote, per target × component × path. created
	// is a bit set: 1 = the installer created the file, 2 = it created the containing block/object.
	createInstallerStateTable = `
		CREATE TABLE IF NOT EXISTS installer_state (
			target TEXT NOT NULL,
			component TEXT NOT NULL,
			path TEXT NOT NULL,
			entry_hash TEXT NOT NULL DEFAULT '',
			version TEXT NOT NULL DEFAULT '',
			created INTEGER NOT NULL DEFAULT 0,
			installed_at TEXT DEFAULT (datetime('now')),
			PRIMARY KEY (target, component, path)
		);
	`
)

func initUsageSchema(conn *sql.DB) {
	conn.Exec(createQueriesTable)
	conn.Exec(addQueriesFileBaselineTokensColumn)
	conn.Exec(addQueriesFullBaselineTokensColumn)
	conn.Exec(addQueriesCPUMsColumn)
	conn.Exec(addQueriesTokensUsedColumn)
	conn.Exec(addQueriesSymbolBaselineTokensColumn)
	conn.Exec(addQueriesDedupTokensSavedColumn)
	conn.Exec(addQueriesSavingsVsFilesColumn)
	conn.Exec(addQueriesDedupedCountColumn)
	conn.Exec(addQueriesModeColumn)
	conn.Exec(addQueriesCacheHitColumn)

	conn.Exec(createSessionsTable)
	conn.Exec(addSessionsSymbolNameColumn)
	conn.Exec(addSessionsStartLineColumn)

	conn.Exec(createSettingsTable)

	conn.Exec(createAgentConfigsTable)

	conn.Exec(createContextNoteAccessTable)
	conn.Exec(addContextNoteAccessRepairReasonColumn)
	conn.Exec(addContextNoteAccessMetadataJSONColumn)

	conn.Exec(createMemoryAccessTable)

	conn.Exec(createContextSessionStatsTable)
	conn.Exec(createProjectLinksTable)
	conn.Exec(createInstallerStateTable)
}
