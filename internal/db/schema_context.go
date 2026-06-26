package db

import "database/sql"

func initContextSchema(conn *sql.DB) {
	conn.Exec(`
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

	conn.Exec(`
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

	conn.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, content, content='doc_content', content_rowid='id')`)
	conn.Exec(`CREATE TRIGGER IF NOT EXISTS docs_fts_ins AFTER INSERT ON doc_content BEGIN
		INSERT INTO docs_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
	END`)
	conn.Exec(`CREATE TRIGGER IF NOT EXISTS docs_fts_del AFTER DELETE ON doc_content BEGIN
		INSERT INTO docs_fts(docs_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
	END`)

	conn.Exec(`
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
	conn.Exec(`ALTER TABLE context_notes ADD COLUMN kind TEXT DEFAULT ''`)
	conn.Exec(`ALTER TABLE context_notes ADD COLUMN metadata_json TEXT DEFAULT ''`)

	conn.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS context_notes_fts USING fts5(ref, session_id, label, content)`)

	conn.Exec(`
		CREATE TABLE IF NOT EXISTS kv_repair_events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT,
			project_path TEXT,
			ref TEXT,
			repair_reason TEXT NOT NULL,
			outcome TEXT,
			model_id TEXT,
			kv_quant TEXT,
			token_est INTEGER DEFAULT 0,
			detail TEXT,
			metadata_json TEXT,
			created_at TEXT DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_kv_repair_events_at ON kv_repair_events(created_at);
		CREATE INDEX IF NOT EXISTS idx_kv_repair_events_reason ON kv_repair_events(repair_reason);
	`)

	conn.Exec(`
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
	conn.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS structured_memory_fts USING fts5(ref, subject, predicate, object, rule)`)
}
