package db

import "database/sql"

func initIndexSchema(conn *sql.DB) {
	conn.Exec(`
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
	conn.Exec(`ALTER TABLE symbols ADD COLUMN skeleton TEXT`)
	conn.Exec(`ALTER TABLE symbols ADD COLUMN embed_hash TEXT`)

	conn.Exec(`
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

	conn.Exec(`CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(name, fqn, code, content='symbols', content_rowid='id')`)

	conn.Exec(`
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

	conn.Exec(`
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

	conn.Exec(`
		CREATE TABLE IF NOT EXISTS indexed_files (
			file TEXT NOT NULL,
			project_path TEXT NOT NULL,
			indexed_at DATETIME NOT NULL,
			PRIMARY KEY (file, project_path)
		);
	`)

	conn.Exec(`
		CREATE TABLE IF NOT EXISTS embed_pending (
			file TEXT NOT NULL,
			project_path TEXT NOT NULL,
			reason TEXT NOT NULL DEFAULT 'failed',
			updated_at INTEGER NOT NULL,
			PRIMARY KEY (file, project_path)
		);
		CREATE INDEX IF NOT EXISTS idx_embed_pending_project ON embed_pending(project_path);
	`)
}

func ensureIndexFTSTriggers(conn *sql.DB) {
	conn.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_ins AFTER INSERT ON symbols BEGIN
		INSERT INTO symbols_fts(rowid, name, fqn, code) VALUES (new.id, new.name, new.fqn, new.code);
	END`)
	conn.Exec(`CREATE TRIGGER IF NOT EXISTS symbols_fts_del AFTER DELETE ON symbols BEGIN
		INSERT INTO symbols_fts(symbols_fts, rowid, name, fqn, code) VALUES('delete', old.id, old.name, old.fqn, old.code);
	END`)
}
