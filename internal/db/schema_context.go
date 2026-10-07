package db

import "database/sql"

const (
	createDocSourcesTable = `
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
	`
	createDocContentTable = `
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
	`
	createDocsFTSTable         = `CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, content, content='doc_content', content_rowid='id')`
	createDocsFTSInsertTrigger = `CREATE TRIGGER IF NOT EXISTS docs_fts_ins AFTER INSERT ON doc_content BEGIN
		INSERT INTO docs_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
	END`
	createDocsFTSDeleteTrigger = `CREATE TRIGGER IF NOT EXISTS docs_fts_del AFTER DELETE ON doc_content BEGIN
		INSERT INTO docs_fts(docs_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
	END`
	createContextNotesTable = `
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
	`
	addContextNotesKindColumn         = `ALTER TABLE context_notes ADD COLUMN kind TEXT DEFAULT ''`
	addContextNotesMetadataJSONColumn = `ALTER TABLE context_notes ADD COLUMN metadata_json TEXT DEFAULT ''`
	addContextNotesRevisionColumn     = `ALTER TABLE context_notes ADD COLUMN revision INTEGER DEFAULT 1`
	createContextNotesFTSTable        = `CREATE VIRTUAL TABLE IF NOT EXISTS context_notes_fts USING fts5(ref, session_id, label, content)`
	// Superseded bodies only: context_notes holds the live revision, this table holds
	// every body an edit replaced, so edit_context can revert without keeping the
	// current copy twice.
	createContextNoteRevisionsTable = `
		CREATE TABLE IF NOT EXISTS context_note_revisions (
			ref TEXT NOT NULL,
			revision INTEGER NOT NULL,
			content TEXT NOT NULL,
			token_est INTEGER DEFAULT 0,
			op TEXT DEFAULT '',
			created_at TEXT DEFAULT (datetime('now')),
			PRIMARY KEY (ref, revision)
		);
		CREATE INDEX IF NOT EXISTS idx_context_note_revisions_ref ON context_note_revisions(ref);
	`
	// Reusable context functions: the model-defined half of Context Language Models
	// (arXiv 2609.37725). The paper's agent writes a Python function into its own
	// context file and invokes it dozens of times per trace; here the function is a
	// named (pattern, replacement) pair in SQLite, which is the subset that is
	// auditable and cannot smuggle executable code into the store.
	createContextFnsTable = `
		CREATE TABLE IF NOT EXISTS context_fns (
			name TEXT PRIMARY KEY,
			project_path TEXT,
			session_id TEXT,
			description TEXT DEFAULT '',
			pattern TEXT NOT NULL,
			replacement TEXT NOT NULL DEFAULT '',
			pattern_hash TEXT NOT NULL DEFAULT '',
			version INTEGER NOT NULL DEFAULT 1,
			call_count INTEGER NOT NULL DEFAULT 0,
			notes_touched INTEGER NOT NULL DEFAULT 0,
			tokens_reclaimed INTEGER NOT NULL DEFAULT 0,
			created_at TEXT DEFAULT (datetime('now')),
			updated_at TEXT DEFAULT (datetime('now')),
			retired_at TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_context_fns_project ON context_fns(project_path);
		CREATE INDEX IF NOT EXISTS idx_context_fns_session ON context_fns(session_id);
	`
	createKVRepairEventsTable = `
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
	`
	createStructuredMemoryTable = `
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
	`
	createStructuredMemoryFTSTable = `CREATE VIRTUAL TABLE IF NOT EXISTS structured_memory_fts USING fts5(ref, subject, predicate, object, rule)`
	// Handoff trees. Timestamps are written by the handoff package as UTC "YYYY-MM-DD HH:MM:SS"
	// (datetime('now') format) so they compare correctly against each other and SQLite's clock.
	createHandoffTreesTable = `
		CREATE TABLE IF NOT EXISTS handoff_trees (
			tree_id TEXT PRIMARY KEY,
			root_session_id TEXT NOT NULL,
			project_path TEXT,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			last_access_at TEXT NOT NULL DEFAULT (datetime('now')),
			tokens_used INTEGER NOT NULL DEFAULT 0,
			entries_used INTEGER NOT NULL DEFAULT 0
		);
		CREATE INDEX IF NOT EXISTS idx_handoff_trees_root ON handoff_trees(root_session_id);
	`
	createHandoffsTable = `
		CREATE TABLE IF NOT EXISTS handoffs (
			ref TEXT PRIMARY KEY,
			tree_id TEXT NOT NULL,
			parent_session_id TEXT NOT NULL,
			parent_child_session_id TEXT,
			depth INTEGER NOT NULL DEFAULT 1,
			mode TEXT NOT NULL DEFAULT 'fresh',
			label TEXT,
			brief TEXT NOT NULL,
			project_path TEXT,
			child_count INTEGER NOT NULL DEFAULT 0,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			last_access_at TEXT NOT NULL DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_handoffs_parent ON handoffs(parent_session_id);
		CREATE INDEX IF NOT EXISTS idx_handoffs_tree ON handoffs(tree_id);
	`
	createHandoffSnapshotItemsTable = `
		CREATE TABLE IF NOT EXISTS handoff_snapshot_items (
			id INTEGER PRIMARY KEY,
			handoff_ref TEXT NOT NULL,
			section TEXT NOT NULL,
			ord INTEGER NOT NULL DEFAULT 0,
			item_key TEXT,
			label TEXT,
			content TEXT,
			file_rel TEXT,
			fqn TEXT,
			kind TEXT,
			start_line INTEGER,
			end_line INTEGER,
			fingerprint TEXT,
			token_est INTEGER NOT NULL DEFAULT 0
		);
		CREATE INDEX IF NOT EXISTS idx_handoff_snapshot_items_section ON handoff_snapshot_items(handoff_ref, section, ord);
	`
	createHandoffChildrenTable = `
		CREATE TABLE IF NOT EXISTS handoff_children (
			child_session_id TEXT PRIMARY KEY,
			handoff_ref TEXT NOT NULL,
			tree_id TEXT NOT NULL,
			label TEXT,
			status TEXT NOT NULL DEFAULT 'open',
			project_path TEXT,
			opened_at TEXT NOT NULL DEFAULT (datetime('now')),
			last_activity_at TEXT NOT NULL DEFAULT (datetime('now')),
			result_ref TEXT,
			summary TEXT,
			summary_source TEXT,
			summary_truncated INTEGER NOT NULL DEFAULT 0,
			result_status TEXT,
			search_calls INTEGER NOT NULL DEFAULT 0,
			repeat_calls INTEGER NOT NULL DEFAULT 0,
			tokens_available INTEGER NOT NULL DEFAULT 0,
			tokens_delivered INTEGER NOT NULL DEFAULT 0
		);
		CREATE INDEX IF NOT EXISTS idx_handoff_children_tree_status ON handoff_children(tree_id, status);
	`
	createHandoffResultsTable = `
		CREATE TABLE IF NOT EXISTS handoff_results (
			id INTEGER PRIMARY KEY,
			child_session_id TEXT NOT NULL,
			result_ref TEXT NOT NULL,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			superseded_at TEXT
		);
	`
	createScratchpadEntriesTable = `
		CREATE TABLE IF NOT EXISTS scratchpad_entries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			tree_id TEXT NOT NULL,
			author_session_id TEXT NOT NULL,
			type TEXT NOT NULL,
			text TEXT NOT NULL,
			refs_json TEXT,
			token_est INTEGER NOT NULL DEFAULT 0,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			retracted_at TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_scratchpad_entries_tree ON scratchpad_entries(tree_id, id);
	`
	createHandoffClaimsTable = `
		CREATE TABLE IF NOT EXISTS handoff_claims (
			tree_id TEXT NOT NULL,
			key TEXT NOT NULL,
			holder_session_id TEXT NOT NULL,
			reason TEXT,
			granted_at TEXT NOT NULL DEFAULT (datetime('now')),
			PRIMARY KEY (tree_id, key)
		);
	`
	createHandoffClaimQueueTable = `
		CREATE TABLE IF NOT EXISTS handoff_claim_queue (
			id INTEGER PRIMARY KEY,
			tree_id TEXT NOT NULL,
			key TEXT NOT NULL,
			session_id TEXT NOT NULL,
			reason TEXT,
			enqueued_at TEXT NOT NULL DEFAULT (datetime('now'))
		);
		CREATE INDEX IF NOT EXISTS idx_handoff_claim_queue_key ON handoff_claim_queue(tree_id, key, id);
	`
	createHandoffClaimGrantsTable = `
		CREATE TABLE IF NOT EXISTS handoff_claim_grants (
			id INTEGER PRIMARY KEY,
			session_id TEXT NOT NULL,
			tree_id TEXT NOT NULL,
			key TEXT NOT NULL,
			granted_at TEXT NOT NULL DEFAULT (datetime('now')),
			notified_at TEXT
		);
	`
)

func initContextSchema(conn *sql.DB) {
	conn.Exec(createDocSourcesTable)

	conn.Exec(createDocContentTable)

	conn.Exec(createDocsFTSTable)
	conn.Exec(createDocsFTSInsertTrigger)
	conn.Exec(createDocsFTSDeleteTrigger)

	conn.Exec(createContextNotesTable)
	conn.Exec(addContextNotesKindColumn)
	conn.Exec(addContextNotesMetadataJSONColumn)
	conn.Exec(addContextNotesRevisionColumn)

	conn.Exec(createContextNotesFTSTable)

	conn.Exec(createContextNoteRevisionsTable)

	conn.Exec(createContextFnsTable)

	conn.Exec(createKVRepairEventsTable)

	conn.Exec(createStructuredMemoryTable)
	conn.Exec(createStructuredMemoryFTSTable)

	conn.Exec(createHandoffTreesTable)
	conn.Exec(createHandoffsTable)
	conn.Exec(createHandoffSnapshotItemsTable)
	conn.Exec(createHandoffChildrenTable)
	conn.Exec(createHandoffResultsTable)
	conn.Exec(createScratchpadEntriesTable)
	conn.Exec(createHandoffClaimsTable)
	conn.Exec(createHandoffClaimQueueTable)
	conn.Exec(createHandoffClaimGrantsTable)
}
