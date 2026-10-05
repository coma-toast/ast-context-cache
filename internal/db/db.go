package db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/startup"
)

const (
	selectSettingQuery      = "SELECT value FROM settings WHERE key = ?"
	upsertSettingQuery      = "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
	selectAllSettingsQuery  = "SELECT key, value FROM settings"
	selectAgentConfigsQuery = "SELECT id, agent_type, install_path, is_global, instructions_hash, installed_at FROM agent_configs ORDER BY agent_type"
	upsertAgentConfigQuery  = `INSERT INTO agent_configs (agent_type, install_path, is_global, instructions_hash) VALUES (?, ?, ?, ?)
		ON CONFLICT(agent_type, install_path) DO UPDATE SET instructions_hash = excluded.instructions_hash, installed_at = datetime('now')`
	deleteAgentConfigQuery = "DELETE FROM agent_configs WHERE agent_type = ? AND install_path = ?"
	upsertIndexedFileQuery = `INSERT INTO indexed_files (file, project_path, indexed_at, parser_version) VALUES (?, ?, ?, ?)
		ON CONFLICT(file, project_path) DO UPDATE SET indexed_at = excluded.indexed_at, parser_version = excluded.parser_version`
	selectIndexedFilesQuery = "SELECT file, indexed_at, COALESCE(parser_version, 0) FROM indexed_files WHERE project_path = ?"
	deleteIndexedFileQuery  = "DELETE FROM indexed_files WHERE file = ? AND project_path = ?"
	vacuumQuery             = `VACUUM`
)

// DefaultLogPath is the default ast-mcp server log file (ast-mcp start / dashboard Logs tab).
func DefaultLogPath() string {
	home := os.Getenv("HOME")
	if home == "" {
		return filepath.Join(".astcache", "ast-mcp.log")
	}
	return filepath.Join(home, ".astcache", "ast-mcp.log")
}

func Init() error {
	if _, err := ResolveDataDir(); err != nil {
		p := locationOverridePath()
		return errs.WrapMessage(fmt.Sprintf("configured data directory unavailable (reconnect the drive, or delete %s to use the default location)", p), err, "override_path", p)
	}
	idxPath := indexDBPath()
	ctxPath := contextDBPath()
	usePath := usageDBPath()
	if err := os.MkdirAll(cacheDir(), 0755); err != nil {
		return err
	}
	// Once per process, before any pool opens: sweep zero-byte legacy DB files.
	removeEmptyLegacyDBs(cacheDir())
	if needsSplitMigration(usePath, idxPath) {
		startup.SetMessage("Migrating database to split layout…")
		if err := migrateSplitDB(usePath, idxPath, ctxPath); err != nil {
			return err
		}
	}
	startup.SetMessage("Opening databases…")
	// Not ready while the pools are reassigned; re-checked on every return below,
	// including the errors, which leave some of them open.
	poolsOpen.Store(false)
	defer syncPoolsOpen()
	// A previous Init's batchers read DB; stop them before it's reassigned.
	stopWriteBatchers()
	var err error
	IndexDB, err = openPool(idxPath)
	if err != nil {
		return fmtOpenErr("index", idxPath, err)
	}
	ContextDB, err = openPool(ctxPath)
	if err != nil {
		return fmtOpenErr("context", ctxPath, err)
	}
	DB, err = openPool(usePath)
	if err != nil {
		return fmtOpenErr("usage", usePath, err)
	}
	initIndexSchema(IndexDB)
	initUsageSchema(DB)
	initContextSchema(ContextDB)
	if err := createFTSTriggers(IndexDB); err != nil {
		// Not fatal: StartFTSSelfCheck retries, and search still works on whatever
		// the indexes already hold.
		logger.Warn("Failed to create FTS triggers", "error", err)
	}
	startIndexWriter()
	StartWriteBatchers()
	startFTSRebuild(IndexDB)
	return nil
}

func GetSetting(key, defaultValue string) string {
	if DB == nil {
		return defaultValue
	}
	var val string
	err := DB.QueryRow(selectSettingQuery, key).Scan(&val)
	if err != nil {
		return defaultValue
	}
	return val
}

func SetSetting(key, value string) error {
	_, err := DB.Exec(upsertSettingQuery, key, value)
	return err
}

func GetAllSettings() map[string]string {
	result := map[string]string{}
	if DB == nil {
		return result
	}
	rows, err := DB.Query(selectAllSettingsQuery)
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
	if DB == nil {
		return nil, nil
	}
	rows, err := DB.Query(selectAgentConfigsQuery)
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
	_, err := DB.Exec(upsertAgentConfigQuery, agentType, installPath, map[bool]int{true: 1, false: 0}[isGlobal], hash)
	return err
}

func RemoveAgentConfig(agentType, installPath string) error {
	_, err := DB.Exec(deleteAgentConfigQuery, agentType, installPath)
	return err
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

// QualifiedName returns a symbol's in-file qualified name — Class.method for a
// member, the bare name for a top-level symbol — from the fqn the indexer stores
// as "<file basename>.<qualified name>". Any other fqn shape (plaintext rows use
// "<path>#plaintext") yields name, so callers can treat q != name as "member".
func QualifiedName(fqn, file, name string) string {
	if q, ok := strings.CutPrefix(fqn, filepath.Base(file)+"."); ok && q != "" {
		return q
	}
	return name
}

func UpsertIndexedFile(file, projectPath string, indexedAt time.Time) {
	_ = IndexWrite(func(tx *sql.Tx) error {
		return UpsertIndexedFileWith(tx, file, projectPath, indexedAt)
	})
}

// ParserVersion reports the symbol-extractor version a file would be indexed
// with today. The indexer installs it (db cannot import indexer); files whose
// indexed_files row carries an older version are treated as stale.
var ParserVersion = func(file string) int { return 0 }

// UpsertIndexedFileWith writes indexed_files using the given executor (e.g. within a transaction).
func UpsertIndexedFileWith(e Execer, file, projectPath string, indexedAt time.Time) error {
	_, err := e.Exec(upsertIndexedFileQuery, file, projectPath, indexedAt.Format(time.RFC3339), ParserVersion(file))
	return err
}

// GetIndexedFiles maps each indexed file to when it was indexed. A file indexed
// by an older parser (see ParserVersion) maps to the zero time so mtime-based
// catch-up re-indexes it even though it hasn't changed on disk.
func GetIndexedFiles(projectPath string) map[string]time.Time {
	result := map[string]time.Time{}
	conn, err := IndexReader()
	if err != nil {
		return result
	}
	rows, err := conn.Query(selectIndexedFilesQuery, projectPath)
	if err != nil {
		return result
	}
	defer rows.Close()
	for rows.Next() {
		var file, ts string
		var version int
		rows.Scan(&file, &ts, &version)
		if t, err := time.Parse(time.RFC3339, ts); err == nil {
			if version < ParserVersion(file) {
				t = time.Time{}
			}
			result[file] = t
		}
	}
	return result
}

func DeleteIndexedFile(file, projectPath string) {
	_ = IndexWrite(func(tx *sql.Tx) error {
		_, err := tx.Exec(deleteIndexedFileQuery, file, projectPath)
		return err
	})
}

// FormatFileSize formats a byte count for dashboard display.
func FormatFileSize(bytes int64) string {
	switch {
	case bytes >= 1024*1024*1024:
		return fmt.Sprintf("%.2f GB", float64(bytes)/(1024*1024*1024))
	case bytes >= 1024*1024:
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1024*1024))
	case bytes >= 1024:
		return fmt.Sprintf("%d KB", bytes/1024)
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// deferredStartupWALCheckpoint runs after init so ForceCheckpointWAL does not contend with embedder startup.
func deferredStartupWALCheckpoint(walAtStart int64) {
	time.Sleep(2 * time.Minute)
	indexWal := IndexWalBytes()
	if indexWal <= walTruncateBytes {
		return
	}
	logger.Info("Running deferred startup WAL checkpoint", "index_wal", FormatFileSize(indexWal), "boot_index_wal", FormatFileSize(walAtStart))
	maintainWAL("startup", true)
}

func StartWALCheckpoint() {
	if wal := IndexWalBytes(); wal > walTruncateBytes {
		logger.Info("Large index WAL at startup, deferring checkpoint until server is up", "index_wal", FormatFileSize(wal))
		go deferredStartupWALCheckpoint(wal)
	}
	passiveTicker := time.NewTicker(30 * time.Second)
	maintTicker := time.NewTicker(90 * time.Second)
	truncateTicker := time.NewTicker(30 * time.Minute)
	vacuumTicker := time.NewTicker(24 * time.Hour)
	retentionTicker := time.NewTicker(24 * time.Hour)
	go func() {
		time.Sleep(90 * time.Second)
		retryQueryRetention("startup")
	}()
	for {
		select {
		case <-passiveTicker.C:
			if wal := IndexWalBytes(); wal >= walPassiveBytes && wal <= walTruncateBytes {
				runPassiveCheckpoint()
			}
		case <-maintTicker.C:
			runWALMaintenanceCycle("periodic")
		case <-truncateTicker.C:
			wal := IndexWalBytes()
			if wal <= walTruncateBytes {
				continue
			}
			if shouldForceCheckpoint(wal) {
				maintainWAL("scheduled", true)
			} else {
				runWALMaintenanceCycle("scheduled")
			}
		case <-retentionTicker.C:
			retryQueryRetention("daily")
		case <-vacuumTicker.C:
			Compact()
		}
	}
}

var (
	compactMu       sync.Mutex
	compactRunning  bool
	compactPending  bool
	compactRunCount int // test-only observability; incremented under compactMu
)

// Compact runs VACUUM on all three databases. Several independent callers can trigger
// this concurrently (deleting/resetting more than one project fires one goroutine each,
// plus the manual prune action and the daily ticker) — VACUUM is exclusive and expensive,
// and running several at once against the same files causes cascading "database is
// locked" errors and runaway WAL growth. But a plain "skip if already running" guard
// silently drops work: if project A's delete arrives while project B's compaction is
// already mid-VACUUM, A's freed pages never get reclaimed unless something runs again
// after B finishes. So a concurrent call instead marks a pending flag; the in-progress
// run checks it after finishing and does one more pass if anything came in meanwhile,
// coalescing any number of concurrent callers into at most two actual VACUUM runs.
func Compact() {
	compactMu.Lock()
	if compactRunning {
		compactPending = true
		compactMu.Unlock()
		logger.Info("VACUUM already in progress, will run again once it finishes")
		return
	}
	compactRunning = true
	compactMu.Unlock()

	// Reuses the WAL-maintenance status/banner plumbing so the dashboard shows a busy
	// indicator instead of index/vector counts silently reading as zero while the
	// databases are quiesced for VACUUM.
	beginWALMaintenance("vacuum")
	setWALPhase(WALPhaseVacuum, "")
	defer endWALMaintenance()

	for {
		runCompactOnce()
		compactMu.Lock()
		compactRunCount++
		if !compactPending {
			compactRunning = false
			compactMu.Unlock()
			return
		}
		compactPending = false
		compactMu.Unlock()
	}
}

func runCompactOnce() {
	logger.Info("Running VACUUM on index, context, and usage databases")
	start := time.Now()
	for _, c := range []*sql.DB{IndexDB, ContextDB, DB} {
		if c != nil {
			c.Exec(vacuumQuery)
		}
	}
	logger.Info("VACUUM completed", "duration", time.Since(start))
}
