package db

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/startup"
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
	idxPath := indexDBPath()
	ctxPath := contextDBPath()
	usePath := usageDBPath()
	if err := os.MkdirAll(cacheDir(), 0755); err != nil {
		return err
	}
	if needsSplitMigration(usePath, idxPath) {
		startup.SetMessage("Migrating database to split layout…")
		if err := migrateSplitDB(usePath, idxPath, ctxPath); err != nil {
			return err
		}
	}
	startup.SetMessage("Opening databases…")
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
	if err := openCheckpointPools(); err != nil {
		return err
	}
	initIndexSchema(IndexDB)
	initUsageSchema(DB)
	initContextSchema(ContextDB)
	ensureIndexFTSTriggers(IndexDB)
	startIndexWriter()
	StartWriteBatchers()
	go IndexDB.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	return nil
}

func GetSetting(key, defaultValue string) string {
	if DB == nil {
		return defaultValue
	}
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

// EnsureFTSTriggers ensures symbol FTS triggers on the index database.
func EnsureFTSTriggers() {
	ensureIndexFTSTriggers(IndexDB)
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
	_ = IndexWrite(func(tx *sql.Tx) error {
		return UpsertIndexedFileWith(tx, file, projectPath, indexedAt)
	})
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
	rows, err := IndexDB.Query("SELECT file, indexed_at FROM indexed_files WHERE project_path = ?", projectPath)
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
	_ = IndexWrite(func(tx *sql.Tx) error {
		_, err := tx.Exec("DELETE FROM indexed_files WHERE file = ? AND project_path = ?", file, projectPath)
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
	wal := walFileBytes()
	if wal <= walTruncateBytes {
		return
	}
	if wal >= walForceBytes {
		log.Printf("Startup deferred WAL force checkpoint (wal=%s, was %s at boot)", FormatFileSize(wal), FormatFileSize(walAtStart))
		ForceCheckpointWAL()
		return
	}
	if busy, frames, ckpt, err := CheckpointWAL(true); err == nil {
		log.Printf("Startup WAL checkpoint: busy=%d log=%d checkpointed=%d wal=%s", busy, frames, ckpt, FormatFileSize(WalFileBytes()))
	}
}

func StartWALCheckpoint() {
	if wal := walFileBytes(); wal > walTruncateBytes {
		log.Printf("Large WAL at startup (%s) — deferring checkpoint until server is up", FormatFileSize(wal))
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
			if wal := walFileBytes(); wal >= walPassiveBytes && wal <= walTruncateBytes {
				runPassiveCheckpoint()
			}
		case <-maintTicker.C:
			runWALMaintenanceCycle("periodic")
		case <-truncateTicker.C:
			wal := walFileBytes()
			if wal <= walTruncateBytes {
				continue
			}
			if maybeForceCheckpoint(wal, "scheduled") {
				continue
			}
			runWALMaintenanceCycle("scheduled")
		case <-retentionTicker.C:
			retryQueryRetention("daily")
		case <-vacuumTicker.C:
			Compact()
		}
	}
}

func Compact() {
	log.Println("Running VACUUM on index, context, and usage databases...")
	start := time.Now()
	for _, c := range []*sql.DB{IndexDB, ContextDB, DB} {
		if c != nil {
			c.Exec(`VACUUM`)
		}
	}
	log.Printf("VACUUM completed in %v", time.Since(start))
}
