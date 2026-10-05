package db

import (
	"fmt"
	"os"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// InitUsage opens only the usage pool (settings, installer state) for short-lived CLI commands
// such as `ast-mcp install`. Unlike Init it starts no index writer, write batchers, or FTS
// rebuild, so it is safe to run beside a live server sharing the same WAL databases. When the
// one-time split migration is still pending it falls back to a full Init. Close releases it.
func InitUsage() error {
	if _, err := ResolveDataDir(); err != nil {
		p := locationOverridePath()
		return errs.WrapMessage(fmt.Sprintf("configured data directory unavailable (reconnect the drive, or delete %s to use the default location)", p), err, "override_path", p)
	}
	usePath := usageDBPath()
	if needsSplitMigration(usePath, indexDBPath()) {
		return Init()
	}
	if err := os.MkdirAll(cacheDir(), 0o755); err != nil {
		return err
	}
	conn, err := openPool(usePath)
	if err != nil {
		return fmtOpenErr("usage", usePath, err)
	}
	DB = conn
	initUsageSchema(DB)
	syncPoolsOpen()
	return nil
}
