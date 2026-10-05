package installer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Pre-4.0 installs were recorded in agent_configs. The v3 installer overwrote whole files (even
// ~/.claude.json, with markdown), so the first 4.0 run re-checks each record on disk and keeps
// the findings as warnings (IN-13). The table itself is kept for one release.
const (
	legacyMigratedSetting = "installer_legacy_migrated"
	legacyWarningsSetting = "installer_legacy_warnings"

	selectLegacyAgentConfigsQuery = "SELECT agent_type, install_path, is_global, COALESCE(installed_at, '') FROM agent_configs ORDER BY agent_type, install_path"
)

type legacyRecord struct {
	agentType   string
	installPath string
	global      bool
	installedAt string
}

// LegacyWarnings returns the stored findings of the legacy record check.
func (s *realService) LegacyWarnings() []string {
	var out []string
	if err := json.Unmarshal([]byte(db.GetSetting(legacyWarningsSetting, "[]")), &out); err != nil || out == nil {
		return []string{}
	}
	return out
}

// migrateLegacy runs the legacy check once per data directory. A failure is retried on the next start.
func (s *realService) migrateLegacy() {
	if db.DB == nil || db.GetSetting(legacyMigratedSetting, "") == "1" {
		return
	}
	recs, err := readLegacyRecords()
	if err != nil {
		logger.Warn("Failed to read legacy install records", "error", err)
		return
	}
	warnings := []string{}
	for _, r := range recs {
		warnings = append(warnings, s.legacyWarning(r))
	}
	b, err := json.Marshal(warnings)
	if err != nil {
		logger.Warn("Failed to encode legacy install warnings", "error", err)
		return
	}
	if err := db.SetSetting(legacyWarningsSetting, string(b)); err != nil {
		logger.Warn("Failed to save legacy install warnings", "error", err)
		return
	}
	if err := db.SetSetting(legacyMigratedSetting, "1"); err != nil {
		logger.Warn("Failed to record legacy install check", "error", err)
		return
	}
	logger.Info("Checked legacy install records", "records", len(recs), "warnings", len(warnings))
}

// legacyWarning explains what a v3 record means today and how to recover.
func (s *realService) legacyWarning(r legacyRecord) string {
	when := ""
	if r.installedAt != "" {
		when = " (recorded " + r.installedAt + ")"
	}
	if !r.global {
		return "Pre-4.0 project-scope install record: " + r.agentType + " wrote " + r.installPath + " relative to the directory the server ran in" + when +
			". 4.0 manages global installs only; that file is no longer tracked and can be deleted from the project if you no longer need it."
	}
	path := r.installPath
	if rest, ok := strings.CutPrefix(path, "~/"); ok {
		path = filepath.Join(s.home, rest)
	}
	state := ""
	if data, err := os.ReadFile(path); os.IsNotExist(err) {
		state = " The file no longer exists."
	} else if err == nil && strings.HasSuffix(path, ".json") && !json.Valid(data) {
		state = " It is not valid JSON right now."
	}
	if r.agentType == string(TargetClaudeCode) {
		return "A pre-4.0 install wrote markdown instructions over ~/.claude.json" + when + ", which can erase Claude Code's settings and MCP servers." + state +
			" If Claude Code lost its configuration, restore ~/.claude.json from Claude Code's own backups in ~/.claude/backups/, then run `ast-mcp install --target claude_code`."
	}
	return "A pre-4.0 install replaced " + r.installPath + when + " with a config holding only ast-context-cache, so other entries in that file may have been lost." + state +
		" Check the file (or your editor's backups), then run `ast-mcp install --target " + r.agentType + "`."
}

func readLegacyRecords() ([]legacyRecord, error) {
	rows, err := db.DB.Query(selectLegacyAgentConfigsQuery)
	if err != nil {
		return nil, errs.WrapMessage("failed to query agent_configs", err)
	}
	defer rows.Close()
	var out []legacyRecord
	for rows.Next() {
		var r legacyRecord
		var global int
		if err := rows.Scan(&r.agentType, &r.installPath, &global, &r.installedAt); err != nil {
			return nil, errs.WrapMessage("failed to scan agent_configs", err)
		}
		r.global = global == 1
		out = append(out, r)
	}
	return out, errs.WrapMessage("failed to read agent_configs", rows.Err())
}
