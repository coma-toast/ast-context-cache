package contextnotes

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	KindKvRepair    = "kv_repair"
	TagKvRepair     = "kv_repair"
	RepairProactive = "proactive"
	RepairCacheMiss = "cache_miss"
	RepairQuality   = "quality"
	RepairManual    = "manual"
)

const (
	kvRepairNoteWhereClause     = `(kind = ? OR tags LIKE ? OR tags LIKE ? OR tags = ?)`
	selectKvRepairArchivesQuery = `SELECT COUNT(*), COALESCE(SUM(CASE WHEN access_count=0 THEN 1 ELSE 0 END),0)
		FROM context_notes WHERE `
	countNotesWhereQuery             = `SELECT COUNT(*) FROM context_notes WHERE `
	createdSinceClause               = ` AND created_at >= ?`
	createdBetweenClause             = ` AND created_at >= ? AND created_at < ?`
	repairAccessSinceClause          = `repair_reason != '' AND accessed_at >= ?`
	repairAccessBetweenClause        = `repair_reason != '' AND accessed_at >= ? AND accessed_at < ?`
	selectRepairAccessTotalsQuery    = `SELECT COUNT(*), COALESCE(SUM(virtual_tokens),0) FROM context_note_access WHERE `
	countRepairAccessWhereQuery      = `SELECT COUNT(*) FROM context_note_access WHERE `
	countRepairAccessByReasonQuery   = `SELECT COUNT(*) FROM context_note_access WHERE repair_reason = ? AND accessed_at >= ?`
	eventCreatedSinceClause          = `created_at >= ?`
	countKvRepairEventsByReasonQuery = `SELECT COUNT(*) FROM kv_repair_events WHERE repair_reason = ? AND `
	countKvRepairEventsSuccessQuery  = `SELECT COUNT(*) FROM kv_repair_events WHERE outcome = 'success' AND `
	countKvRepairEventsFailedQuery   = `SELECT COUNT(*) FROM kv_repair_events WHERE outcome = 'failed' AND `
	insertKvRepairEventQuery         = `INSERT INTO kv_repair_events
		(session_id, project_path, ref, repair_reason, outcome, model_id, kv_quant, token_est, detail, metadata_json)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
)

// KvRepairMetadata is persisted in context_notes.metadata_json for kv_repair archives.
type KvRepairMetadata struct {
	ModelID     string `json:"model_id,omitempty"`
	KvQuant     string `json:"kv_quant,omitempty"`
	TokenCount  int    `json:"token_count,omitempty"`
	TriggerHint string `json:"trigger_hint,omitempty"`
	ChunkOffset int    `json:"chunk_offset,omitempty"`
}

// KvRepairDashboardStats aggregates kv_repair observability for the dashboard.
type KvRepairDashboardStats struct {
	ArchivesActive          int            `json:"archives_active"`
	ArchivesStored30d       int            `json:"archives_stored_30d"`
	RepairsTotal30d         int            `json:"repairs_total_30d"`
	RepairsByReason         map[string]int `json:"repairs_by_reason"`
	RepairUtilizationPct30d float64        `json:"repair_utilization_pct_30d"`
	RepairOrphans           int            `json:"repair_orphans"`
	TokensRepaired30d       int            `json:"tokens_repaired_30d"`
	QualitySignals30d       int            `json:"quality_signals_30d"`
	CacheMissSignals30d     int            `json:"cache_miss_signals_30d"`
	ManualSignals30d        int            `json:"manual_signals_30d"`
	RepairSuccessRate       float64        `json:"repair_success_rate"`
	TodayRepairs            int            `json:"today_repairs"`
	TodayArchives           int            `json:"today_archives"`
}

// ReportEventInput records a kv_repair signal without fetching archived text.
type ReportEventInput struct {
	Reason      string
	Outcome     string
	SessionID   string
	ProjectPath string
	Ref         string
	ModelID     string
	KvQuant     string
	TokenEst    int
	Detail      string
}

func normalizeRepairReason(reason string) string {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case RepairCacheMiss, RepairQuality, RepairManual, RepairProactive:
		return strings.ToLower(strings.TrimSpace(reason))
	default:
		return ""
	}
}

func normalizeRepairOutcome(outcome string) string {
	switch strings.ToLower(strings.TrimSpace(outcome)) {
	case "success", "failed", "skipped":
		return strings.ToLower(strings.TrimSpace(outcome))
	default:
		return ""
	}
}

func hasKvRepairTag(tagStr string) bool {
	for _, t := range strings.Split(tagStr, ",") {
		if strings.TrimSpace(t) == TagKvRepair {
			return true
		}
	}
	return false
}

// IsKvRepairNote reports whether a note is a kv_repair archive.
func IsKvRepairNote(n Note) bool {
	if strings.TrimSpace(n.Kind) == KindKvRepair {
		return true
	}
	return hasKvRepairTag(n.Tags)
}

func resolveStoreKind(kind string, tagStr string, metadata map[string]interface{}) (string, string) {
	kind = strings.TrimSpace(kind)
	if kind == "" && hasKvRepairTag(tagStr) {
		kind = KindKvRepair
	}
	metaJSON := ""
	if len(metadata) > 0 {
		if b, err := json.Marshal(metadata); err == nil {
			metaJSON = string(b)
		}
	} else if kind == KindKvRepair {
		metaJSON = "{}"
	}
	return kind, metaJSON
}

// ParseKvRepairMetadata decodes metadata_json on a note.
func ParseKvRepairMetadata(n Note) (KvRepairMetadata, error) {
	var m KvRepairMetadata
	if strings.TrimSpace(n.MetadataJSON) == "" {
		return m, nil
	}
	err := json.Unmarshal([]byte(n.MetadataJSON), &m)
	return m, err
}

func kvRepairNoteWhere(projectPath string) (where string, args []any) {
	where = kvRepairNoteWhereClause
	args = []any{KindKvRepair, "%" + TagKvRepair + ",%", "%," + TagKvRepair + "%", TagKvRepair}
	if projectPath != "" {
		where = "(" + where + ")" + andProjectPathClause
		args = append(args, projectPath)
	}
	return where, args
}

// localDayStart is local midnight offset days from today; callers convert it with db.SQLTime
// so it compares correctly against the UTC created_at/accessed_at columns.
func localDayStart(offset int) time.Time {
	y, m, d := time.Now().Date()
	return time.Date(y, m, d+offset, 0, 0, 0, 0, time.Local)
}

// KvRepairDashboardStatsFor returns rollup metrics for kv_repair archives and repairs.
func KvRepairDashboardStatsFor(projectPath string, windowDays int) KvRepairDashboardStats {
	if windowDays <= 0 {
		windowDays = 30
	}
	ds := KvRepairDashboardStats{RepairsByReason: map[string]int{}}
	where, args := kvRepairNoteWhere(projectPath)
	db.ContextDB.QueryRow(selectKvRepairArchivesQuery+where, args...).
		Scan(&ds.ArchivesActive, &ds.RepairOrphans)
	cutoff := db.SQLTime(localDayStart(-windowDays))
	todayStart := db.SQLTime(localDayStart(0))
	tomorrowStart := db.SQLTime(localDayStart(1))
	storeWhere := where + createdSinceClause
	storeArgs := append(append([]any{}, args...), cutoff)
	db.ContextDB.QueryRow(countNotesWhereQuery+storeWhere, storeArgs...).Scan(&ds.ArchivesStored30d)
	todayStoreArgs := append(append([]any{}, args...), todayStart, tomorrowStart)
	db.ContextDB.QueryRow(countNotesWhereQuery+where+createdBetweenClause, todayStoreArgs...).Scan(&ds.TodayArchives)
	accessWhere := repairAccessSinceClause
	accessArgs := []any{cutoff}
	if projectPath != "" {
		accessWhere += andProjectPathClause
		accessArgs = append(accessArgs, projectPath)
	}
	db.DB.QueryRow(selectRepairAccessTotalsQuery+accessWhere, accessArgs...).
		Scan(&ds.RepairsTotal30d, &ds.TokensRepaired30d)
	todayAccessArgs := []any{todayStart, tomorrowStart}
	todayAccessWhere := repairAccessBetweenClause
	if projectPath != "" {
		todayAccessWhere += andProjectPathClause
		todayAccessArgs = append(todayAccessArgs, projectPath)
	}
	db.DB.QueryRow(countRepairAccessWhereQuery+todayAccessWhere, todayAccessArgs...).Scan(&ds.TodayRepairs)
	for _, reason := range []string{RepairCacheMiss, RepairQuality, RepairManual} {
		q := countRepairAccessByReasonQuery
		qargs := []any{reason, cutoff}
		if projectPath != "" {
			q += andProjectPathClause
			qargs = append(qargs, projectPath)
		}
		var n int
		db.DB.QueryRow(q, qargs...).Scan(&n)
		ds.RepairsByReason[reason] = n
	}
	eventWhere := eventCreatedSinceClause
	eventArgs := []any{cutoff}
	if projectPath != "" {
		eventWhere += andProjectPathClause
		eventArgs = append(eventArgs, projectPath)
	}
	db.ContextDB.QueryRow(countKvRepairEventsByReasonQuery+eventWhere, append([]any{RepairQuality}, eventArgs...)...).Scan(&ds.QualitySignals30d)
	db.ContextDB.QueryRow(countKvRepairEventsByReasonQuery+eventWhere, append([]any{RepairCacheMiss}, eventArgs...)...).Scan(&ds.CacheMissSignals30d)
	db.ContextDB.QueryRow(countKvRepairEventsByReasonQuery+eventWhere, append([]any{RepairManual}, eventArgs...)...).Scan(&ds.ManualSignals30d)
	if ds.ArchivesStored30d > 0 {
		ds.RepairUtilizationPct30d = float64(ds.RepairsTotal30d) / float64(ds.ArchivesStored30d) * 100
	}
	var success, failed int
	db.ContextDB.QueryRow(countKvRepairEventsSuccessQuery+eventWhere, eventArgs...).Scan(&success)
	db.ContextDB.QueryRow(countKvRepairEventsFailedQuery+eventWhere, eventArgs...).Scan(&failed)
	if success+failed > 0 {
		ds.RepairSuccessRate = float64(success) / float64(success+failed) * 100
	}
	return ds
}

// ReportKvRepairEvent logs a host/agent repair signal for observability.
func ReportKvRepairEvent(in ReportEventInput) error {
	reason := normalizeRepairReason(in.Reason)
	if reason == "" {
		reason = RepairManual
	}
	outcome := normalizeRepairOutcome(in.Outcome)
	meta := map[string]interface{}{}
	if in.Detail != "" {
		meta["detail"] = in.Detail
	}
	metaJSON := ""
	if len(meta) > 0 {
		if b, err := json.Marshal(meta); err == nil {
			metaJSON = string(b)
		}
	}
	_, err := db.ContextDB.Exec(insertKvRepairEventQuery,
		in.SessionID, in.ProjectPath, in.Ref, reason, outcome, in.ModelID, in.KvQuant, in.TokenEst, in.Detail, metaJSON)
	return err
}

func appendKvRepairStats(stats map[string]interface{}, projectPath string) {
	if stats == nil {
		return
	}
	kv := KvRepairDashboardStatsFor(projectPath, 30)
	stats["kv_repair_30d"] = kv
}

// AppendKvRepairStatsPublic adds kv_repair rollup to a stats map (MCP responses).
func AppendKvRepairStatsPublic(stats map[string]interface{}, projectPath string) {
	appendKvRepairStats(stats, projectPath)
}
