package contextnotes

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// KindOffload marks a tool result the server offloaded itself; it is fetched back by ref.
	KindOffload = "offload"
	// KindTranscriptArchive is reserved for archived host transcripts (P4).
	KindTranscriptArchive = "transcript_archive"
	// ExpiredStatus is the status Fetch reports for an offload ref whose note has expired.
	ExpiredStatus = "expired"
	// tombstoneRetention is how long Fetch can still explain an expired ref.
	tombstoneRetention = 30 * 24 * time.Hour

	// notOffloadClause keeps offload and transcript archive notes out of the session and
	// global quotas and out of session LRU eviction.
	notOffloadClause = ` AND COALESCE(kind,'') NOT IN ('` + KindOffload + `','` + KindTranscriptArchive + `')`

	selectOffloadTokensQuery = `SELECT COALESCE(SUM(token_est),0) FROM context_notes WHERE kind = '` + KindOffload + `'`
	selectOldestOffloadQuery = `SELECT ref, COALESCE(metadata_json,'') FROM context_notes WHERE kind = '` + KindOffload + `'
		ORDER BY COALESCE(last_accessed_at, created_at) ASC LIMIT 1`
	selectExpiredOffloadsQuery = `SELECT ref, COALESCE(metadata_json,'') FROM context_notes WHERE kind = '` + KindOffload + `'
		AND COALESCE(last_accessed_at, created_at) < ?`
	insertTombstoneQuery = `INSERT OR REPLACE INTO context_note_tombstones (ref, kind, tool, args_json, expired_at)
		VALUES (?, ?, ?, ?, ?)`
	selectTombstoneQuery = `SELECT COALESCE(tool,''), COALESCE(args_json,'') FROM context_note_tombstones WHERE ref = ?`
	pruneTombstonesQuery = `DELETE FROM context_note_tombstones WHERE expired_at < ?`
)

// ExpiredRef is what Fetch returns for a ref whose offload note has expired.
type ExpiredRef struct {
	Ref    string         `json:"ref"`
	Status string         `json:"status"`
	Tool   string         `json:"tool,omitempty"`
	Args   map[string]any `json:"args,omitempty"`
}

type offloadMeta struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args"`
}

// StoreOffload stores a full tool result as an offload note, without embedding it.
func StoreOffload(sessionID, projectPath, text, tool string, args map[string]any, target string) (*StoreResult, error) {
	label := strings.TrimSpace(tool + " " + target)
	return Store(sessionID, text, label, projectPath, nil, KindOffload, map[string]any{"tool": tool, "args": args}, nil)
}

// OffloadStub is the line prepended to a truncated result, pointing at the full one.
func OffloadStub(ref, tool, target string, tokens int) string {
	head := strings.Join(strings.Fields(strings.Join([]string{ref, tool, target}, " ")), " ")
	return "[" + head + ", " + formatTokenCount(tokens) + " tok — head shown, fetch_context for all]"
}

// formatTokenCount renders 9400 as "9.4k" and values under 1000 as plain integers.
func formatTokenCount(n int) string {
	if n < 1000 {
		return strconv.Itoa(n)
	}
	return strings.TrimSuffix(strconv.FormatFloat(float64(n)/1000, 'f', 1, 64), ".0") + "k"
}

// checkOffloadLimit evicts the least recently used offload notes until addTokens fits
// under the offload budget, tombstoning each one.
func checkOffloadLimit(addTokens int, lim Limits) ([]string, error) {
	if addTokens > lim.MaxOffloadTokensGlobal {
		return nil, newLimitError("offload_tokens", addTokens, lim.MaxOffloadTokensGlobal, addTokens)
	}
	var evicted []string
	for {
		var used int
		if err := db.ContextDB.QueryRow(selectOffloadTokensQuery).Scan(&used); err != nil {
			return evicted, errs.WrapMessage("failed to total offload tokens", err)
		}
		if used+addTokens <= lim.MaxOffloadTokensGlobal {
			return evicted, nil
		}
		var ref, meta string
		if err := db.ContextDB.QueryRow(selectOldestOffloadQuery).Scan(&ref, &meta); err != nil {
			return evicted, newLimitError("offload_tokens", used, lim.MaxOffloadTokensGlobal, addTokens)
		}
		// Stop on a failed delete: the same note would come back as the oldest.
		if err := expireOffloads(map[string]string{ref: meta}); err != nil {
			return evicted, errs.WrapMessage("failed to evict offload note", err, "ref", ref)
		}
		evicted = append(evicted, ref)
	}
}

// PurgeExpiredOffloads deletes offload notes not accessed (or stored) within ttl, leaving a
// tombstone so Fetch can say the ref expired, and prunes tombstones older than 30 days.
func PurgeExpiredOffloads(ttl time.Duration) (int, error) {
	rows, err := db.ContextDB.Query(selectExpiredOffloadsQuery, db.SQLTime(time.Now().Add(-ttl)))
	if err != nil {
		return 0, errs.WrapMessage("failed to list expired offload notes", err)
	}
	metas := map[string]string{}
	for rows.Next() {
		var ref, meta string
		if err := rows.Scan(&ref, &meta); err != nil {
			rows.Close()
			return 0, errs.WrapMessage("failed to scan expired offload note", err)
		}
		metas[ref] = meta
	}
	rows.Close()
	if err := expireOffloads(metas); err != nil {
		return 0, err
	}
	if _, err := db.ContextDB.Exec(pruneTombstonesQuery, db.SQLTime(time.Now().Add(-tombstoneRetention))); err != nil {
		return len(metas), errs.WrapMessage("failed to prune context note tombstones", err)
	}
	if len(metas) > 0 {
		logger.Info("Purged expired offload notes", "purged", len(metas), "ttl", ttl)
	}
	return len(metas), nil
}

// RunOffloadPurge is the hourly job: PurgeExpiredOffloads with the configured TTL.
func RunOffloadPurge() {
	if _, err := PurgeExpiredOffloads(OffloadTTL()); err != nil {
		logger.Warn("Offload purge failed", "error", err)
	}
}

// expireOffloads writes a tombstone for each ref (metadata_json by ref), then deletes the notes.
func expireOffloads(metas map[string]string) error {
	if len(metas) == 0 {
		return nil
	}
	now := db.SQLTime(time.Now())
	refs := make([]string, 0, len(metas))
	for ref, raw := range metas {
		var m offloadMeta
		_ = json.Unmarshal([]byte(raw), &m)
		args := ""
		if len(m.Args) > 0 && string(m.Args) != "null" {
			args = string(m.Args)
		}
		if _, err := db.ContextDB.Exec(insertTombstoneQuery, ref, KindOffload, m.Tool, args, now); err != nil {
			return errs.WrapMessage("failed to write context note tombstone", err, "ref", ref)
		}
		refs = append(refs, ref)
	}
	if _, _, _, err := deleteRefs(refs, ""); err != nil {
		return errs.WrapMessage("failed to delete expired offload notes", err, "count", len(refs))
	}
	return nil
}

// tombstoneFor reports whether ref belongs to an expired offload note.
func tombstoneFor(ref string) (ExpiredRef, bool) {
	var tool, argsJSON string
	if err := db.ContextDB.QueryRow(selectTombstoneQuery, ref).Scan(&tool, &argsJSON); err != nil {
		return ExpiredRef{}, false
	}
	e := ExpiredRef{Ref: ref, Status: ExpiredStatus, Tool: tool}
	if argsJSON != "" {
		_ = json.Unmarshal([]byte(argsJSON), &e.Args)
	}
	return e, true
}
