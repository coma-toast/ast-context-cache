package trail

import (
	"database/sql"
	"encoding/json"
	"errors"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	// timeFormat is fixed-width so created_at sorts and compares as text.
	timeFormat         = "2006-01-02T15:04:05.000000000Z"
	selectTrailColumns = `COALESCE(session_id,''), COALESCE(tool,''), COALESCE(query,''), COALESCE(query_norm,''),
		COALESCE(filters_key,''), COALESCE(mode,''), COALESCE(doc_type,''), COALESCE(project_path,''),
		COALESCE(hit_count,0), COALESCE(zero_hit,0), COALESCE(top_hits_json,''), COALESCE(created_at,'')`
	selectSessionTrailQuery = `SELECT ` + selectTrailColumns + `
		FROM search_trail WHERE session_id = ?
		ORDER BY created_at DESC, id DESC LIMIT ?`
	// The match key is rebuilt in SQL exactly as Entry.MatchKey builds it.
	selectTrailMatchQuery = `SELECT ` + selectTrailColumns + `
		FROM search_trail
		WHERE session_id = ?
		AND COALESCE(tool,'') || '|' || COALESCE(query_norm,'') || '|' || COALESCE(filters_key,'') || '|' || COALESCE(doc_type,'') = ?
		ORDER BY created_at DESC, id DESC LIMIT 1`
	deleteTrailBeforeQuery = `DELETE FROM search_trail WHERE created_at < ?`
)

func toRow(e Entry) db.TrailRow {
	topHits := ""
	if len(e.TopHits) > 0 {
		data, _ := json.Marshal(e.TopHits)
		topHits = string(data)
	}
	return db.TrailRow{
		SessionID: e.SessionID, Tool: e.Tool, Query: e.Query, QueryNorm: e.QueryNorm,
		FiltersKey: e.FiltersKey, Mode: e.Mode, DocType: e.DocType, ProjectPath: e.ProjectPath,
		HitCount: e.HitCount, ZeroHit: e.ZeroHit, TopHitsJSON: topHits, CreatedAt: e.At.UTC().Format(timeFormat),
	}
}

func loadSession(sid string, limit int) ([]Entry, error) {
	rows, err := db.DB.Query(selectSessionTrailQuery, sid, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Entry
	for rows.Next() {
		e, err := scanEntry(rows)
		if err != nil {
			return out, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

func loadMatch(sid, matchKey string) (Entry, bool, error) {
	e, err := scanEntry(db.DB.QueryRow(selectTrailMatchQuery, sid, matchKey))
	if errors.Is(err, sql.ErrNoRows) {
		return Entry{}, false, nil
	}
	if err != nil {
		return Entry{}, false, err
	}
	return e, true, nil
}

func deleteBefore(cutoff time.Time) (int64, error) {
	res, err := db.DB.Exec(deleteTrailBeforeQuery, cutoff.UTC().Format(timeFormat))
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

type scanner interface {
	Scan(dest ...any) error
}

func scanEntry(s scanner) (Entry, error) {
	var e Entry
	var zeroHit int
	var topHits, createdAt string
	if err := s.Scan(&e.SessionID, &e.Tool, &e.Query, &e.QueryNorm, &e.FiltersKey, &e.Mode, &e.DocType, &e.ProjectPath, &e.HitCount, &zeroHit, &topHits, &createdAt); err != nil {
		return Entry{}, err
	}
	e.ZeroHit = zeroHit != 0
	if topHits != "" {
		if err := json.Unmarshal([]byte(topHits), &e.TopHits); err != nil {
			logger.Warn("Failed to decode search trail top hits", "error", err, "session", e.SessionID)
		}
	}
	if at, err := time.Parse(timeFormat, createdAt); err == nil {
		e.At = at
	}
	return e, nil
}
