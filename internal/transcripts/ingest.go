// Package transcripts ingests host token usage from Claude Code transcripts (TL-5). It is
// opt-in (setting transcript_usage_ingest), reads ~/.claude/projects/*/*.jsonl from a saved
// byte offset per file, and keeps only per-day token totals: transcript text is never stored.
package transcripts

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/logging"
)

const (
	// SettingKey turns the hourly ingest on ("true"); it is off by default.
	SettingKey     = "transcript_usage_ingest"
	settingDefault = "false"
	dayFormat      = "2006-01-02"

	selectOffsetQuery = `SELECT offset, mtime FROM host_usage_offsets WHERE path = ?`
	upsertOffsetQuery = `INSERT INTO host_usage_offsets (path, offset, mtime) VALUES (?, ?, ?)
		ON CONFLICT(path) DO UPDATE SET offset = excluded.offset, mtime = excluded.mtime`
	upsertDailyQuery = `INSERT INTO host_usage_daily (day, project_dir, input, output, cache_read, cache_write) VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(day, project_dir) DO UPDATE SET input = input + excluded.input, output = output + excluded.output,
		cache_read = cache_read + excluded.cache_read, cache_write = cache_write + excluded.cache_write`
	selectDailySeriesQuery = `SELECT day, SUM(input), SUM(output), SUM(cache_read), SUM(cache_write)
		FROM host_usage_daily WHERE day >= ? GROUP BY day ORDER BY day`
)

var logger = logging.Tagged("transcripts")

// usageMarker is a cheap pre-filter: lines without it carry no usage and are not decoded.
var usageMarker = []byte(`"usage"`)

// line is the only shape decoded from a transcript line; message content is never read.
type line struct {
	Timestamp string `json:"timestamp"`
	RequestID string `json:"requestId"`
	Message   *struct {
		ID    string `json:"id"`
		Usage *struct {
			InputTokens              int64 `json:"input_tokens"`
			OutputTokens             int64 `json:"output_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	} `json:"message"`
}

// Usage is token usage for one day (and project directory when aggregating).
type Usage struct {
	Day        string `json:"Day"`
	Input      int64  `json:"Input"`
	Output     int64  `json:"Output"`
	CacheRead  int64  `json:"CacheRead"`
	CacheWrite int64  `json:"CacheWrite"`
}

type dayProject struct{ day, project string }

type fileProgress struct {
	path          string
	offset, mtime int64
}

// Result summarizes one ingest pass.
type Result struct {
	Files int
	Lines int
}

// Enabled reports whether the transcript_usage_ingest setting is on.
func Enabled() bool {
	return db.GetSetting(SettingKey, settingDefault) == "true"
}

// DefaultRoot is ~/.claude/projects.
func DefaultRoot() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".claude", "projects")
}

// RunOnce ingests DefaultRoot when the setting is on; runEvery calls it hourly.
func RunOnce() {
	if !Enabled() {
		return
	}
	if _, err := IngestOnce(DefaultRoot()); err != nil {
		logger.Warn("Transcript usage ingest failed", "error", err)
	}
}

// IngestOnce reads new lines from root/*/*.jsonl and adds their usage to host_usage_daily.
// Each file resumes from its saved offset; a partial last line is left for the next pass.
func IngestOnce(root string) (Result, error) {
	var res Result
	if db.DB == nil {
		return res, errs.New("usage database not open")
	}
	paths, err := filepath.Glob(filepath.Join(root, "*", "*.jsonl"))
	if err != nil {
		return res, errs.WrapMessage("failed to list transcripts", err, "root", root)
	}
	totals := map[dayProject]*Usage{}
	var progress []fileProgress
	for _, p := range paths {
		fp, n, err := ingestFile(p, totals)
		if err != nil {
			logger.Warn("Failed to read transcript", "error", err, "path", p)
			continue
		}
		if fp != nil {
			progress = append(progress, *fp)
			res.Files++
			res.Lines += n
		}
	}
	return res, save(totals, progress)
}

func ingestFile(path string, totals map[dayProject]*Usage) (*fileProgress, int, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, 0, err
	}
	var offset, mtime int64
	if err := db.DB.QueryRow(selectOffsetQuery, path).Scan(&offset, &mtime); err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, 0, err
	}
	if fi.Size() < offset {
		offset = 0 // replaced or truncated: read the new content from the start
	}
	if fi.Size() == offset && fi.ModTime().Unix() == mtime {
		return nil, 0, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, 0, err
	}
	project := filepath.Base(filepath.Dir(path))
	seen := map[string]bool{}
	r := bufio.NewReader(f)
	n := 0
	for {
		b, err := r.ReadBytes('\n')
		if err == io.EOF {
			break // an unterminated last line is still being written
		}
		if err != nil {
			return nil, 0, err
		}
		offset += int64(len(b))
		n++
		addLine(b, project, seen, totals)
	}
	return &fileProgress{path: path, offset: offset, mtime: fi.ModTime().Unix()}, n, nil
}

// addLine adds one line's usage. Claude Code writes the same message's usage on every
// content-block line, so repeats of a message/request id within a pass count once.
func addLine(b []byte, project string, seen map[string]bool, totals map[dayProject]*Usage) {
	if !bytes.Contains(b, usageMarker) {
		return
	}
	var l line
	if json.Unmarshal(b, &l) != nil || l.Message == nil || l.Message.Usage == nil {
		return
	}
	if id := l.Message.ID + "\x00" + l.RequestID; id != "\x00" {
		if seen[id] {
			return
		}
		seen[id] = true
	}
	ts, err := time.Parse(time.RFC3339, l.Timestamp)
	if err != nil {
		return
	}
	k := dayProject{day: ts.Local().Format(dayFormat), project: project}
	u := totals[k]
	if u == nil {
		u = &Usage{Day: k.day}
		totals[k] = u
	}
	u.Input += l.Message.Usage.InputTokens
	u.Output += l.Message.Usage.OutputTokens
	u.CacheRead += l.Message.Usage.CacheReadInputTokens
	u.CacheWrite += l.Message.Usage.CacheCreationInputTokens
}

// save writes the totals and the new offsets in one transaction, so a failed pass is
// retried in full instead of counting lines twice.
func save(totals map[dayProject]*Usage, progress []fileProgress) error {
	if len(progress) == 0 {
		return nil
	}
	tx, err := db.DB.Begin()
	if err != nil {
		return errs.WrapMessage("failed to begin host usage write", err)
	}
	defer tx.Rollback()
	for k, u := range totals {
		if _, err := tx.Exec(upsertDailyQuery, k.day, k.project, u.Input, u.Output, u.CacheRead, u.CacheWrite); err != nil {
			return errs.WrapMessage("failed to write host usage", err, "day", k.day, "project", k.project)
		}
	}
	for _, p := range progress {
		if _, err := tx.Exec(upsertOffsetQuery, p.path, p.offset, p.mtime); err != nil {
			return errs.WrapMessage("failed to save transcript offset", err, "path", p.path)
		}
	}
	return tx.Commit()
}

// DailySeries returns per-day usage summed over projects for the last days days.
func DailySeries(days int) ([]Usage, error) {
	out := []Usage{}
	if db.DB == nil {
		return out, nil
	}
	since := time.Now().AddDate(0, 0, -days+1).Format(dayFormat)
	rows, err := db.DB.Query(selectDailySeriesQuery, since)
	if err != nil {
		return out, errs.WrapMessage("failed to read host usage", err, "since", since)
	}
	defer rows.Close()
	for rows.Next() {
		var u Usage
		if err := rows.Scan(&u.Day, &u.Input, &u.Output, &u.CacheRead, &u.CacheWrite); err != nil {
			return out, errs.WrapMessage("failed to scan host usage", err)
		}
		out = append(out, u)
	}
	return out, rows.Err()
}
