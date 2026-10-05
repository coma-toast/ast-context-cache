package docs

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	upsertDocSourceQuery = `INSERT INTO doc_sources (name, type, url, version) VALUES (?, ?, ?, ?)
		 ON CONFLICT(name, type, url) DO UPDATE SET version = excluded.version`
	selectDocSourceIDQuery          = "SELECT id FROM doc_sources WHERE name = ? AND type = ? AND url = ?"
	deleteDocContentQuery           = "DELETE FROM doc_content WHERE source_id = ?"
	deleteDocSourceQuery            = "DELETE FROM doc_sources WHERE id = ?"
	countDocSourcesQuery            = "SELECT COUNT(*) FROM doc_sources"
	listDocSourcesQuery             = "SELECT id, name, type, url, COALESCE(version,''), COALESCE(last_updated,''), created_at FROM doc_sources ORDER BY name LIMIT ? OFFSET ?"
	selectDocSourceQuery            = "SELECT name, type, url FROM doc_sources WHERE id = ?"
	updateDocSourceUpdatedQuery     = "UPDATE doc_sources SET last_updated = ? WHERE id = ?"
	insertDocContentQuery           = `INSERT INTO doc_content (source_id, title, content, path, content_hash) VALUES (?, ?, ?, ?, ?)`
	rebuildDocsFTSQuery             = `INSERT INTO docs_fts(docs_fts) VALUES('rebuild')`
	selectDocSourceLastUpdatedQuery = "SELECT COALESCE(last_updated,'') FROM doc_sources WHERE id = ?"
	listDocEntriesBySourceQuery     = `
		SELECT id, source_id, title, content, COALESCE(path,''), COALESCE(content_hash,''), updated_at
		FROM doc_content WHERE source_id = ? ORDER BY id`
)

// DocSourceMaxAge is how long cached doc content is kept before background re-fetch.
const DocSourceMaxAge = 7 * 24 * time.Hour

type DocSource struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	URL         string `json:"url"`
	Version     string `json:"version,omitempty"`
	LastUpdated string `json:"last_updated,omitempty"`
	CreatedAt   string `json:"created_at"`
}

type DocEntry struct {
	ID          int    `json:"id"`
	SourceID    int    `json:"source_id"`
	Title       string `json:"title"`
	Content     string `json:"content"`
	Path        string `json:"path,omitempty"`
	ContentHash string `json:"content_hash,omitempty"`
	UpdatedAt   string `json:"updated_at"`
}

func AddSource(name, docType, docURL, version string) (int, error) {
	_, err := db.ContextDB.Exec(upsertDocSourceQuery, name, docType, docURL, version)
	if err != nil {
		return 0, err
	}
	var id int
	err = db.ContextDB.QueryRow(selectDocSourceIDQuery, name, docType, docURL).Scan(&id)
	return id, err
}

func RemoveSource(id int) error {
	if err := deleteDocVectors(id); err != nil {
		return err
	}
	db.ContextDB.Exec(deleteDocContentQuery, id)
	_, err := db.ContextDB.Exec(deleteDocSourceQuery, id)
	rebuildDocsFTS()
	return err
}

func ListSources() ([]DocSource, error) {
	sources, _, _, err := ListSourcesPaged(1, 0)
	return sources, err
}

// ListSourcesPaged returns doc sources ordered by name. perPage <= 0 means no limit (all rows).
// page is clamped to valid range; the returned page is the clamped value.
func ListSourcesPaged(page, perPage int) ([]DocSource, int, int, error) {
	if page < 1 {
		page = 1
	}
	var total int
	if err := db.ContextDB.QueryRow(countDocSourcesQuery).Scan(&total); err != nil {
		return nil, 0, page, err
	}
	if total == 0 {
		return nil, 0, 1, nil
	}
	if perPage <= 0 {
		perPage = total
	}
	totalPages := (total + perPage - 1) / perPage
	if page > totalPages {
		page = totalPages
	}
	offset := (page - 1) * perPage
	rows, err := db.ContextDB.Query(listDocSourcesQuery, perPage, offset)
	if err != nil {
		return nil, total, page, err
	}
	defer rows.Close()
	var sources []DocSource
	for rows.Next() {
		var s DocSource
		rows.Scan(&s.ID, &s.Name, &s.Type, &s.URL, &s.Version, &s.LastUpdated, &s.CreatedAt)
		sources = append(sources, s)
	}
	return sources, total, page, nil
}

func UpdateSource(id int) (usedPlaywright bool, err error) {
	var name, docType, docURL string
	err = db.ContextDB.QueryRow(selectDocSourceQuery, id).Scan(&name, &docType, &docURL)
	if err != nil {
		return false, err
	}

	content, usedPlaywright, err := fetchDocs(docURL, docType)
	if err != nil {
		return false, errs.WrapMessage("failed to fetch doc source", err, "source_id", id)
	}

	if err := deleteDocVectors(id); err != nil {
		return false, err
	}
	db.ContextDB.Exec(deleteDocContentQuery, id)
	if err := storeEntries(id, content); err != nil {
		return false, err
	}
	db.ContextDB.Exec(updateDocSourceUpdatedQuery, time.Now().Format(time.RFC3339), id)
	go EmbedSource(id)
	return usedPlaywright, nil
}

func storeEntries(sourceID int, entries []DocEntry) error {
	for _, entry := range entries {
		hash := contentHash(entry.Content)
		_, err := db.ContextDB.Exec(insertDocContentQuery, sourceID, entry.Title, entry.Content, entry.Path, hash)
		if err != nil {
			return err
		}
	}
	rebuildDocsFTS()
	return nil
}

func rebuildDocsFTS() {
	db.ContextDB.Exec(rebuildDocsFTSQuery)
}

func UpdateAllSources() {
	if db.ContextDB == nil {
		return
	}
	sources, err := ListSources()
	if err != nil {
		return
	}
	for _, s := range sources {
		if !SourceNeedsRefresh(s.LastUpdated) {
			continue
		}
		if _, err := UpdateSource(s.ID); err != nil {
			logger.Warn("Failed to refresh doc source", "source_id", s.ID, "error", err)
		}
	}
}

// SourceNeedsRefresh reports whether a doc source should be re-fetched from its URL.
func SourceNeedsRefresh(lastUpdated string) bool {
	if lastUpdated == "" {
		return true
	}
	lastTime, err := time.Parse(time.RFC3339, lastUpdated)
	if err != nil {
		return true
	}
	return time.Since(lastTime) >= DocSourceMaxAge
}

// FetchAndCache registers a doc source, fetches when missing/stale/forced, and returns cached entries.
// When renderJS is true, the stored type becomes webpage when Playwright is available; otherwise html.
func FetchAndCache(name, docType, docURL, version string, force, renderJS bool) (id int, entries []DocEntry, refreshed, usedPlaywright bool, err error) {
	docType = NormalizeDocType(docType, renderJS)
	id, err = AddSource(name, docType, docURL, version)
	if err != nil {
		return 0, nil, false, false, err
	}
	var lastUpdated string
	if err = db.ContextDB.QueryRow(selectDocSourceLastUpdatedQuery, id).Scan(&lastUpdated); err != nil {
		return id, nil, false, false, err
	}
	if force || SourceNeedsRefresh(lastUpdated) {
		usedPlaywright, err = UpdateSource(id)
		if err != nil {
			return id, nil, false, false, err
		}
		refreshed = true
	} else {
		go EmbedSource(id)
	}
	entries, err = ListEntriesBySource(id)
	return id, entries, refreshed, usedPlaywright, err
}

func ListEntriesBySource(sourceID int) ([]DocEntry, error) {
	rows, err := db.ContextDB.Query(listDocEntriesBySourceQuery, sourceID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var entries []DocEntry
	for rows.Next() {
		var e DocEntry
		rows.Scan(&e.ID, &e.SourceID, &e.Title, &e.Content, &e.Path, &e.ContentHash, &e.UpdatedAt)
		entries = append(entries, e)
	}
	return entries, nil
}

func fetchDocs(docURL, docType string) ([]DocEntry, bool, error) {
	parsedURL, err := url.Parse(docURL)
	if err != nil {
		return nil, false, err
	}

	switch docType {
	case "markdown", "md":
		entries, err := fetchMarkdown(parsedURL)
		return entries, false, err
	case "html":
		return fetchHTML(parsedURL, false)
	case "webpage":
		return fetchHTML(parsedURL, true)
	case "json", "api":
		entries, err := fetchJSONDocs(parsedURL)
		return entries, false, err
	default:
		entries, err := fetchMarkdown(parsedURL)
		return entries, false, err
	}
}

func fetchMarkdown(u *url.URL) ([]DocEntry, error) {
	body, err := fetchURL(u.String())
	if err != nil {
		return nil, err
	}
	return chunkMarkdown(string(body), u.Path), nil
}

func fetchHTML(u *url.URL, renderJS bool) ([]DocEntry, bool, error) {
	if renderJS && RenderEnabled() {
		body, err := fetchRenderedURL(u.String())
		if err == nil {
			entries, err := htmlEntriesFromBody(string(body), u)
			return entries, true, err
		}
		logger.Warn("Playwright render failed, falling back to plain HTTP", "url", u.String(), "error", err)
	} else if renderJS && !RenderEnabled() {
		logger.Info("Playwright unavailable, fetching as plain HTML", "url", u.String())
	}
	body, err := fetchURL(u.String())
	if err != nil {
		return nil, false, err
	}
	entries := chunkHTML(string(body), u.Path)
	if entriesSparse(entries) && RenderEnabled() {
		if rendered, rerr := fetchRenderedURL(u.String()); rerr == nil {
			if ren, rerr := htmlEntriesFromBody(string(rendered), u); rerr == nil && !entriesSparse(ren) {
				return ren, true, nil
			}
		}
	}
	if len(entries) == 0 {
		return nil, false, errs.New("no extractable content", "url", u.String())
	}
	return entries, false, nil
}

func htmlEntriesFromBody(body string, u *url.URL) ([]DocEntry, error) {
	entries := chunkHTML(body, u.Path)
	if len(entries) == 0 {
		return nil, errs.New("no extractable content", "url", u.String())
	}
	return entries, nil
}

func fetchJSONDocs(u *url.URL) ([]DocEntry, error) {
	body, err := fetchURL(u.String())
	if err != nil {
		return nil, err
	}
	return chunkJSON(string(body), u.Path), nil
}

// docFetchClient bounds how long a doc-source fetch can hang — http.Get's
// default client has no timeout at all.
var docFetchClient = &http.Client{Timeout: 20 * time.Second}

func fetchURL(raw string) ([]byte, error) {
	if strings.HasPrefix(raw, "file://") {
		return fetchLocalFile(raw)
	}
	resp, err := docFetchClient.Get(raw)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	// A bare http.Get read the body regardless of status, so a 404/500 error
	// page got chunked and cached as if it were real documentation — later
	// search_docs/fetch_doc calls would return it with no indication anything
	// had failed.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, errs.New(fmt.Sprintf("unexpected status %s", resp.Status), "url", raw, "status", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// fetchLocalFile reads a file:// doc source straight off disk — Go's http.Client has no
// handler for the file scheme, so refreshing a source registered against a local path
// (a repo's own docs, a synced notes file) needs its own path instead of http.Get.
func fetchLocalFile(raw string) ([]byte, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, errs.WrapMessage("failed to parse file URL", err, "url", raw)
	}
	path := u.Path
	if u.Host != "" && u.Host != "localhost" {
		// file://host/path (rare outside Windows UNC paths) — best effort, not a case
		// this tool otherwise needs to support.
		path = "/" + u.Host + path
	}
	if path == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "file URL has no path", "url", raw)
	}
	return os.ReadFile(path)
}

type markdownSection struct {
	Title   string
	Content string
}

func splitMarkdownSections(content string) []markdownSection {
	lines := strings.Split(content, "\n")
	var sections []markdownSection
	var currentTitle string
	var currentContent strings.Builder
	flush := func() {
		if currentTitle == "" {
			return
		}
		sections = append(sections, markdownSection{
			Title:   currentTitle,
			Content: strings.TrimSpace(currentContent.String()),
		})
	}
	for _, line := range lines {
		if level, title := markdownHeading(line); level > 0 && level <= 2 {
			flush()
			currentTitle = title
			currentContent.Reset()
			continue
		}
		currentContent.WriteString(line + "\n")
	}
	flush()
	return sections
}

func markdownHeading(line string) (level int, title string) {
	line = strings.TrimSpace(line)
	for i := 6; i >= 1; i-- {
		prefix := strings.Repeat("#", i) + " "
		if strings.HasPrefix(line, prefix) {
			return i, strings.TrimSpace(line[i+1:])
		}
	}
	return 0, ""
}

func contentHash(text string) string {
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:])
}
