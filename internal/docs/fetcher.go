package docs

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
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
	_, err := db.DB.Exec(
		`INSERT INTO doc_sources (name, type, url, version) VALUES (?, ?, ?, ?)
		 ON CONFLICT(name, type, url) DO UPDATE SET version = excluded.version`,
		name, docType, docURL, version)
	if err != nil {
		return 0, err
	}
	var id int
	err = db.DB.QueryRow("SELECT id FROM doc_sources WHERE name = ? AND type = ? AND url = ?", name, docType, docURL).Scan(&id)
	return id, err
}

func RemoveSource(id int) error {
	deleteDocVectors(id)
	db.DB.Exec("DELETE FROM doc_content WHERE source_id = ?", id)
	_, err := db.DB.Exec("DELETE FROM doc_sources WHERE id = ?", id)
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
	if err := db.DB.QueryRow("SELECT COUNT(*) FROM doc_sources").Scan(&total); err != nil {
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
	rows, err := db.DB.Query(
		"SELECT id, name, type, url, COALESCE(version,''), COALESCE(last_updated,''), created_at FROM doc_sources ORDER BY name LIMIT ? OFFSET ?",
		perPage, offset)
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

func UpdateSource(id int) error {
	var name, docType, docURL string
	err := db.DB.QueryRow("SELECT name, type, url FROM doc_sources WHERE id = ?", id).Scan(&name, &docType, &docURL)
	if err != nil {
		return err
	}

	content, err := fetchDocs(docURL, docType)
	if err != nil {
		return fmt.Errorf("fetch failed: %w", err)
	}

	deleteDocVectors(id)
	db.DB.Exec("DELETE FROM doc_content WHERE source_id = ?", id)
	if err := storeEntries(id, content); err != nil {
		return err
	}
	db.DB.Exec("UPDATE doc_sources SET last_updated = ? WHERE id = ?", time.Now().Format(time.RFC3339), id)
	go EmbedSource(id)
	return nil
}

func storeEntries(sourceID int, entries []DocEntry) error {
	for _, entry := range entries {
		hash := contentHash(entry.Content)
		_, err := db.DB.Exec(
			`INSERT INTO doc_content (source_id, title, content, path, content_hash) VALUES (?, ?, ?, ?, ?)`,
			sourceID, entry.Title, entry.Content, entry.Path, hash)
		if err != nil {
			return err
		}
	}
	rebuildDocsFTS()
	return nil
}

func rebuildDocsFTS() {
	db.DB.Exec(`INSERT INTO docs_fts(docs_fts) VALUES('rebuild')`)
}

func UpdateAllSources() {
	sources, err := ListSources()
	if err != nil {
		return
	}
	for _, s := range sources {
		if !SourceNeedsRefresh(s.LastUpdated) {
			continue
		}
		UpdateSource(s.ID)
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
func FetchAndCache(name, docType, docURL, version string, force bool) (id int, entries []DocEntry, refreshed bool, err error) {
	id, err = AddSource(name, docType, docURL, version)
	if err != nil {
		return 0, nil, false, err
	}
	var lastUpdated string
	if err = db.DB.QueryRow("SELECT COALESCE(last_updated,'') FROM doc_sources WHERE id = ?", id).Scan(&lastUpdated); err != nil {
		return id, nil, false, err
	}
	if force || SourceNeedsRefresh(lastUpdated) {
		if err = UpdateSource(id); err != nil {
			return id, nil, false, err
		}
		refreshed = true
	} else {
		go EmbedSource(id)
	}
	entries, err = ListEntriesBySource(id)
	return id, entries, refreshed, err
}

func ListEntriesBySource(sourceID int) ([]DocEntry, error) {
	rows, err := db.DB.Query(`
		SELECT id, source_id, title, content, COALESCE(path,''), COALESCE(content_hash,''), updated_at
		FROM doc_content WHERE source_id = ? ORDER BY id`, sourceID)
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

func fetchDocs(docURL, docType string) ([]DocEntry, error) {
	parsedURL, err := url.Parse(docURL)
	if err != nil {
		return nil, err
	}

	switch docType {
	case "markdown", "md":
		return fetchMarkdown(parsedURL)
	case "html", "webpage":
		return fetchHTML(parsedURL)
	case "json", "api":
		return fetchJSONDocs(parsedURL)
	default:
		return fetchMarkdown(parsedURL)
	}
}

func fetchMarkdown(u *url.URL) ([]DocEntry, error) {
	body, err := fetchURL(u.String())
	if err != nil {
		return nil, err
	}
	return chunkMarkdown(string(body), u.Path), nil
}

func fetchHTML(u *url.URL) ([]DocEntry, error) {
	body, err := fetchURL(u.String())
	if err != nil {
		return nil, err
	}
	entries := chunkHTML(string(body), u.Path)
	if len(entries) == 0 {
		return nil, fmt.Errorf("no extractable content from %s", u.String())
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

func fetchURL(raw string) ([]byte, error) {
	resp, err := http.Get(raw)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
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
