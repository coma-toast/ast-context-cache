package docs

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

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
	result, err := db.DB.Exec(
		`INSERT INTO doc_sources (name, type, url, version) VALUES (?, ?, ?, ?)
		 ON CONFLICT(name, type, url) DO UPDATE SET version = excluded.version`,
		name, docType, docURL, version)
	if err != nil {
		return 0, err
	}
	id, _ := result.LastInsertId()
	return int(id), nil
}

func RemoveSource(id int) error {
	db.DB.Exec("DELETE FROM doc_content WHERE source_id = ?", id)
	_, err := db.DB.Exec("DELETE FROM doc_sources WHERE id = ?", id)
	return err
}

func ListSources() ([]DocSource, error) {
	rows, err := db.DB.Query("SELECT id, name, type, url, COALESCE(version,''), COALESCE(last_updated,''), created_at FROM doc_sources ORDER BY name")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sources []DocSource
	for rows.Next() {
		var s DocSource
		rows.Scan(&s.ID, &s.Name, &s.Type, &s.URL, &s.Version, &s.LastUpdated, &s.CreatedAt)
		sources = append(sources, s)
	}
	return sources, nil
}

func SearchDocs(query string, limit int) ([]DocEntry, error) {
	if limit <= 0 {
		limit = 10
	}
	rows, err := db.DB.Query(`
		SELECT dc.id, dc.source_id, dc.title, dc.content, COALESCE(dc.path,''), COALESCE(dc.content_hash,''), dc.updated_at
		FROM doc_content dc
		JOIN doc_sources ds ON dc.source_id = ds.id
		WHERE dc.title LIKE ? OR dc.content LIKE ?
		ORDER BY dc.updated_at DESC
		LIMIT ?
	`, "%"+query+"%", "%"+query+"%", limit)
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

	db.DB.Exec("DELETE FROM doc_content WHERE source_id = ?", id)

	for _, entry := range content {
		hash := contentHash(entry.Content)
		db.DB.Exec(
			`INSERT INTO doc_content (source_id, title, content, path, content_hash) VALUES (?, ?, ?, ?, ?)`,
			id, entry.Title, entry.Content, entry.Path, hash)
	}

	db.DB.Exec(`INSERT INTO docs_fts(docs_fts) VALUES('rebuild')`)
	db.DB.Exec("UPDATE doc_sources SET last_updated = ? WHERE id = ?", time.Now().Format(time.RFC3339), id)

	return nil
}

func UpdateAllSources() {
	sources, err := ListSources()
	if err != nil {
		return
	}

	for _, s := range sources {
		if s.LastUpdated != "" {
			lastTime, parseErr := time.Parse(time.RFC3339, s.LastUpdated)
			if parseErr == nil && time.Since(lastTime) < 24*time.Hour {
				continue
			}
		}
		UpdateSource(s.ID)
	}
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
	resp, err := http.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	content := string(body)
	sections := splitMarkdownSections(content)

	var entries []DocEntry
	for _, section := range sections {
		if section.Title != "" && section.Content != "" {
			entries = append(entries, DocEntry{
				Title:   section.Title,
				Content: section.Content,
				Path:    u.Path,
			})
		}
	}

	if len(entries) == 0 {
		entries = append(entries, DocEntry{
			Title:   u.Path,
			Content: content,
			Path:    u.Path,
		})
	}

	return entries, nil
}

func fetchHTML(u *url.URL) ([]DocEntry, error) {
	resp, err := http.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	content := extractTextFromHTML(string(body))
	return []DocEntry{
		{
			Title:   u.Path,
			Content: content,
			Path:    u.Path,
		},
	}, nil
}

func fetchJSONDocs(u *url.URL) ([]DocEntry, error) {
	resp, err := http.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return []DocEntry{
		{
			Title:   u.Path,
			Content: string(body),
			Path:    u.Path,
		},
	}, nil
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

	for _, line := range lines {
		if strings.HasPrefix(line, "# ") || strings.HasPrefix(line, "## ") || strings.HasPrefix(line, "### ") {
			if currentTitle != "" {
				sections = append(sections, markdownSection{
					Title:   currentTitle,
					Content: strings.TrimSpace(currentContent.String()),
				})
			}
			currentTitle = strings.TrimPrefix(strings.TrimPrefix(strings.TrimPrefix(line, "# "), "## "), "### ")
			currentContent.Reset()
		} else {
			currentContent.WriteString(line + "\n")
		}
	}

	if currentTitle != "" {
		sections = append(sections, markdownSection{
			Title:   currentTitle,
			Content: strings.TrimSpace(currentContent.String()),
		})
	}

	return sections
}

func extractTextFromHTML(html string) string {
	re := regexp.MustCompile(`<[^>]+>`)
	text := re.ReplaceAllString(html, " ")
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func contentHash(text string) string {
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:])
}
