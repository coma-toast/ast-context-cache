package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// handleProjects used to return every distinct project_path ever queried with
// no limit at all. It's now capped (like /api/recent) and ordered by activity
// — this confirms the query still returns the expected rows, correctly
// ordered, after adding ORDER BY/LIMIT.
func TestHandleProjectsOrdersByActivity(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	os.Unsetenv("DB_PATH")
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}

	if _, err := db.DB.Exec(`INSERT INTO queries (tool_name, project_path, timestamp) VALUES ('t', '/proj/quiet', datetime('now'))`); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if _, err := db.DB.Exec(`INSERT INTO queries (tool_name, project_path, timestamp) VALUES ('t', '/proj/busy', datetime('now'))`); err != nil {
			t.Fatal(err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/api/projects", nil)
	rec := httptest.NewRecorder()
	handleProjects(rec, req)

	var out []map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v (body=%s)", err, rec.Body.String())
	}
	if len(out) != 2 {
		t.Fatalf("got %d projects, want 2: %+v", len(out), out)
	}
	if out[0]["path"] != "/proj/busy" {
		t.Fatalf("expected busiest project first, got %+v", out[0])
	}
}
