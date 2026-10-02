package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

func TestHandleProjectExcludesSavesAndPurges(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(home, "configSync")
	dup := filepath.Join(p, "llama-cpp-tq-tom", "main.go")
	keep := filepath.Join(p, "main.go")
	for _, f := range []string{dup, keep} {
		if err := os.MkdirAll(filepath.Dir(f), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(f, []byte("package p\n\nfunc F() {}\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := indexer.IndexFile(f, p); err != nil {
			t.Fatal(err)
		}
	}

	body := `{"project_path":"` + p + `","patterns":["llama-cpp-tq-tom/","  ",""]}`
	rec := httptest.NewRecorder()
	handleProjectExcludes(rec, httptest.NewRequest(http.MethodPost, "/api/project-excludes", strings.NewReader(body)))
	var resp struct {
		Status   string   `json:"status"`
		Patterns []string `json:"patterns"`
		Error    string   `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Status != "ok" || len(resp.Patterns) != 1 || resp.Patterns[0] != "llama-cpp-tq-tom/" {
		t.Fatalf("response %+v", resp)
	}
	if got := buildSettingsData(settingsBuildOpts{}).ProjectIndexExcludes[p]; len(got) != 1 {
		t.Fatalf("settings data excludes = %v", got)
	}

	deadline := time.Now().Add(5 * time.Second)
	for {
		files := db.GetIndexedFiles(p)
		_, dupLeft := files[dup]
		if _, ok := files[keep]; !ok {
			t.Fatal("non-excluded file purged")
		}
		if !dupLeft {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("excluded file not purged after saving per-project excludes")
		}
		time.Sleep(20 * time.Millisecond)
	}
	_ = db.SetProjectIndexExcludes(p, nil)
}
