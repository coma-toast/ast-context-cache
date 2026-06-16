package embedder

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestReloadSwapsWiredSnapshot(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	vec := make([]float32, 768)
	for i := range vec {
		vec[i] = 1
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(httpEmbedResp{Embeddings: [][]float32{vec}})
	}))
	defer srv.Close()
	SetRuntimeHooks(RuntimeHooks{})
	t.Setenv("EMBED_BACKEND", "http")
	t.Setenv("EMBED_HTTP_URL", srv.URL)
	if err := InitRuntime(""); err != nil {
		t.Fatalf("InitRuntime: %v", err)
	}
	wb, _, _, ep, _ := WiredSnapshot()
	if wb != "http" || ep != srv.URL {
		t.Fatalf("wired=%s %s", wb, ep)
	}
	alt := srv.URL + "/alt"
	t.Setenv("EMBED_HTTP_URL", alt)
	if err := Reload(); err != nil {
		t.Fatalf("Reload: %v", err)
	}
	_, _, _, ep2, _ := WiredSnapshot()
	if ep2 != alt {
		t.Fatalf("wired endpoint=%q want %s", ep2, alt)
	}
}
