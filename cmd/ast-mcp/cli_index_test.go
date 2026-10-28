package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeDashboard answers the dashboard API the index and prune-projects commands
// call, recording each request body.
func fakeDashboard(t *testing.T, responses map[string]string) *[]map[string]interface{} {
	t.Helper()
	var bodies []map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		body["_path"] = r.URL.Path
		bodies = append(bodies, body)
		resp, ok := responses[r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write([]byte(resp))
	}))
	t.Cleanup(srv.Close)
	t.Setenv("AST_DASHBOARD_URL", srv.URL)
	return &bodies
}

func TestIndexCommandPostsEachDirectory(t *testing.T) {
	dir := t.TempDir()
	bodies := fakeDashboard(t, map[string]string{
		indexProjectPath: `{"status":"indexed","symbols":42,"projects":["` + dir + `/a","` + dir + `/b"]}`,
	})
	code, out, _ := runCLITest(t, "index", dir)
	require.Equal(t, exitOK, code)
	require.Len(t, *bodies, 1)
	assert.Equal(t, dir, (*bodies)[0]["project_path"])
	assert.Contains(t, out, "indexed "+dir+": 42 symbols")
	assert.Contains(t, out, "checkout "+dir+"/a")

	code, _, _ = runCLITest(t, "index")
	assert.Equal(t, exitError, code)
}

func TestPruneProjectsDefaultsToDryRun(t *testing.T) {
	bodies := fakeDashboard(t, map[string]string{
		pruneSpaceRootsPath: `{"apply":false,"space_roots":[
			{"path":"/s/covered","files":10,"symbols":90,"checkouts":["/s/covered/api"],"covered":true},
			{"path":"/s/partial","files":3,"symbols":9,"checkouts":["/s/partial/api","/s/partial/web"],"unindexed":["/s/partial/web"],"covered":false}]}`,
	})
	code, out, _ := runCLITest(t, "prune-projects", "--space-roots")
	require.Equal(t, exitOK, code)
	require.Len(t, *bodies, 1)
	assert.Equal(t, false, (*bodies)[0]["apply"])
	assert.Contains(t, out, "/s/covered  files=10 symbols=90 checkouts=1  covered")
	assert.Contains(t, out, "not indexed: /s/partial/web")
	assert.Contains(t, out, "dry run: re-run with --apply to prune 1 covered space root(s)")

	code, _, _ = runCLITest(t, "prune-projects", "--space-roots", "--apply")
	require.Equal(t, exitOK, code)
	assert.Equal(t, true, (*bodies)[1]["apply"])

	code, _, _ = runCLITest(t, "prune-projects")
	assert.Equal(t, exitError, code, "--space-roots is required")
}
