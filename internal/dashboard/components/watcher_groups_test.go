package components

import "testing"

func TestGroupWatchers(t *testing.T) {
	ws := []WatcherInfo{
		{ProjectPath: "/Users/jason/git/slapi", Name: "slapi", Label: "slapi", Active: true},
		{ProjectPath: "/Users/jason/spaces/nightly/slapi", Name: "slapi", Label: "slapi · nightly", Workspace: "nightly", Active: true},
		{ProjectPath: "/Users/jason/spaces/pipeline/slapi", Name: "slapi", Label: "slapi · pipeline", Workspace: "pipeline", Active: true},
	}
	local, spaces := GroupWatchers(ws)
	if len(local) != 1 || local[0].Name != "slapi" {
		t.Fatalf("local=%+v", local)
	}
	if len(spaces) != 2 || spaces[0].Space != "nightly" || spaces[1].Space != "pipeline" {
		t.Fatalf("spaces=%+v", spaces)
	}
}
