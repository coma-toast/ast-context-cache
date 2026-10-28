package mcp

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSpaceFileProject(t *testing.T) {
	space := t.TempDir()
	repo := filepath.Join(space, "repo")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(space, "go.work"), []byte("go 1.23\n"), 0o644)
	os.WriteFile(filepath.Join(repo, ".git"), []byte("gitdir: x\n"), 0o644)
	plain := t.TempDir()
	tests := []struct {
		name, file, project, want string
	}{
		{"file in a checkout belongs to the checkout", filepath.Join(repo, "a.go"), space, repo},
		{"loose file at a space root has no project", filepath.Join(space, "plan.md"), space, ""},
		{"plain project is unchanged", filepath.Join(plain, "a.go"), plain, plain},
		{"checkout project is unchanged", filepath.Join(repo, "a.go"), repo, repo},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := spaceFileProject(tt.file, tt.project); got != tt.want {
				t.Fatalf("spaceFileProject=%q want %q", got, tt.want)
			}
		})
	}
}
