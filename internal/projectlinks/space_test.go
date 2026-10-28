package projectlinks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSpaceCheckouts(t *testing.T) {
	// layout entries: "d/" is a directory, "d/.git/" a clone, "d/.git" a worktree
	// (.git file), anything else a regular file.
	tests := []struct {
		name   string
		layout []string
		want   []string
	}{
		{"space with go.work and two worktrees", []string{"go.work", "a/.git", "b/.git", "notes/"}, []string{"a", "b"}},
		{"space with SPACE.md and a clone", []string{"SPACE.md", "a/.git/"}, []string{"a"}},
		{"worktree child marks a space without markers", []string{"a/.git"}, []string{"a"}},
		{"folder of clones without markers is not a space", []string{"a/.git/", "b/.git/"}, nil},
		{"markers without checkouts is not a space", []string{"go.work", "notes/"}, nil},
		{"plain folder", []string{"notes/", "todo.md"}, nil},
		{"a checkout is never a space", []string{".git/", "go.work", "sub/.git"}, nil},
		{"hidden child checkouts are ignored", []string{"go.work", ".cache/.git/"}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, e := range tt.layout {
				p := filepath.Join(dir, e)
				if e[len(e)-1] == '/' {
					if err := os.MkdirAll(p, 0o755); err != nil {
						t.Fatal(err)
					}
					continue
				}
				if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(p, []byte("gitdir: x\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			got := SpaceCheckouts(dir)
			if len(got) != len(tt.want) {
				t.Fatalf("SpaceCheckouts=%v want %v", got, tt.want)
			}
			for i, w := range tt.want {
				if got[i] != filepath.Join(NormalizePath(dir), w) {
					t.Fatalf("SpaceCheckouts[%d]=%q want %q", i, got[i], w)
				}
			}
			targets := IndexTargets(dir)
			if len(tt.want) == 0 && (len(targets) != 1 || targets[0] != NormalizePath(dir)) {
				t.Fatalf("IndexTargets=%v want the folder itself", targets)
			}
			if len(tt.want) > 0 && len(targets) != len(tt.want) {
				t.Fatalf("IndexTargets=%v want the checkouts", targets)
			}
		})
	}
}
