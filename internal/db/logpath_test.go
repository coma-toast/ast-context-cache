package db

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestResolveServerLogPathEnv(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", "/custom/ast.log")
	if got := ResolveServerLogPath(); got != "/custom/ast.log" {
		t.Fatalf("got %q", got)
	}
}

func TestResolveServerLogPathNewest(t *testing.T) {
	t.Setenv("AST_MCP_LOG_PATH", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	defaultPath := DefaultLogPath()
	mcpPath := McpLocalLogPath()
	os.MkdirAll(filepath.Dir(defaultPath), 0755)
	os.MkdirAll(filepath.Dir(mcpPath), 0755)
	if err := os.WriteFile(defaultPath, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(mcpPath, []byte("newer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := ResolveServerLogPath(); got != mcpPath {
		t.Fatalf("got %q want %q", got, mcpPath)
	}
}

func TestThrottledEmbedWorkers(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	if got := ThrottledEmbedWorkers(10); got != 10 {
		t.Fatalf("ok=%d", got)
	}
}
