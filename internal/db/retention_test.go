package db

import (
	"os"
	"testing"
	"time"
)

func TestRunQueryRetention(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	old := time.Now().AddDate(0, 0, -120).Format(time.RFC3339)
	newTS := time.Now().Format(time.RFC3339)
	DB.Exec(`INSERT INTO queries (timestamp, tool_name, result_chars, duration_ms, interface) VALUES (?, 'old', 1, 1, 'http')`, old)
	DB.Exec(`INSERT INTO queries (timestamp, tool_name, result_chars, duration_ms, interface) VALUES (?, 'new', 1, 1, 'http')`, newTS)
	_ = SetSetting("query_retention_enabled", "true")
	_ = SetSetting("query_retention_max_age_days", "90")
	if n := RunQueryRetention(); n != 1 {
		t.Fatalf("deleted=%d want 1", n)
	}
	var count int
	DB.QueryRow("SELECT COUNT(*) FROM queries").Scan(&count)
	if count != 1 {
		t.Fatalf("remaining=%d", count)
	}
}

func TestRunQueryRetentionDisabled(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	_ = SetSetting("query_retention_enabled", "false")
	if n := RunQueryRetention(); n != 0 {
		t.Fatalf("deleted=%d want 0", n)
	}
	_ = os.Getenv("HOME")
}
