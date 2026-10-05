package dashboard

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

func TestPanelUsesIndexDB(t *testing.T) {
	if !panelUsesIndexDB("symbol-chart") {
		t.Fatal("symbol-chart should use index db")
	}
	if panelUsesIndexDB("index-health") {
		t.Fatal("index-health should not be blocked")
	}
}

func TestFlushPartialSkipsIndexDBDuringMaintenance(t *testing.T) {
	db.BeginWALMaintenanceForTest("test")
	defer db.EndWALMaintenanceForTest()

	blocked := db.WALMaintenanceActive() && panelUsesIndexDB("symbol-chart")
	if !blocked {
		t.Fatal("expected symbol-chart blocked during maintenance")
	}
	allowed := !(db.WALMaintenanceActive() && panelUsesIndexDB("index-health"))
	if !allowed {
		t.Fatal("index-health should not be blocked")
	}
}

func TestHandoffsPanelMatchesOnlyHandoffs(t *testing.T) {
	if !panelMatchesMask("handoffs", realtime.Handoffs) {
		t.Fatal("handoffs panel should refresh on realtime.Handoffs")
	}
	if panelMatchesMask("handoffs", realtime.QueryLogged|realtime.SettingsChanged) {
		t.Fatal("handoffs panel should ignore unrelated reasons")
	}
	if panelUsesIndexDB("handoffs") {
		t.Fatal("handoffs reads context.db, not the index")
	}
}
