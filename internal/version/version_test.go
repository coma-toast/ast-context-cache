package version

import "testing"

func TestVersionFromEmbed(t *testing.T) {
	if Version == "" || Version == "dev" {
		t.Fatalf("expected VERSION file embedded, got %q", Version)
	}
}
