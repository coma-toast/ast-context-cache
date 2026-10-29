package sys

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGuestAt(t *testing.T) {
	tests := []struct {
		name  string
		files map[string]string
		want  bool
	}{
		{"bare metal", map[string]string{"proc/cpuinfo": "processor\t: 0\nflags\t\t: fpu vme sse2\n"}, false},
		{"no proc", nil, false},
		{"lxc via systemd", map[string]string{"run/systemd/container": "lxc\n"}, true},
		{"docker", map[string]string{".dockerenv": ""}, true},
		{"podman", map[string]string{"run/.containerenv": ""}, true},
		{"vm", map[string]string{"proc/cpuinfo": "processor\t: 0\nflags\t\t: fpu sse2 hypervisor lahf_lm\n"}, true},
		{"hypervisor only as substring", map[string]string{"proc/cpuinfo": "flags\t\t: fpu nohypervisorx\n"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			for name, body := range tt.files {
				p := filepath.Join(root, name)
				if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if got := guestAt(root); got != tt.want {
				t.Errorf("guestAt = %v, want %v", got, tt.want)
			}
		})
	}
}
