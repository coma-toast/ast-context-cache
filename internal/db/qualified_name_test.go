package db

import "testing"

func TestQualifiedName(t *testing.T) {
	for _, c := range []struct{ fqn, file, name, want string }{
		{"llamacpp.py.LlamaCppClient.load_model", "/p/clients/llamacpp.py", "load_model", "LlamaCppClient.load_model"},
		{"llamacpp.py.load_model", "/p/llamacpp.py", "load_model", "load_model"},
		{"server.go.Server.Handle", "/p/server.go", "Handle", "Server.Handle"},
		// Plaintext rows store "<path>#plaintext"; other shapes fall back to the name.
		{"/p/app.log#plaintext", "/p/app.log", "app.log", "app.log"},
		{"pkg.VectorCache", "/p/cache.go", "VectorCache", "VectorCache"},
		{"", "/p/a.py", "f", "f"},
	} {
		if got := QualifiedName(c.fqn, c.file, c.name); got != c.want {
			t.Errorf("QualifiedName(%q, %q, %q) = %q, want %q", c.fqn, c.file, c.name, got, c.want)
		}
	}
}
