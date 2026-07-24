package components

import "time"

type Project struct {
	Path           string
	Name           string
	Label          string
	Workspace      string
	Branch         string
	RepoKey        string
	QueryCount     int
	SymbolCount    int
	FileCount      int
	Pinned         bool
	LinkedChildren []string
	LinkedParent   string
}

type HealthInfo struct {
	Version   string
	StartTime time.Time
}
