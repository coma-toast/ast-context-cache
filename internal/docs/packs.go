package docs

// PackSource is one curated documentation URL for a starter pack install.
type PackSource struct {
	Name    string `json:"name"`
	Type    string `json:"type"`
	URL     string `json:"url"`
	Version string `json:"version,omitempty"`
}

// InstallPackResult reports the outcome of adding one pack source.
type InstallPackResult struct {
	Name string `json:"name"`
	ID   int    `json:"id"`
	Err  string `json:"error,omitempty"`
}

// StarterPack is a small curated set of common library docs for fresh installs.
// Types match fetch_doc / add_doc_source (markdown, html, webpage, json).
var StarterPack = []PackSource{
	{Name: "React Learn", Type: "html", URL: "https://react.dev/learn"},
	{Name: "Go Effective Go", Type: "html", URL: "https://go.dev/doc/effective_go"},
	{Name: "MUI Getting Started", Type: "html", URL: "https://mui.com/material-ui/getting-started/"},
	{Name: "TypeScript Handbook", Type: "html", URL: "https://www.typescriptlang.org/docs/handbook/intro.html"},
}

// InstallStarterPack registers each StarterPack entry via AddSource and queues a force refresh.
// Existing (name, type, url) rows are upserted; refresh is always queued. Returns how many
// sources were successfully registered.
func InstallStarterPack() (added int, results []InstallPackResult) {
	results = make([]InstallPackResult, 0, len(StarterPack))
	for _, s := range StarterPack {
		r := InstallPackResult{Name: s.Name}
		id, err := AddSource(s.Name, s.Type, s.URL, s.Version)
		if err != nil {
			r.Err = err.Error()
			results = append(results, r)
			continue
		}
		r.ID = id
		ForceRefreshSource(id)
		added++
		results = append(results, r)
	}
	return added, results
}
