package dashboard

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"
)

//go:embed ui/dist/*
var uiAssets embed.FS

var uiFS fs.FS

func initUIAssets() {
	sub, err := fs.Sub(uiAssets, "ui/dist")
	if err != nil {
		uiFS = uiAssets
		return
	}
	uiFS = sub
}

func handleUISPA(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/dashboard")
	if path == "" || path == "/" {
		path = "/index.html"
	}
	if !strings.Contains(path, ".") {
		path = "/index.html"
	}
	data, err := fs.ReadFile(uiFS, strings.TrimPrefix(path, "/"))
	if err != nil {
		data, err = fs.ReadFile(uiFS, "index.html")
		if err != nil {
			http.NotFound(w, r)
			return
		}
	}
	if strings.HasSuffix(path, ".html") {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
	} else if strings.HasSuffix(path, ".js") {
		w.Header().Set("Content-Type", "application/javascript")
	} else if strings.HasSuffix(path, ".css") {
		w.Header().Set("Content-Type", "text/css")
	}
	w.Write(data)
}

func handleRootRedirect(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	http.Redirect(w, r, "/dashboard/", http.StatusFound)
}
