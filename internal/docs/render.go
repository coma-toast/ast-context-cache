package docs

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const renderScriptName = "fetch-rendered-page.py"

var (
	renderScriptOnce sync.Once
	renderScriptPath string
	renderScriptErr  error

	playwrightProbeOnce sync.Once
	playwrightAvailable bool
)

// RenderDisabled reports whether DOC_RENDER_DISABLE opts out of Playwright rendering.
func RenderDisabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("DOC_RENDER_DISABLE")))
	return v == "1" || v == "true" || v == "yes"
}

// RenderEnabled reports whether JS rendering via Playwright is available.
func RenderEnabled() bool {
	if RenderDisabled() {
		return false
	}
	if _, err := findRenderScript(); err != nil {
		return false
	}
	return probePlaywrightImport()
}

func probePlaywrightImport() bool {
	playwrightProbeOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, renderPython(), "-c", "import playwright")
		playwrightAvailable = cmd.Run() == nil
	})
	return playwrightAvailable
}

// ResetRenderProbeForTest clears cached Playwright availability (tests only).
func ResetRenderProbeForTest() {
	playwrightProbeOnce = sync.Once{}
	playwrightAvailable = false
	renderScriptOnce = sync.Once{}
	renderScriptPath = ""
	renderScriptErr = nil
}

func findRenderScript() (string, error) {
	renderScriptOnce.Do(func() {
		var candidates []string
		if p := strings.TrimSpace(os.Getenv("DOC_RENDER_SCRIPT")); p != "" {
			candidates = append(candidates, p)
		}
		if root := strings.TrimSpace(os.Getenv("AST_MCP_ROOT")); root != "" {
			candidates = append(candidates, filepath.Join(root, "scripts", renderScriptName))
		}
		if exe, err := os.Executable(); err == nil {
			dir := filepath.Dir(exe)
			candidates = append(candidates,
				filepath.Join(dir, "scripts", renderScriptName),
				filepath.Join(dir, "..", "scripts", renderScriptName),
			)
		}
		if wd, err := os.Getwd(); err == nil {
			candidates = append(candidates, filepath.Join(wd, "scripts", renderScriptName))
		}
		for _, c := range candidates {
			if st, err := os.Stat(c); err == nil && !st.IsDir() {
				renderScriptPath = c
				return
			}
		}
		renderScriptErr = fmt.Errorf("playwright render script not found (run: playwright install firefox; or set DOC_RENDER_SCRIPT)")
	})
	return renderScriptPath, renderScriptErr
}

func renderPython() string {
	if p := strings.TrimSpace(os.Getenv("DOC_PYTHON")); p != "" {
		return p
	}
	return "python3"
}

func renderTimeout() time.Duration {
	if s := strings.TrimSpace(os.Getenv("DOC_RENDER_TIMEOUT")); s != "" {
		if d, err := time.ParseDuration(s); err == nil && d > 0 {
			return d
		}
	}
	return 90 * time.Second
}

func fetchRenderedURL(raw string) ([]byte, error) {
	script, err := findRenderScript()
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), renderTimeout())
	defer cancel()
	cmd := exec.CommandContext(ctx, renderPython(), script, "--url", raw)
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("render timeout after %s", renderTimeout())
		}
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			return nil, fmt.Errorf("render failed: %s", strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, fmt.Errorf("render failed: %w", err)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("render returned empty body for %s", raw)
	}
	return out, nil
}

// NormalizeDocType maps render_js requests to webpage for persistent caching.
// When Playwright is unavailable, webpage/render_js requests downgrade to html.
func NormalizeDocType(docType string, renderJS bool) string {
	docType = strings.ToLower(strings.TrimSpace(docType))
	wantRender := renderJS || docType == "webpage"
	if wantRender {
		if RenderEnabled() {
			return "webpage"
		}
		if docType == "webpage" || docType == "" || renderJS {
			return "html"
		}
	}
	switch docType {
	case "md":
		return "markdown"
	case "api":
		return "json"
	case "":
		return "markdown"
	default:
		return docType
	}
}

func docTypeUsesRender(docType string) bool {
	return strings.ToLower(strings.TrimSpace(docType)) == "webpage"
}

func entriesSparse(entries []DocEntry) bool {
	n := 0
	for _, e := range entries {
		n += len(strings.TrimSpace(e.Content))
	}
	return n < 200
}
