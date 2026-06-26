package components

import "github.com/coma-toast/ast-context-cache/internal/startup"

func (h IndexHealth) StartupMessage() string {
	if !startup.Starting() {
		return ""
	}
	return startup.Message()
}

func (h IndexHealth) ShowStartupBanner() bool {
	return startup.Starting()
}

func (h Health) StartupMessage() string {
	if !startup.Starting() {
		return ""
	}
	return startup.Message()
}

func (h Health) ShowStartupLoading() bool {
	return h.EmbedderState == "loading" || startup.Starting()
}
