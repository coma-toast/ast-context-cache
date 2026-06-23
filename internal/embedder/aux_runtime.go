package embedder

import (
	"fmt"
	"log"
	"sync"
)

const embedAuxBackendSetting = "EMBED_AUX_BACKEND"

var (
	auxMu          sync.Mutex
	auxModelDir    string
	auxInitialized bool
	auxRaw         Interface
	auxBackend     string
	auxModel       string
)

// AuxBackend returns the configured catch-up embedder backend (default onnx).
func AuxBackend() string {
	b := normalizeEmbedBackend(EffectiveEnv(embedAuxBackendSetting))
	if b == "" {
		return "onnx"
	}
	return b
}

// InitAuxRuntime wires the auxiliary embed queue embedder from its stored profile.
func InitAuxRuntime(modelDir string) error {
	auxModelDir = modelDir
	auxInitialized = true
	return reloadAuxRuntime()
}

// RawAux returns the auxiliary embedder (queue workers only; no health tracking).
func RawAux() Interface {
	auxMu.Lock()
	defer auxMu.Unlock()
	return auxRaw
}

// AuxSnapshot returns display metadata for the auxiliary embedder.
func AuxSnapshot() (backend, model string) {
	auxMu.Lock()
	defer auxMu.Unlock()
	return auxBackend, auxModel
}

// AuxSharesPrimary reports whether aux uses the same backend as the primary embedder.
func AuxSharesPrimary() bool {
	return normalizeEmbedBackend(AuxBackend()) == normalizeEmbedBackend(EffectiveEnv("EMBED_BACKEND"))
}

// AuxRuntimeReady reports whether InitAuxRuntime has completed.
func AuxRuntimeReady() bool {
	auxMu.Lock()
	defer auxMu.Unlock()
	return auxInitialized
}

// ReloadAuxRuntime rebuilds the auxiliary embedder from current settings.
func ReloadAuxRuntime() error {
	return reloadAuxRuntime()
}

func reloadAuxRuntime() error {
	if !auxInitialized {
		return fmt.Errorf("aux embedder runtime not initialized")
	}
	s := SettingsForStoredProfile(AuxBackend())
	raw, err := NewFromSettings(s, auxModelDir)
	if err != nil {
		return err
	}
	b, m, _, _, _ := SnapshotForSettings(s)
	auxMu.Lock()
	auxRaw = raw
	auxBackend = b
	auxModel = m
	auxMu.Unlock()
	log.Printf("Aux embedder active: backend=%s model=%s", b, m)
	return nil
}
