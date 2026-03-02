package embedder

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

const (
	modelONNXURL     = "https://huggingface.co/onnx-models/all-mpnet-base-v2-onnx/resolve/main/model.onnx"
	tokenizerJSONURL = "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json"
)

// EnsureModel creates modelDir if needed and downloads model.onnx and tokenizer.json if missing.
// Call before New() so the embedder can load. Returns nil if both files exist or were downloaded successfully.
func EnsureModel(modelDir string) error {
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("create model dir: %w", err)
	}

	onnxPath := filepath.Join(modelDir, "model.onnx")
	tokPath := filepath.Join(modelDir, "tokenizer.json")

	missingONNX, err := fileExists(onnxPath)
	if err != nil {
		return fmt.Errorf("check model.onnx: %w", err)
	}
	missingTok, err := fileExists(tokPath)
	if err != nil {
		return fmt.Errorf("check tokenizer.json: %w", err)
	}
	if !missingONNX && !missingTok {
		return nil
	}

	if missingONNX {
		log.Printf("Embedder: downloading model.onnx (this may take a minute)...")
		if err := downloadFile(modelONNXURL, onnxPath); err != nil {
			return fmt.Errorf("download model.onnx: %w", err)
		}
		log.Printf("Embedder: model.onnx ready")
	}
	if missingTok {
		log.Printf("Embedder: downloading tokenizer.json...")
		if err := downloadFile(tokenizerJSONURL, tokPath); err != nil {
			return fmt.Errorf("download tokenizer.json: %w", err)
		}
		log.Printf("Embedder: tokenizer.json ready")
	}

	return nil
}

func fileExists(path string) (missing bool, err error) {
	_, err = os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return true, nil
		}
		return true, err
	}
	return false, nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	written, err := io.Copy(f, resp.Body)
	if err != nil {
		os.Remove(dest)
		return err
	}
	log.Printf("Embedder: wrote %s (%d bytes)", dest, written)
	return nil
}
