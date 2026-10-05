package embedder

import (
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	modelONNXURL     = "https://huggingface.co/onnx-models/all-mpnet-base-v2-onnx/resolve/main/model.onnx"
	tokenizerJSONURL = "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json"
)

// EnsureModel creates modelDir if needed and downloads model.onnx and tokenizer.json if missing.
// Call before New() so the embedder can load. Returns nil if both files exist or were downloaded successfully.
func EnsureModel(modelDir string) error {
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return errs.WrapMessage("failed to create model dir", err, "path", modelDir)
	}

	onnxPath := filepath.Join(modelDir, "model.onnx")
	tokPath := filepath.Join(modelDir, "tokenizer.json")

	missingONNX, err := fileExists(onnxPath)
	if err != nil {
		return errs.WrapMessage("failed to check model.onnx", err, "path", onnxPath)
	}
	missingTok, err := fileExists(tokPath)
	if err != nil {
		return errs.WrapMessage("failed to check tokenizer.json", err, "path", tokPath)
	}
	if !missingONNX && !missingTok {
		return nil
	}

	if missingONNX {
		logger.Info("Downloading model.onnx (this may take a minute)", "path", onnxPath)
		if err := downloadFile(modelONNXURL, onnxPath); err != nil {
			return errs.WrapMessage("failed to download model.onnx", err)
		}
		logger.Info("Model file ready", "file", "model.onnx")
	}
	if missingTok {
		logger.Info("Downloading tokenizer.json", "path", tokPath)
		if err := downloadFile(tokenizerJSONURL, tokPath); err != nil {
			return errs.WrapMessage("failed to download tokenizer.json", err)
		}
		logger.Info("Model file ready", "file", "tokenizer.json")
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
		return errs.New("unexpected download status", "url", url, "status", resp.Status)
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
	logger.Info("Wrote model file", "path", dest, "bytes", written)
	return nil
}
