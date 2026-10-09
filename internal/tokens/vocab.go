package tokens

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"io"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// o200kSHA256 is the published sha256 of o200k_base.tiktoken (openai/tiktoken load.py).
const o200kSHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"

//go:embed o200k_base.tiktoken.gz
var o200kGzip []byte

// vocabulary returns the decompressed o200k_base.tiktoken file.
func vocabulary() ([]byte, error) {
	zr, err := gzip.NewReader(bytes.NewReader(o200kGzip))
	if err != nil {
		return nil, errs.WrapMessage("unable to open embedded vocabulary", err, "encoding", encodingName)
	}
	defer zr.Close()
	data, err := io.ReadAll(zr)
	if err != nil {
		return nil, errs.WrapMessage("unable to decompress embedded vocabulary", err, "encoding", encodingName)
	}
	return data, nil
}
