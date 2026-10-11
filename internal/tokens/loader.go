package tokens

import (
	"bytes"
	"encoding/base64"
	"path"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// embeddedLoader serves the embedded o200k_base vocabulary. It satisfies
// tiktoken.BpeLoader so tiktoken-go can be pointed at it with SetBpeLoader.
type embeddedLoader struct{}

// LoadTiktokenBpe returns the mergeable ranks of the embedded vocabulary. Only
// o200k_base is embedded; any other file is an error.
func (embeddedLoader) LoadTiktokenBpe(tiktokenBpeFile string) (map[string]int, error) {
	if name := path.Base(tiktokenBpeFile); name != encodingName+".tiktoken" {
		return nil, errs.New("vocabulary not embedded", "file", name)
	}
	data, err := vocabulary()
	if err != nil {
		return nil, err
	}
	return parseRanks(data)
}

// parseRanks parses "<base64 token> <rank>" lines into a token-to-rank map.
func parseRanks(data []byte) (map[string]int, error) {
	ranks := make(map[string]int, bytes.Count(data, []byte{'\n'})+1)
	buf := make([]byte, 0, 64)
	for line := range bytes.SplitSeq(data, []byte{'\n'}) {
		if len(line) == 0 {
			continue
		}
		tok, rank, ok := bytes.Cut(line, []byte{' '})
		if !ok {
			return nil, errs.New("malformed vocabulary line", "line", string(line))
		}
		var err error
		buf, err = base64.StdEncoding.AppendDecode(buf[:0], tok)
		if err != nil {
			return nil, errs.WrapMessage("unable to decode vocabulary token", err, "token", string(tok))
		}
		r, err := strconv.Atoi(string(rank))
		if err != nil {
			return nil, errs.WrapMessage("unable to parse vocabulary rank", err, "rank", string(rank))
		}
		ranks[string(buf)] = r
	}
	return ranks, nil
}
