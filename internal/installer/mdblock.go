package installer

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Markdown blocks are delimited by version-stamped markers (IN-10). The sha in the begin marker
// is the hash of the body as written, so a later mismatch means the user edited the block.
const (
	blockBeginPrefix = "<!-- ast-context-cache:begin"
	blockEndMarker   = "<!-- ast-context-cache:end -->"
)

var blockBeginRe = regexp.MustCompile(`^<!-- ast-context-cache:begin v=(\S+) sha=([0-9a-f]+) -->$`)

// mdBlock is a located marker block. start and end are byte offsets of the begin marker and the
// end of the end marker (its line break excluded).
type mdBlock struct {
	start, end   int
	version, sha string
	body         string
}

// renderBlock returns the marker block for body, stamped with version, using newline nl.
func renderBlock(version, body, nl string) string {
	b := normalizeBody(body)
	s := blockBeginPrefix + " v=" + version + " sha=" + blockHash(b) + " -->\n" + b + "\n" + blockEndMarker
	if nl != "\n" {
		s = strings.ReplaceAll(s, "\n", nl)
	}
	return s
}

// findBlock locates our block. A second begin marker, a malformed begin marker, or a missing end
// marker is an error: editing would risk the user's text around it.
func findBlock(data string) (mdBlock, bool, error) {
	i := strings.Index(data, blockBeginPrefix)
	if i < 0 {
		return mdBlock{}, false, nil
	}
	if strings.Count(data, blockBeginPrefix) > 1 {
		return mdBlock{}, false, errs.NewCode(errs.CodeConflict, "more than one ast-context-cache block")
	}
	lineEnd := strings.IndexByte(data[i:], '\n')
	if lineEnd < 0 {
		return mdBlock{}, false, errs.NewCode(errs.CodeInvalidInput, "ast-context-cache begin marker without end marker")
	}
	m := blockBeginRe.FindStringSubmatch(strings.TrimRight(data[i:i+lineEnd], "\r"))
	if m == nil {
		return mdBlock{}, false, errs.NewCode(errs.CodeInvalidInput, "malformed ast-context-cache begin marker")
	}
	j := strings.Index(data[i+lineEnd:], blockEndMarker)
	if j < 0 {
		return mdBlock{}, false, errs.NewCode(errs.CodeInvalidInput, "ast-context-cache begin marker without end marker")
	}
	bodyEnd := i + lineEnd + j
	return mdBlock{start: i, end: bodyEnd + len(blockEndMarker), version: m[1], sha: m[2], body: normalizeBody(data[i+lineEnd+1 : bodyEnd])}, true, nil
}

// upsertBlock replaces our block in data, or appends it after a blank line.
func upsertBlock(data, block, nl string) (string, error) {
	b, ok, err := findBlock(data)
	if err != nil {
		return "", err
	}
	if ok {
		return data[:b.start] + block + data[b.end:], nil
	}
	if data == "" {
		return block + nl, nil
	}
	if !strings.HasSuffix(data, "\n") {
		data += nl
	}
	return data + nl + block + nl, nil
}

// removeBlock deletes our block with its line break, and the blank separator line upsertBlock
// added when the block was last in the file.
func removeBlock(data string) (string, bool, error) {
	b, ok, err := findBlock(data)
	if err != nil || !ok {
		return data, false, err
	}
	before, after := data[:b.start], data[b.end:]
	if strings.HasPrefix(after, "\r\n") {
		after = after[2:]
	} else {
		after = strings.TrimPrefix(after, "\n")
	}
	if after == "" {
		switch {
		case strings.HasSuffix(before, "\r\n\r\n"):
			before = before[:len(before)-2]
		case strings.HasSuffix(before, "\n\n"):
			before = before[:len(before)-1]
		}
	}
	return before + after, true, nil
}

// blockStatus compares a located block with the body this version would write.
func blockStatus(b mdBlock, desiredBody, version string) Status {
	switch {
	case blockHash(b.body) != b.sha:
		return StatusModifiedByUser
	case b.body != normalizeBody(desiredBody) || b.version != version:
		return StatusOutdated
	default:
		return StatusInstalled
	}
}

func blockHash(body string) string {
	sum := sha256.Sum256([]byte(normalizeBody(body)))
	return hex.EncodeToString(sum[:])[:12]
}

func normalizeBody(s string) string {
	return strings.Trim(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
}

// newlineOf returns the file's newline style.
func newlineOf(data string) string {
	if strings.Contains(data, "\r\n") {
		return "\r\n"
	}
	return "\n"
}

// splitFrontmatter splits a "---" YAML frontmatter header from the rest of a file.
func splitFrontmatter(s string) (string, string) {
	if !strings.HasPrefix(s, "---\n") {
		return "", s
	}
	end := strings.Index(s[4:], "\n---\n")
	if end < 0 {
		return "", s
	}
	cut := 4 + end + len("\n---\n")
	return s[:cut], s[cut:]
}
