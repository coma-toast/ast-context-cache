package instance

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"image/png"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Favicon modes stored in FaviconModeKey.
const (
	FaviconDefault = "default"
	FaviconColor   = "color"
	FaviconUpload  = "upload"
)

// Favicon settings keys. FaviconUploadKey holds the upload as a data URL
// ("data:<type>;base64,<bytes>"); FaviconOverrideKey works like OverrideKey.
const (
	FaviconModeKey     = "instance_favicon_mode"
	FaviconColorKey    = "instance_favicon_color"
	FaviconBadgeKey    = "instance_favicon_badge"
	FaviconUploadKey   = "instance_favicon_upload"
	FaviconOverrideKey = "instance_favicon_override"
)

// Favicon env vars: either one locks the favicon to color mode unless FaviconOverrideKey is on.
const (
	FaviconColorEnv = "AST_INSTANCE_FAVICON_COLOR"
	FaviconBadgeEnv = "AST_INSTANCE_FAVICON_BADGE"
)

// Upload limits and the content types an upload may have.
const (
	FaviconMaxBytes = 64 << 10
	FaviconPNG      = "image/png"
	FaviconSVG      = "image/svg+xml"
	FaviconICO      = "image/x-icon"
	// DefaultFaviconColor is the purple of today's icon, used by color mode without a color.
	DefaultFaviconColor = "#863bff"
)

// boltPath is the outline of today's icon (ui/public/favicon.svg, viewBox 0 0 48 46).
const boltPath = "M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z"

// FaviconSettings is what the dashboard stores for the favicon.
type FaviconSettings struct {
	Mode, Color, Badge string
	// Upload is the stored data URL, empty when nothing was uploaded.
	Upload   string
	Override bool
}

// Favicon is the resolved favicon: the mode in effect, where it came from
// (env|override|settings|default), and the bytes to serve.
type Favicon struct {
	Mode, Source, Color, Badge string
	ContentType                string
	Body                       []byte
}

// Version is a short hash of the served bytes, used as the ?v= cache buster.
func (f Favicon) Version() string {
	sum := sha256.Sum256(append([]byte(f.ContentType+"\x00"), f.Body...))
	return hex.EncodeToString(sum[:6])
}

var badEnvOnce sync.Map

// FaviconExternal returns the valid AST_INSTANCE_FAVICON_COLOR and
// AST_INSTANCE_FAVICON_BADGE values. An invalid value is ignored and logged once.
func FaviconExternal() (color, badge string) {
	envValue := func(name string, norm func(string) (string, error)) string {
		raw := strings.TrimSpace(os.Getenv(name))
		v, err := norm(raw)
		if err != nil && raw != "" {
			if _, seen := badEnvOnce.LoadOrStore(name+"="+raw, true); !seen {
				slog.Warn("Ignoring invalid favicon env var", "env", name, "value", raw, "error", err)
			}
			return ""
		}
		return v
	}
	return envValue(FaviconColorEnv, NormalizeFaviconColor), envValue(FaviconBadgeEnv, NormalizeFaviconBadge)
}

// FaviconLocked reports whether an env var sets the favicon. Without the override
// the favicon settings cannot be changed from the dashboard.
func FaviconLocked() bool {
	c, b := FaviconExternal()
	return c != "" || b != ""
}

// NormalizeFaviconColor accepts #rgb or #rrggbb (the # is optional) and returns
// lowercase #rrggbb. Empty stays empty.
func NormalizeFaviconColor(v string) (string, error) {
	v = strings.ToLower(strings.TrimPrefix(strings.TrimSpace(v), "#"))
	if v == "" {
		return "", nil
	}
	if strings.Trim(v, "0123456789abcdef") != "" || (len(v) != 3 && len(v) != 6) {
		return "", errs.New(fmt.Sprintf("favicon color %q must be a hex color like #3366ff", v))
	}
	if len(v) == 3 {
		v = string([]byte{v[0], v[0], v[1], v[1], v[2], v[2]})
	}
	return "#" + v, nil
}

// NormalizeFaviconBadge trims the badge and checks it is 0–2 letters or digits.
func NormalizeFaviconBadge(v string) (string, error) {
	v = strings.TrimSpace(v)
	if n := utf8.RuneCountInString(v); n > 2 {
		return "", errs.New(fmt.Sprintf("favicon badge must be 1–2 characters (got %d)", n))
	}
	for _, r := range v {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return "", errs.New(fmt.Sprintf("favicon badge may only use letters and digits (got %q)", r))
		}
	}
	return v, nil
}

// ValidateFaviconMode checks a FaviconModeKey value; empty means default.
func ValidateFaviconMode(v string) error {
	switch v {
	case "", FaviconDefault, FaviconColor, FaviconUpload:
		return nil
	}
	return errs.New(fmt.Sprintf("favicon mode %q must be default, color or upload", v))
}

// ParseFaviconUpload decodes a data URL upload, checks the size and sniffs the
// type from the bytes (a declared type must agree). SVG must pass
// CheckSafeSVG. It returns the normalized data URL to store, the type and the bytes.
func ParseFaviconUpload(dataURL string) (stored, contentType string, body []byte, err error) {
	rest, isData := strings.CutPrefix(strings.TrimSpace(dataURL), "data:")
	meta, payload, ok := strings.Cut(rest, ",")
	if !isData || !ok || !strings.HasSuffix(meta, ";base64") {
		return "", "", nil, errs.New("favicon upload must be a base64 data URL (data:image/png;base64,...)")
	}
	declared := strings.ToLower(strings.TrimSpace(strings.Split(meta, ";")[0]))
	if base64.StdEncoding.DecodedLen(len(payload)) > FaviconMaxBytes+3 {
		return "", "", nil, errs.New(fmt.Sprintf("favicon upload is about %d KB; the limit is 64 KB", base64.StdEncoding.DecodedLen(len(payload))>>10))
	}
	if body, err = base64.StdEncoding.DecodeString(payload); err != nil {
		return "", "", nil, errs.New("favicon upload is not valid base64")
	}
	if len(body) > FaviconMaxBytes {
		return "", "", nil, errs.New(fmt.Sprintf("favicon upload is %d bytes; the limit is 65536 (64 KB)", len(body)))
	}
	if contentType, err = sniffFavicon(body); err != nil {
		return "", "", nil, err
	}
	if !declaredMatches(declared, contentType) {
		return "", "", nil, errs.New(fmt.Sprintf("favicon upload declares %s but the file is %s; use a PNG, SVG or ICO file", declared, contentType))
	}
	return "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(body), contentType, body, nil
}

func declaredMatches(declared, sniffed string) bool {
	switch declared {
	case "", "application/octet-stream", sniffed:
		return true
	case "image/vnd.microsoft.icon", "image/ico", "image/icon":
		return sniffed == FaviconICO
	}
	return false
}

// sniffFavicon returns the type of a PNG, ICO or SVG upload, checking SVG with CheckSafeSVG.
func sniffFavicon(b []byte) (string, error) {
	switch {
	case bytes.HasPrefix(b, []byte("\x89PNG\r\n\x1a\n")):
		if _, err := png.DecodeConfig(bytes.NewReader(b)); err != nil {
			return "", errs.New("favicon upload looks like a PNG but does not decode: " + err.Error())
		}
		return FaviconPNG, nil
	case len(b) >= 6 && bytes.HasPrefix(b, []byte{0, 0, 1, 0}) && (b[4] != 0 || b[5] != 0):
		return FaviconICO, nil
	case utf8.Valid(b) && bytes.Contains(b, []byte("<svg")):
		if err := CheckSafeSVG(b); err != nil {
			return "", err
		}
		return FaviconSVG, nil
	}
	return "", errs.New("favicon upload must be a PNG, SVG or ICO file")
}

// svgDeniedElements can run script, embed other documents or fetch external content.
var svgDeniedElements = map[string]bool{
	"script": true, "foreignobject": true, "iframe": true, "object": true, "embed": true,
	"handler": true, "listener": true, "audio": true, "video": true, "image": true, "feimage": true,
}

// CheckSafeSVG rejects an SVG that could run script or load anything external when
// opened directly: DOCTYPE/ENTITY declarations, script-capable elements (script,
// foreignObject, iframe, image, ...), XHTML-namespace elements, on* event handler
// attributes, href values other than local #fragments, javascript: values, url()
// references other than url(#...), animations that set href or on* attributes,
// and @import in style. Rejecting is safer than rewriting: there is no partial
// SVG left behind that a missed rule could still exploit.
func CheckSafeSVG(b []byte) error {
	dec := xml.NewDecoder(bytes.NewReader(b))
	dec.Strict = true
	depth, inStyle := 0, false
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return errs.New("SVG favicon is not well-formed XML: " + err.Error())
		}
		switch t := tok.(type) {
		case xml.Directive:
			return errs.New("SVG favicon may not contain DOCTYPE or ENTITY declarations")
		case xml.ProcInst:
			if t.Target != "xml" {
				return errs.New("SVG favicon may not contain <?" + t.Target + "?> processing instructions")
			}
		case xml.StartElement:
			name := strings.ToLower(t.Name.Local)
			if depth == 0 && name != "svg" {
				return errs.New("favicon upload is not an SVG (root element is <" + t.Name.Local + ">)")
			}
			if svgDeniedElements[name] || t.Name.Space == "http://www.w3.org/1999/xhtml" {
				return errs.New("SVG favicon may not contain <" + t.Name.Local + "> elements")
			}
			for _, a := range t.Attr {
				if err := checkSVGAttr(name, a); err != nil {
					return err
				}
			}
			depth, inStyle = depth+1, name == "style"
		case xml.EndElement:
			depth, inStyle = depth-1, false
		case xml.CharData:
			if inStyle {
				if err := checkSVGValue("<style>", string(t)); err != nil {
					return err
				}
				if strings.Contains(strings.ToLower(string(t)), "@import") {
					return errs.New("SVG favicon <style> may not use @import")
				}
			}
		}
	}
	if depth != 0 {
		return errs.New("SVG favicon is not well-formed XML")
	}
	return nil
}

func checkSVGAttr(elem string, a xml.Attr) error {
	name := strings.ToLower(a.Name.Local)
	val := strings.TrimSpace(a.Value)
	switch {
	case strings.HasPrefix(name, "on"):
		return errs.New("SVG favicon may not use event handler attributes (" + a.Name.Local + ")")
	case name == "href" && !strings.HasPrefix(val, "#"):
		return errs.New("SVG favicon may only link to #fragments in the same file (" + a.Name.Local + "=" + val + ")")
	case name == "attributename" && (strings.HasPrefix(strings.ToLower(val), "on") || strings.HasSuffix(strings.ToLower(val), "href")):
		return errs.New("SVG favicon <" + elem + "> may not animate " + val)
	}
	return checkSVGValue(a.Name.Local, val)
}

// checkSVGValue rejects javascript: and url() references other than url(#...).
func checkSVGValue(where, val string) error {
	compact := strings.ToLower(strings.Join(strings.Fields(val), ""))
	if strings.Contains(compact, "javascript:") {
		return errs.New("SVG favicon may not use javascript: (" + where + ")")
	}
	for rest := compact; ; {
		i := strings.Index(rest, "url(")
		if i < 0 {
			return nil
		}
		rest = strings.TrimLeft(rest[i+4:], `'"`)
		if !strings.HasPrefix(rest, "#") {
			return errs.New("SVG favicon may only reference url(#...) in the same file (" + where + ")")
		}
	}
}

// ResolveFavicon picks the favicon in effect: the env color/badge unless the
// override is on, else the stored mode. defaultIcon is today's icon. An upload
// mode without a valid upload falls back to the default icon.
func ResolveFavicon(s FaviconSettings, defaultIcon []byte) Favicon {
	extColor, extBadge := FaviconExternal()
	f := Favicon{Mode: s.Mode, Source: SourceSettings, Color: s.Color, Badge: s.Badge}
	switch {
	case (extColor != "" || extBadge != "") && !s.Override:
		f = Favicon{Mode: FaviconColor, Source: SourceEnv, Color: extColor, Badge: extBadge}
	case extColor != "" || extBadge != "":
		f.Source = SourceOverride
	case f.Mode == "" || f.Mode == FaviconDefault:
		f.Source = SourceDefault
	}
	if f.Mode == FaviconUpload {
		if _, ct, body, err := ParseFaviconUpload(s.Upload); err == nil {
			f.ContentType, f.Body = ct, body
			return f
		}
		f.Mode = FaviconDefault
	}
	if f.Mode != FaviconColor {
		f.Mode, f.Color, f.Badge, f.ContentType, f.Body = FaviconDefault, "", "", FaviconSVG, defaultIcon
		if len(defaultIcon) == 0 {
			f.Body = RenderFaviconSVG(DefaultFaviconColor, "")
		}
		return f
	}
	if c, err := NormalizeFaviconColor(f.Color); err != nil || c == "" {
		f.Color = DefaultFaviconColor
	} else {
		f.Color = c
	}
	if b, err := NormalizeFaviconBadge(f.Badge); err == nil {
		f.Badge = b
	} else {
		f.Badge = ""
	}
	f.ContentType, f.Body = FaviconSVG, RenderFaviconSVG(f.Color, f.Badge)
	return f
}

// RenderFaviconSVG draws today's bolt filled with color (a valid #rrggbb), with a
// soft highlight, and an optional 1–2 character badge in the bottom-right corner.
func RenderFaviconSVG(color, badge string) []byte {
	var sb strings.Builder
	sb.WriteString(`<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">`)
	sb.WriteString(`<defs><linearGradient id="h" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".45"/><stop offset=".6" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>`)
	sb.WriteString(`<g transform="translate(0 1)"><path fill="` + color + `" d="` + boltPath + `"/><path fill="url(#h)" d="` + boltPath + `"/></g>`)
	if badge != "" {
		size := "24"
		if utf8.RuneCountInString(badge) > 1 {
			size = "18"
		}
		var esc bytes.Buffer
		_ = xml.EscapeText(&esc, []byte(badge))
		sb.WriteString(`<rect x="17" y="19" width="30" height="28" rx="7" fill="` + color + `" stroke="#fff" stroke-width="2.5"/>`)
		sb.WriteString(`<text x="32" y="33" dominant-baseline="central" text-anchor="middle" font-family="Inter,Helvetica,Arial,sans-serif" font-weight="700" font-size="` + size + `" fill="` + contrastText(color) + `">` + esc.String() + `</text>`)
	}
	sb.WriteString(`</svg>`)
	return []byte(sb.String())
}

// contrastText returns black or white, whichever reads better on a #rrggbb background.
func contrastText(color string) string {
	rgb, err := hex.DecodeString(strings.TrimPrefix(color, "#"))
	if err != nil || len(rgb) != 3 {
		return "#fff"
	}
	if 0.299*float64(rgb[0])+0.587*float64(rgb[1])+0.114*float64(rgb[2]) > 160 {
		return "#111"
	}
	return "#fff"
}
