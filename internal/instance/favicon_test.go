package instance

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/png"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testPNG(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 16, 16))))
	return buf.Bytes()
}

// testICO is a 1-image ICO header with a 16x16 entry; the payload is not checked.
var testICO = append([]byte{0, 0, 1, 0, 1, 0, 16, 16, 0, 0, 1, 0, 32, 0, 4, 0, 0, 0, 22, 0, 0, 0}, 1, 2, 3, 4)

const safeSVG = `<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 16 16"><defs><linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient></defs><style>.a{fill:url(#g)}</style><rect class="a" width="16" height="16" fill="url(#g)"/><use xlink:href="#g"/></svg>`

func dataURL(ct string, b []byte) string {
	return "data:" + ct + ";base64," + base64.StdEncoding.EncodeToString(b)
}

func TestNormalizeFaviconColor(t *testing.T) {
	cases := []struct{ in, want, err string }{
		{in: "", want: ""},
		{in: "#3366FF", want: "#3366ff"},
		{in: "3366ff", want: "#3366ff"},
		{in: "#abc", want: "#aabbcc"},
		{in: "#12345", err: "hex color"},
		{in: "#ggg", err: "hex color"},
		{in: "red", err: "hex color"},
		{in: `#1"/><script>`, err: "hex color"},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got, err := NormalizeFaviconColor(tc.in)
			if tc.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestNormalizeFaviconBadge(t *testing.T) {
	cases := []struct{ in, want, err string }{
		{in: "", want: ""},
		{in: " J ", want: "J"},
		{in: "M2", want: "M2"},
		{in: "日本", want: "日本"},
		{in: "abc", err: "1–2 characters"},
		{in: "<", err: "letters and digits"},
		{in: "a&", err: "letters and digits"},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got, err := NormalizeFaviconBadge(tc.in)
			if tc.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestParseFaviconUpload(t *testing.T) {
	pngBytes := testPNG(t)
	cases := []struct {
		name, in, wantType, err string
	}{
		{name: "png", in: dataURL("image/png", pngBytes), wantType: FaviconPNG},
		{name: "ico", in: dataURL("image/x-icon", testICO), wantType: FaviconICO},
		{name: "ico vnd type", in: dataURL("image/vnd.microsoft.icon", testICO), wantType: FaviconICO},
		{name: "svg", in: dataURL("image/svg+xml", []byte(safeSVG)), wantType: FaviconSVG},
		{name: "untyped png", in: dataURL("application/octet-stream", pngBytes), wantType: FaviconPNG},
		{name: "not a data url", in: "https://example.com/x.png", err: "base64 data URL"},
		{name: "bad base64", in: "data:image/png;base64,!!!", err: "not valid base64"},
		{name: "too large", in: dataURL("image/png", append(pngBytes, make([]byte, FaviconMaxBytes)...)), err: "the limit is 64 KB"},
		{name: "just over the limit", in: dataURL("image/png", append(pngBytes, make([]byte, FaviconMaxBytes-len(pngBytes)+1)...)), err: "the limit is 65536"},
		{name: "jpeg", in: dataURL("image/jpeg", []byte("\xff\xd8\xff\xe0 jpeg")), err: "must be a PNG, SVG or ICO"},
		{name: "html", in: dataURL("text/html", []byte("<html><body>hi</body></html>")), err: "must be a PNG, SVG or ICO"},
		{name: "declared type mismatch", in: dataURL("image/png", []byte(safeSVG)), err: "declares image/png but the file is image/svg+xml"},
		{name: "broken png", in: dataURL("image/png", []byte("\x89PNG\r\n\x1a\nnope")), err: "does not decode"},
		{name: "html wrapping svg", in: dataURL("image/svg+xml", []byte(`<html><svg/></html>`)), err: "root element is <html>"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stored, ct, body, err := ParseFaviconUpload(tc.in)
			if tc.err != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.wantType, ct)
			assert.Equal(t, dataURL(ct, body), stored, "stored with the sniffed type")
		})
	}
}

func TestCheckSafeSVG(t *testing.T) {
	const ns = `xmlns="http://www.w3.org/2000/svg"`
	cases := []struct{ name, svg, err string }{
		{name: "safe", svg: safeSVG},
		{name: "script", svg: `<svg ` + ns + `><script>alert(1)</script></svg>`, err: "<script>"},
		{name: "script any case", svg: `<svg ` + ns + `><SCRIPT>alert(1)</SCRIPT></svg>`, err: "<SCRIPT>"},
		{name: "xhtml element", svg: `<svg ` + ns + ` xmlns:h="http://www.w3.org/1999/xhtml"><h:div/></svg>`, err: "<div>"},
		{name: "foreignObject", svg: `<svg ` + ns + `><foreignObject><p>x</p></foreignObject></svg>`, err: "<foreignObject>"},
		{name: "onload", svg: `<svg ` + ns + ` onload="alert(1)"/>`, err: "event handler"},
		{name: "onclick on child", svg: `<svg ` + ns + `><rect onClick="alert(1)"/></svg>`, err: "event handler"},
		{name: "javascript href", svg: `<svg ` + ns + `><a href="javascript:alert(1)"><rect/></a></svg>`, err: "#fragments"},
		{name: "external use", svg: `<svg ` + ns + ` xmlns:xlink="http://www.w3.org/1999/xlink"><use xlink:href="https://evil.example/x.svg#a"/></svg>`, err: "#fragments"},
		{name: "external image", svg: `<svg ` + ns + `><image href="#a"/></svg>`, err: "<image>"},
		{name: "external url fill", svg: `<svg ` + ns + `><rect fill="url( 'https://evil.example/p.svg#a')"/></svg>`, err: "url(#...)"},
		{name: "style import", svg: `<svg ` + ns + `><style>@import "https://evil.example/a.css";</style></svg>`, err: "@import"},
		{name: "style external url", svg: `<svg ` + ns + `><style>rect{fill:url(https://evil.example/p)}</style></svg>`, err: "url(#...)"},
		{name: "animate href", svg: `<svg ` + ns + `><a href="#x"><set attributeName="href" to="javascript:alert(1)"/></a></svg>`, err: "may not animate"},
		{name: "javascript in values", svg: `<svg ` + ns + `><animate attributeName="x" values="java script:1;javascript:alert(1)"/></svg>`, err: "javascript:"},
		{name: "doctype entity", svg: `<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg ` + ns + `>&x;</svg>`, err: "DOCTYPE"},
		{name: "xml-stylesheet", svg: `<?xml-stylesheet href="https://evil.example/a.css"?><svg ` + ns + `/>`, err: "processing instructions"},
		{name: "not xml", svg: `<svg ` + ns + `><rect></svg>`, err: "well-formed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckSafeSVG([]byte(tc.svg))
			if tc.err == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.err)
		})
	}
}

func TestResolveFavicon(t *testing.T) {
	def := []byte("<svg>default</svg>")
	upload := dataURL("image/png", testPNG(t))
	cases := []struct {
		name, envColor, envBadge                             string
		s                                                    FaviconSettings
		wantMode, wantSource, wantColor, wantBadge, wantType string
	}{
		{name: "nothing set", wantMode: FaviconDefault, wantSource: SourceDefault, wantType: FaviconSVG},
		{name: "color", s: FaviconSettings{Mode: FaviconColor, Color: "#00aa00", Badge: "M"}, wantMode: FaviconColor, wantSource: SourceSettings, wantColor: "#00aa00", wantBadge: "M", wantType: FaviconSVG},
		{name: "color without color", s: FaviconSettings{Mode: FaviconColor}, wantMode: FaviconColor, wantSource: SourceSettings, wantColor: DefaultFaviconColor, wantType: FaviconSVG},
		{name: "upload", s: FaviconSettings{Mode: FaviconUpload, Upload: upload}, wantMode: FaviconUpload, wantSource: SourceSettings, wantType: FaviconPNG},
		{name: "upload mode without upload", s: FaviconSettings{Mode: FaviconUpload}, wantMode: FaviconDefault, wantSource: SourceSettings, wantType: FaviconSVG},
		{name: "env locks", envColor: "#f00", envBadge: "E", s: FaviconSettings{Mode: FaviconUpload, Upload: upload}, wantMode: FaviconColor, wantSource: SourceEnv, wantColor: "#ff0000", wantBadge: "E", wantType: FaviconSVG},
		{name: "env badge only", envBadge: "E", wantMode: FaviconColor, wantSource: SourceEnv, wantColor: DefaultFaviconColor, wantBadge: "E", wantType: FaviconSVG},
		{name: "invalid env ignored", envColor: "nope", wantMode: FaviconDefault, wantSource: SourceDefault, wantType: FaviconSVG},
		{name: "override", envColor: "#f00", s: FaviconSettings{Mode: FaviconUpload, Upload: upload, Override: true}, wantMode: FaviconUpload, wantSource: SourceOverride, wantType: FaviconPNG},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(FaviconColorEnv, tc.envColor)
			t.Setenv(FaviconBadgeEnv, tc.envBadge)
			f := ResolveFavicon(tc.s, def)
			assert.Equal(t, tc.wantMode, f.Mode)
			assert.Equal(t, tc.wantSource, f.Source)
			assert.Equal(t, tc.wantColor, f.Color)
			assert.Equal(t, tc.wantBadge, f.Badge)
			assert.Equal(t, tc.wantType, f.ContentType)
			if f.Mode == FaviconDefault {
				assert.Equal(t, def, f.Body)
			}
		})
	}
}

func TestFaviconVersionChanges(t *testing.T) {
	t.Setenv(FaviconColorEnv, "")
	t.Setenv(FaviconBadgeEnv, "")
	seen := map[string]string{}
	for name, s := range map[string]FaviconSettings{
		"default": {},
		"red":     {Mode: FaviconColor, Color: "#ff0000"},
		"blue":    {Mode: FaviconColor, Color: "#0000ff"},
		"blue J":  {Mode: FaviconColor, Color: "#0000ff", Badge: "J"},
		"png":     {Mode: FaviconUpload, Upload: dataURL("image/png", testPNG(t))},
	} {
		v := ResolveFavicon(s, []byte("<svg/>")).Version()
		assert.Len(t, v, 12)
		assert.NotContains(t, seen, v, "%s has the same version as %s", name, seen[v])
		seen[v] = name
	}
	assert.Equal(t, ResolveFavicon(FaviconSettings{}, []byte("<svg/>")).Version(), ResolveFavicon(FaviconSettings{Mode: FaviconDefault}, []byte("<svg/>")).Version())
}

func TestRenderFaviconSVG(t *testing.T) {
	plain := string(RenderFaviconSVG("#123456", ""))
	assert.Contains(t, plain, `fill="#123456"`)
	assert.NotContains(t, plain, "<text")
	badged := string(RenderFaviconSVG("#ffff00", "日"))
	assert.Contains(t, badged, `>日</text>`)
	assert.Contains(t, badged, `fill="#111"`, "dark text on a light badge")
	require.NoError(t, CheckSafeSVG([]byte(badged)), "rendered favicons pass the upload check")
	assert.True(t, strings.HasPrefix(badged, "<svg"))
}
