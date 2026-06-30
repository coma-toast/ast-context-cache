package embedder

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

type openAIErrorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
}

var jsonErrInMsg = regexp.MustCompile(`(?s)\{.*"error"\s*:\s*\{.*\}`)
var apiMessageFieldRe = regexp.MustCompile(`['"]message['"]\s*:\s*['"]([^'"]+)['"]`)
var errorCodePrefixRe = regexp.MustCompile(`(?i)error code:\s*\d+\s*-\s*`)
var httpCodeInMsgRe = regexp.MustCompile(`(?:^|:|\s)(\d{3})(?::|\s|$)`)

// FormatHTTPError builds a concise embedding API failure from an HTTP response body.
func FormatHTTPError(label string, statusCode int, status string, body []byte) error {
	detail := extractAPIErrorMessage(body)
	label = strings.TrimSpace(label)
	if label == "" {
		label = "embed"
	}
	if detail == "" {
		return fmt.Errorf("%s: %s", label, strings.TrimSpace(status))
	}
	code := statusCode
	if code <= 0 {
		code = parseStatusCode(status)
	}
	if code > 0 {
		return fmt.Errorf("%s: %d: %s", label, code, detail)
	}
	return fmt.Errorf("%s: %s: %s", label, strings.TrimSpace(status), detail)
}

func parseStatusCode(status string) int {
	fields := strings.Fields(strings.TrimSpace(status))
	if len(fields) == 0 {
		return 0
	}
	n, _ := strconv.Atoi(fields[0])
	return n
}

func extractAPIErrorMessage(body []byte) string {
	if msg := parseOpenAIErrorJSON(body); msg != "" {
		return cleanAPIErrorMessage(msg)
	}
	s := strings.TrimSpace(string(body))
	if s == "" {
		return ""
	}
	if msg := parseOpenAIErrorJSON([]byte(s)); msg != "" {
		return cleanAPIErrorMessage(msg)
	}
	if len(s) > 240 {
		return s[:240] + "…"
	}
	return s
}

func parseOpenAIErrorJSON(body []byte) string {
	body = bytesTrimSpace(body)
	if len(body) == 0 {
		return ""
	}
	var env openAIErrorEnvelope
	if json.Unmarshal(body, &env) == nil && strings.TrimSpace(env.Error.Message) != "" {
		return strings.TrimSpace(env.Error.Message)
	}
	return ""
}

func bytesTrimSpace(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}

func cleanAPIErrorMessage(msg string) string {
	msg = strings.TrimSpace(msg)
	for {
		changed := false
		for _, prefix := range []string{
			"litellm.ServiceUnavailableError: ServiceUnavailableError: OpenAIException - ",
			"litellm.RateLimitError: RateLimitError: OpenAIException - ",
			"litellm.AuthenticationError: AuthenticationError: OpenAIException - ",
			"litellm.NotFoundError: NotFoundError: OpenAIException - ",
			"litellm.BadRequestError: BadRequestError: OpenAIException - ",
			"litellm.ServiceUnavailableError: ",
			"litellm.RateLimitError: ",
			"litellm.AuthenticationError: ",
			"ServiceUnavailableError: ",
			"RateLimitError: ",
			"AuthenticationError: ",
			"OpenAIException - ",
		} {
			if strings.HasPrefix(msg, prefix) {
				msg = strings.TrimPrefix(msg, prefix)
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	msg = strings.TrimSpace(msg)
	if len(msg) > 240 {
		return msg[:240] + "…"
	}
	return msg
}

// HumanizeEmbedError shortens stored embed errors for dashboard chips and banners.
func HumanizeEmbedError(msg string) string {
	msg = strings.TrimSpace(msg)
	if msg == "" {
		return ""
	}
	if strings.Contains(strings.ToLower(msg), "loading model") {
		return "Embed model loading on backend — embeddings retry automatically"
	}
	code := findHTTPCode(msg)
	if inner := extractAPIErrorField(msg, "message"); inner != "" {
		msg = inner
	} else if extracted := extractEmbeddedJSONError(msg); extracted != "" {
		msg = extracted
	}
	msg = errorCodePrefixRe.ReplaceAllString(msg, "")
	msg = strings.TrimSpace(msg)
	if strings.Contains(msg, "'error'") || strings.Contains(msg, `"error"`) {
		if inner := extractAPIErrorField(msg, "message"); inner != "" {
			msg = inner
		}
	}
	msg = strings.TrimSpace(msg)
	if i := strings.Index(msg, ": "); i >= 0 {
		rest := msg[i+2:]
		if c, detail, ok := parseLabelStatusDetail(rest); ok {
			if hint := statusHint(c); hint != "" {
				return hint + ": " + detail
			}
			return fmt.Sprintf("%d: %s", c, detail)
		}
	}
	if hint := statusHint(code); hint != "" && msg != "" {
		if strings.HasPrefix(strings.ToLower(msg), strings.ToLower(hint)) {
			return truncateDisplay(msg, 120)
		}
		return hint + ": " + truncateDisplay(msg, 96)
	}
	return truncateDisplay(msg, 120)
}

func extractAPIErrorField(s, field string) string {
	re := regexp.MustCompile(`['"]` + regexp.QuoteMeta(field) + `['"]\s*:\s*['"]([^'"]+)['"]`)
	if m := re.FindStringSubmatch(s); len(m) > 1 {
		return cleanAPIErrorMessage(m[1])
	}
	return ""
}

func findHTTPCode(msg string) int {
	if m := httpCodeInMsgRe.FindStringSubmatch(msg); len(m) > 1 {
		if n, err := strconv.Atoi(m[1]); err == nil && n >= 400 && n < 600 {
			return n
		}
	}
	return 0
}

func truncateDisplay(msg string, n int) string {
	msg = strings.TrimSpace(msg)
	if len(msg) <= n {
		return msg
	}
	return msg[:n-1] + "…"
}

func extractEmbeddedJSONError(msg string) string {
	loc := jsonErrInMsg.FindStringIndex(msg)
	if loc == nil {
		return ""
	}
	raw := msg[loc[0]:loc[1]]
	if parsed := parseOpenAIErrorJSON([]byte(raw)); parsed != "" {
		return cleanAPIErrorMessage(parsed)
	}
	return ""
}

func parseLabelStatusDetail(rest string) (code int, detail string, ok bool) {
	parts := strings.SplitN(rest, ": ", 2)
	if len(parts) != 2 {
		return 0, "", false
	}
	n, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil || n < 400 {
		return 0, "", false
	}
	return n, strings.TrimSpace(parts[1]), true
}

func statusHint(code int) string {
	switch code {
	case 401, 403:
		return "Invalid API key"
	case 404:
		return "Model not found"
	case 408, 504:
		return "Request timed out"
	case 429:
		return "Rate limited"
	case 500:
		return "Provider error"
	case 502:
		return "Bad gateway"
	case 503:
		return "Service unavailable"
	default:
		if code >= 500 {
			return "Provider error"
		}
		return ""
	}
}
