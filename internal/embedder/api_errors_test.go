package embedder

import (
	"strings"
	"testing"
)

func TestFormatHTTPError_OpenAIJSON(t *testing.T) {
	body := []byte(`{"error":{"message":"litellm.ServiceUnavailableError: ServiceUnavailableError: OpenAIException - engine overloaded","type":"service_unavailable","code":"503"}}`)
	err := FormatHTTPError("openai embed", 503, "503 Service Unavailable", body)
	got := err.Error()
	if !strings.Contains(got, "503") {
		t.Fatalf("want status code in error, got %q", got)
	}
	if strings.Contains(got, "litellm.") {
		t.Fatalf("want cleaned message, got %q", got)
	}
	if !strings.Contains(got, "engine overloaded") {
		t.Fatalf("want inner message, got %q", got)
	}
}

func TestHumanizeEmbedError_RawJSONBlob(t *testing.T) {
	raw := `openai embed 503 Service Unavailable: {"error":{"message":"litellm.RateLimitError: RateLimitError: OpenAIException - too many requests","type":"rate_limit","code":"429"}}`
	got := HumanizeEmbedError(raw)
	if strings.Contains(got, "{") {
		t.Fatalf("want no JSON in display error, got %q", got)
	}
	if !strings.Contains(strings.ToLower(got), "too many requests") {
		t.Fatalf("want rate-limit message, got %q", got)
	}
}

func TestHumanizeEmbedError_LiteLLMPythonDict(t *testing.T) {
	raw := "openai embed: 503: Error code: 503 - {'error': {'message': 'Loading model', 'type': 'unavailable_error', 'code': 503}}"
	got := HumanizeEmbedError(raw)
	if strings.Contains(got, "{") || strings.Contains(got, "'error'") {
		t.Fatalf("want cleaned message, got %q", got)
	}
	if !strings.Contains(got, "Loading model") {
		t.Fatalf("want inner message, got %q", got)
	}
	if !strings.Contains(got, "Service unavailable") {
		t.Fatalf("want status hint, got %q", got)
	}
}

func TestHumanizeEmbedError_FormattedCode(t *testing.T) {
	got := HumanizeEmbedError("openai embed: 401: Incorrect API key provided")
	if !strings.HasPrefix(got, "Invalid API key:") {
		t.Fatalf("got %q", got)
	}
}
