package memory

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRedactTextSecrets(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		input    string
		secret   string
		wantType string
		wantKeep []string
	}{
		{
			name:     "openai_project_key",
			input:    "set key to sk-proj-AbCdEf1234567890GhIjKlMnOpQr",
			secret:   "sk-proj-AbCdEf1234567890GhIjKlMnOpQr",
			wantType: "api_key",
			wantKeep: []string{"set key to [REDACTED:api_key]"},
		},
		{
			name:     "anthropic_key",
			input:    "ANTHROPIC=sk-ant-api03-AbCdEf1234567890GhIjKlMnOpQr",
			secret:   "sk-ant-api03-AbCdEf1234567890GhIjKlMnOpQr",
			wantType: "api_key",
		},
		{
			name:     "stripe_live_key",
			input:    "stripe=sk-live-AbCdEf1234567890GhIjKlMnOpQr",
			secret:   "sk-live-AbCdEf1234567890GhIjKlMnOpQr",
			wantType: "api_key",
		},
		{
			name:     "github_pat_classic",
			input:    "token is ghp_1234567890abcdefghijABCDEFGHIJ",
			secret:   "ghp_1234567890abcdefghijABCDEFGHIJ",
			wantType: "github_token",
		},
		{
			name:     "github_pat_fine_grained",
			input:    "token is github_pat_11AAAAAA0123456789_abcdefghij1234567890",
			secret:   "github_pat_11AAAAAA0123456789_abcdefghij1234567890",
			wantType: "github_token",
		},
		{
			name:     "aws_access_key",
			input:    "aws key AKIAIOSFODNN7EXAMPLE in env",
			secret:   "AKIAIOSFODNN7EXAMPLE",
			wantType: "aws_access_key",
		},
		{
			name:     "slack_bot_token",
			input:    "slack " + "xoxb-" + "1234567890-" + "123456789012-" + "abcdefghijklmnop",
			secret:   "xoxb-" + "1234567890-" + "123456789012-" + "abcdefghijklmnop",
			wantType: "slack_token",
		},
		{
			name:     "bearer_token",
			input:    "Authorization: Bearer opaqueToken123456789",
			secret:   "opaqueToken123456789",
			wantType: "bearer_token",
			wantKeep: []string{"Authorization: Bearer [REDACTED:bearer_token]"},
		},
		{
			name:     "jwt_token",
			input:    "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
			secret:   "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
			wantType: "jwt",
		},
		{
			name:     "pem_private_key_block",
			input:    "key:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234567890\n-----END RSA PRIVATE KEY-----\ndone",
			secret:   "MIIEowIBAAKCAQEA1234567890",
			wantType: "private_key",
			wantKeep: []string{"key:\n[REDACTED:private_key]\ndone"},
		},
		{
			name:     "pem_single_line",
			input:    "saw -----BEGIN EC PRIVATE KEY----- in log",
			secret:   "-----BEGIN EC PRIVATE KEY-----",
			wantType: "private_key",
		},
		{
			name:     "url_embedded_credentials",
			input:    "dsn is postgres://dbadmin:SuperSecret99@db.internal:5432/app?sslmode=require",
			secret:   "SuperSecret99",
			wantType: "url_credentials",
			wantKeep: []string{"postgres://[REDACTED:url_credentials]@db.internal:5432/app?sslmode=require"},
		},
		{
			name:     "key_value_password",
			input:    "password=mySecretPass123",
			secret:   "mySecretPass123",
			wantType: "password",
			wantKeep: []string{"password=[REDACTED:password]"},
		},
		{
			name:     "key_colon_api_key_quoted",
			input:    `api_key: "custom-secret-value-99"`,
			secret:   "custom-secret-value-99",
			wantType: "api_key",
			wantKeep: []string{`api_key: "[REDACTED:api_key]"`},
		},
		{
			name:     "json_client_secret",
			input:    `{"client_secret": "s3cr3t-cl13nt"}`,
			secret:   "s3cr3t-cl13nt",
			wantType: "client_secret",
			wantKeep: []string{`"client_secret": "[REDACTED:client_secret]"`},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			redacted, types := RedactText(tc.input)
			assert.NotContains(t, redacted, tc.secret)
			assert.Contains(t, redacted, "[REDACTED:"+tc.wantType+"]")
			assert.Contains(t, types, tc.wantType)
			for _, keep := range tc.wantKeep {
				assert.Contains(t, redacted, keep)
			}

			// Idempotency: a second pass must not re-redact or emit new matches.
			redacted2, types2 := RedactText(redacted)
			assert.Equal(t, redacted, redacted2)
			assert.Empty(t, types2)
		})
	}
}

func TestRedactTextZeroFalsePositives(t *testing.T) {
	t.Parallel()

	safeInputs := []struct {
		name  string
		input string
	}{
		{"git_sha_40", "commit e3b0c44298fc1c149afbf4c8996fb92427ae41e4 merged"},
		{"uuid", "request_id=550e8400-e29b-41d4-a716-446655440000"},
		{"memory_refs", "refs mem_0123456789ab ctx_0123456789ab hof_0123456789ab"},
		{"file_paths", "/Users/jason/orca/workspaces/ast-context-cache/sauger/internal/memory/redact.go"},
		{"hyphenated_identifiers", "task-something-with-twenty-char-suffix disk-partition-identifier-123456"},
		{"normal_config", "max_connections=200 port: 8080 timeout=30s log_level=debug password_policy=strict"},
		{"url_without_creds", "https://github.com/org/repo/blob/main/file.go:42 and http://localhost:8080/foo@bar"},
		{"bearer_prose", "the bearer of bad news and Bearer token wording"},
		{"empty", ""},
	}

	for _, tc := range safeInputs {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			redacted, types := RedactText(tc.input)
			assert.Equal(t, tc.input, redacted)
			assert.Empty(t, types)
		})
	}
}

func TestRedactFieldsAndStructs(t *testing.T) {
	t.Parallel()

	after := "api_key=sk-proj-AbCdEf1234567890GhIjKlMnOpQr and AKIAIOSFODNN7EXAMPLE"
	before := "max_connections=100"
	body := "dsn postgres://user:pass123@db.local/app"
	var nilField *string

	fields, warnings := RedactFields(map[string]*string{
		"after_value":  &after,
		"before_value": &before,
		"body":         &body,
		"empty":        nilField,
	})

	assert.Equal(t, []string{"after_value", "body"}, fields)
	assert.Contains(t, warnings, "redacted api_key in after_value")
	assert.Contains(t, warnings, "redacted aws_access_key in after_value")
	assert.Contains(t, warnings, "redacted url_credentials in body")
	assert.Equal(t, "max_connections=100", before)
	assert.False(t, strings.Contains(after, "AbCdEf1234567890"))
	assert.False(t, strings.Contains(body, "pass123"))

	entry := Entry{
		Kind:       KindConfigChange,
		Target:     "env",
		AfterValue: "password=superSecret99",
	}
	efields, ewarn := RedactEntry(&entry)
	require.Equal(t, []string{"after_value"}, efields)
	require.Equal(t, []string{"redacted password in after_value"}, ewarn)
	assert.Equal(t, []string{"after_value"}, entry.RedactedFields)
	assert.Equal(t, "password=[REDACTED:password]", entry.AfterValue)

	storeIn := StoreInput{
		Kind:    KindFact,
		Subject: "auth",
		Object:  "ghp_1234567890abcdefghijABCDEFGHIJ",
	}
	sfields, swarn := RedactStoreInput(&storeIn)
	require.Equal(t, []string{"object"}, sfields)
	require.Equal(t, []string{"redacted github_token in object"}, swarn)
	assert.Equal(t, "[REDACTED:github_token]", storeIn.Object)
}
