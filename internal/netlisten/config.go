package netlisten

import (
	"context"
	"maps"
	"net"
	"os"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
)

// Settings keys, each overridden and locked by its env var when that is non-empty.
const (
	KeyExtraAddrs   = "listen_extra_addrs"
	KeyTrustedHosts = "trusted_hosts"
	KeyAccessToken  = "remote_access_token"

	EnvExtraAddrs   = "AST_LISTEN_EXTRA"
	EnvTrustedHosts = "AST_TRUSTED_HOSTS"
	EnvAccessToken  = "AST_ACCESS_TOKEN"
)

// Keys lists the network access settings in display order.
var Keys = []string{KeyExtraAddrs, KeyTrustedHosts, KeyAccessToken}

var envByKey = map[string]string{
	KeyExtraAddrs:   EnvExtraAddrs,
	KeyTrustedHosts: EnvTrustedHosts,
	KeyAccessToken:  EnvAccessToken,
}

// maxHostnameLen and maxLabelLen are the DNS limits (RFC 1035) for trusted hostnames.
const (
	maxHostnameLen = 253
	maxLabelLen    = 63
)

// Config is the resolved network access configuration.
type Config struct {
	ExtraAddrs   []string
	TrustedHosts []string
	Token        string
	// Locked maps each key set by its env var to that var's name.
	Locked map[string]string
}

// State is the network access view for the dashboard API. It never carries the token.
type State struct {
	ExtraAddrs   []string          `json:"listen_extra_addrs"`
	TrustedHosts []string          `json:"trusted_hosts"`
	TokenSet     bool              `json:"token_set"`
	Locked       map[string]string `json:"locked"`
	Listeners    []Status          `json:"listeners"`
	BaseListen   string            `json:"base_listen"`
	BaseWildcard bool              `json:"base_wildcard"`
}

var (
	// regMu serializes Apply, so a settings save and the retry loop never reconcile at once.
	regMu    sync.Mutex
	managers []*Manager
	baseHost string
	baseAddr []string
	wildcard bool
)

// IsKey reports whether key is one of the network access settings.
func IsKey(key string) bool {
	_, ok := envByKey[key]
	return ok
}

// EnvFor returns the env var that locks key.
func EnvFor(key string) string {
	return envByKey[key]
}

// SetBase records the base listen address (-listen / AST_LISTEN) and the addresses its
// listeners already bind, so extras skip them. A wildcard base already covers every
// interface, so extras are skipped entirely.
func SetBase(listen string, bound ...string) {
	regMu.Lock()
	defer regMu.Unlock()
	baseHost, baseAddr = listen, normalizeIPs(append([]string{listen}, bound...))
	ip := net.ParseIP(listen)
	wildcard = listen == "" || (ip != nil && ip.IsUnspecified())
}

// Register adds managers for Apply to reconcile.
func Register(ms ...*Manager) {
	regMu.Lock()
	defer regMu.Unlock()
	managers = append(managers, ms...)
}

// Unregister removes managers added by Register.
func Unregister(ms ...*Manager) {
	regMu.Lock()
	defer regMu.Unlock()
	managers = slices.DeleteFunc(managers, func(m *Manager) bool { return slices.Contains(ms, m) })
}

// Apply loads the configuration, pushes the trusted hosts and token to httpguard, and
// reconciles every registered manager. Before db.Init it sees only the env vars, which
// is how an AST_ACCESS_TOKEN guards the servers from their first request.
func Apply() Config {
	cfg := Load()
	regMu.Lock()
	defer regMu.Unlock()
	trusted := slices.Concat(cfg.ExtraAddrs, cfg.TrustedHosts)
	if ip := net.ParseIP(baseHost); ip != nil && !wildcard && !ip.IsLoopback() {
		trusted = append(trusted, baseHost)
	}
	httpguard.SetTrusted(trusted)
	httpguard.SetAccessToken(cfg.Token)
	extras := effectiveLocked(cfg.ExtraAddrs)
	if wildcard && len(cfg.ExtraAddrs) > 0 && len(managers) > 0 {
		logger.Info("Skipping extra listen addresses: base listen address covers every interface", "listen", baseHost, "extra", cfg.ExtraAddrs)
	}
	for _, m := range managers {
		m.Reconcile(extras)
	}
	return cfg
}

// effectiveLocked drops addresses the base listeners already bind; regMu must be held.
func effectiveLocked(addrs []string) []string {
	if wildcard {
		return nil
	}
	out := []string{}
	for _, a := range addrs {
		if !slices.Contains(baseAddr, a) {
			out = append(out, a)
		}
	}
	return out
}

// Listeners reports every registered manager's extra listeners.
func Listeners() []Status {
	regMu.Lock()
	ms := slices.Clone(managers)
	regMu.Unlock()
	out := []Status{}
	for _, m := range ms {
		out = append(out, m.Status()...)
	}
	return out
}

// CurrentState is the dashboard's view of the configuration and live listeners.
func CurrentState() State {
	cfg := Load()
	regMu.Lock()
	b, wc := baseHost, wildcard
	regMu.Unlock()
	return State{
		ExtraAddrs:   cfg.ExtraAddrs,
		TrustedHosts: cfg.TrustedHosts,
		TokenSet:     cfg.Token != "",
		Locked:       cfg.Locked,
		Listeners:    Listeners(),
		BaseListen:   b,
		BaseWildcard: wc,
	}
}

// CloseAll closes every registered manager's listeners for shutdown.
func CloseAll() {
	regMu.Lock()
	defer regMu.Unlock()
	for _, m := range managers {
		m.Close()
	}
}

// StartRetry re-applies the configuration every interval while any extra listener is
// failing, so an address that appears later (tailscaled starting after ast-mcp at
// login) gets picked up without a settings change. It stops when ctx is cancelled.
func StartRetry(ctx context.Context, interval time.Duration) {
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if anyErrors() {
					Apply()
				}
			}
		}
	}()
}

func anyErrors() bool {
	regMu.Lock()
	ms := slices.Clone(managers)
	regMu.Unlock()
	return slices.ContainsFunc(ms, (*Manager).HasErrors)
}

// Load resolves each key env > setting. Bad entries, possible only from an env var or a
// hand-edited database since Save validates, are skipped with a warning.
func Load() Config {
	cfg := Config{Locked: map[string]string{}}
	raw := map[string]string{}
	for _, key := range Keys {
		env := envByKey[key]
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			raw[key], cfg.Locked[key] = v, env
			continue
		}
		raw[key] = db.GetSetting(key, "")
	}
	cfg.ExtraAddrs = parseLenient(raw[KeyExtraAddrs], parseAddr, KeyExtraAddrs)
	cfg.TrustedHosts = parseLenient(raw[KeyTrustedHosts], parseHost, KeyTrustedHosts)
	cfg.Token = strings.TrimSpace(raw[KeyAccessToken])
	return cfg
}

// Normalize validates value for key and returns the form to store: a comma-separated
// list of canonical entries, or the trimmed token. It fails with CodeInvalidInput.
func Normalize(key, value string) (string, error) {
	switch key {
	case KeyExtraAddrs:
		list, err := ParseAddrs(value)
		return strings.Join(list, ","), err
	case KeyTrustedHosts:
		list, err := ParseHosts(value)
		return strings.Join(list, ","), err
	case KeyAccessToken:
		tok := strings.TrimSpace(value)
		if strings.ContainsFunc(tok, unicode.IsSpace) {
			return "", errs.NewCode(errs.CodeInvalidInput, "remote_access_token must not contain whitespace")
		}
		return tok, nil
	}
	return "", errs.NewCode(errs.CodeNotFound, "unknown network setting", "key", key)
}

// Save validates and stores values (key to raw value), then applies them live. It
// fails with CodeInvalidInput for a bad value and CodeConflict when an env var locks a
// key; nothing is stored unless every value is accepted.
func Save(values map[string]string) error {
	norm := make(map[string]string, len(values))
	for _, key := range Keys {
		v, ok := values[key]
		if !ok {
			continue
		}
		if env := envByKey[key]; strings.TrimSpace(os.Getenv(env)) != "" {
			return errs.NewCode(errs.CodeConflict, key+" is locked by "+env, "key", key, "env", env)
		}
		n, err := Normalize(key, v)
		if err != nil {
			return err
		}
		norm[key] = n
	}
	if len(norm) == 0 {
		return errs.NewCode(errs.CodeInvalidInput, "no network setting given")
	}
	if db.DB == nil {
		return errs.NewCode(errs.CodeInternal, "settings database not open")
	}
	for _, key := range Keys {
		if v, ok := norm[key]; ok {
			if err := db.SetSetting(key, v); err != nil {
				return errs.WrapMessage("failed to save network setting", err, "key", key)
			}
		}
	}
	// Never log the token value, only whether one is set.
	logger.Info("Network access settings changed", "keys", slices.Sorted(maps.Keys(norm)), "token_set", norm[KeyAccessToken] != "")
	Apply()
	return nil
}

// ParseAddrs splits a comma/whitespace separated list of IP addresses, rejecting
// hostnames, wildcards and loopback (always listened on). Entries come back canonical
// and de-duplicated.
func ParseAddrs(raw string) ([]string, error) {
	return parseList(raw, parseAddr)
}

// ParseHosts splits a comma/whitespace separated list of hostnames or IPs, lowercased
// with any trailing dot dropped.
func ParseHosts(raw string) ([]string, error) {
	return parseList(raw, parseHost)
}

func parseList(raw string, parse func(string) (string, error)) ([]string, error) {
	out := []string{}
	for _, e := range splitList(raw) {
		v, err := parse(e)
		if err != nil {
			return nil, err
		}
		if !slices.Contains(out, v) {
			out = append(out, v)
		}
	}
	return out, nil
}

func parseLenient(raw string, parse func(string) (string, error), key string) []string {
	out := []string{}
	for _, e := range splitList(raw) {
		v, err := parse(e)
		if err != nil {
			logger.Warn("Ignoring invalid network setting entry", "key", key, "entry", e, "error", err)
			continue
		}
		if !slices.Contains(out, v) {
			out = append(out, v)
		}
	}
	return out
}

func splitList(raw string) []string {
	return strings.FieldsFunc(raw, func(r rune) bool { return r == ',' || unicode.IsSpace(r) })
}

func parseAddr(e string) (string, error) {
	ip := net.ParseIP(strings.Trim(e, "[]"))
	switch {
	case ip == nil:
		return "", errs.NewCode(errs.CodeInvalidInput, `listen_extra_addrs: "`+e+`" is not an IP address (put hostnames in trusted_hosts)`, "entry", e)
	case ip.IsUnspecified():
		return "", errs.NewCode(errs.CodeInvalidInput, `listen_extra_addrs: "`+e+`" is a wildcard; use --listen / AST_LISTEN to bind every interface`, "entry", e)
	case ip.IsLoopback():
		return "", errs.NewCode(errs.CodeInvalidInput, `listen_extra_addrs: "`+e+`" is loopback, which is always listened on`, "entry", e)
	}
	return ip.String(), nil
}

func parseHost(e string) (string, error) {
	h := strings.TrimSuffix(strings.ToLower(e), ".")
	if ip := net.ParseIP(strings.Trim(h, "[]")); ip != nil {
		if ip.IsUnspecified() {
			return "", errs.NewCode(errs.CodeInvalidInput, `trusted_hosts: "`+e+`" is a wildcard`, "entry", e)
		}
		return ip.String(), nil
	}
	if !validHostname(h) {
		return "", errs.NewCode(errs.CodeInvalidInput, `trusted_hosts: "`+e+`" is not a valid hostname`, "entry", e)
	}
	return h, nil
}

// validHostname checks RFC 1123 syntax: dot-separated labels of letters, digits and
// inner hyphens. h is already lowercased.
func validHostname(h string) bool {
	if h == "" || len(h) > maxHostnameLen {
		return false
	}
	for _, label := range strings.Split(h, ".") {
		if label == "" || len(label) > maxLabelLen || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, r := range label {
			if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' {
				return false
			}
		}
	}
	return true
}

func normalizeIPs(addrs []string) []string {
	out := []string{}
	for _, a := range addrs {
		if ip := net.ParseIP(strings.Trim(a, "[]")); ip != nil {
			out = append(out, ip.String())
		}
	}
	return out
}
