// Package errs provides errors that carry an optional code, an optional wrapped
// cause, and key-value context fields.
//
// It is a clean-room implementation of the error API described in STYLEGUIDE.md
// (New, NewCode, Wrap, WrapCode, WrapMessage, WrapCodeMessage, HasCode). Wrap*
// functions return the error interface and return nil for a nil cause, so a
// typed-nil *Error can never leak into an error return.
package errs

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
)

// badKey is used for a trailing key-value argument that has no partner, matching slog.
const badKey = "!BADKEY"

// Error is an error with a message, optional codes, an optional wrapped cause, and key-value fields.
// It is immutable once constructed.
type Error struct {
	message string
	codes   Codes
	err     error
	fields  map[string]any
}

// Code is a stable, machine-readable error code.
type Code string

// Codes is a list of error codes, outermost first.
type Codes []Code

// String returns the code as a string.
func (c Code) String() string {
	return string(c)
}

// Has reports whether code is in the list.
func (c Codes) Has(code Code) bool {
	return slices.Contains(c, code)
}

// Strings returns the codes as strings.
func (c Codes) Strings() []string {
	out := make([]string, len(c))
	for i, code := range c {
		out[i] = string(code)
	}
	return out
}

// New returns an error with the given message and key-value fields.
func New(msg string, kv ...any) *Error {
	return &Error{message: msg, fields: toFields(kv)}
}

// NewCode returns an error with a code, message, and key-value fields.
func NewCode(code Code, msg string, kv ...any) *Error {
	return &Error{message: msg, codes: Codes{code}, fields: toFields(kv)}
}

// Wrap adds key-value fields to err. It returns nil when err is nil.
func Wrap(err error, kv ...any) error {
	if err == nil {
		return nil
	}
	return &Error{err: err, fields: toFields(kv)}
}

// WrapCode adds a code and key-value fields to err. It returns nil when err is nil.
func WrapCode(code Code, err error, kv ...any) error {
	if err == nil {
		return nil
	}
	return &Error{codes: Codes{code}, err: err, fields: toFields(kv)}
}

// WrapMessage wraps err with a message and key-value fields. It returns nil when err is nil.
func WrapMessage(msg string, err error, kv ...any) error {
	if err == nil {
		return nil
	}
	return &Error{message: msg, err: err, fields: toFields(kv)}
}

// WrapCodeMessage wraps err with a code, message, and key-value fields. It returns nil when err is nil.
func WrapCodeMessage(code Code, msg string, err error, kv ...any) error {
	if err == nil {
		return nil
	}
	return &Error{message: msg, codes: Codes{code}, err: err, fields: toFields(kv)}
}

// HasCode reports whether err, or any error it wraps, carries code.
func HasCode(err error, code Code) bool {
	return CodesOf(err).Has(code)
}

// CodesOf returns every code carried by err and the errors it wraps, outermost first.
func CodesOf(err error) Codes {
	var e *Error
	if !errors.As(err, &e) {
		return nil
	}
	return e.Codes()
}

// CodeOf returns the outermost code carried by err, or "" when there is none.
func CodeOf(err error) Code {
	codes := CodesOf(err)
	if len(codes) == 0 {
		return ""
	}
	return codes[0]
}

// FieldsOf returns the merged key-value fields of err and the errors it wraps.
func FieldsOf(err error) map[string]any {
	var e *Error
	if !errors.As(err, &e) {
		return nil
	}
	return e.Fields()
}

// Error returns "message: cause", or whichever of the two is present.
func (e *Error) Error() string {
	switch {
	case e.err == nil:
		return e.message
	case e.message == "":
		return e.err.Error()
	default:
		return e.message + ": " + e.err.Error()
	}
}

// Unwrap returns the wrapped cause.
func (e *Error) Unwrap() error {
	return e.err
}

// Message returns the error's own message, without the wrapped cause.
func (e *Error) Message() string {
	return e.message
}

// Codes returns this error's codes followed by the codes of the errors it wraps.
func (e *Error) Codes() Codes {
	out := slices.Clone(e.codes)
	for _, c := range CodesOf(e.err) {
		if !out.Has(c) {
			out = append(out, c)
		}
	}
	return out
}

// Fields returns a copy of the merged fields. Outer errors override inner ones for the same key.
func (e *Error) Fields() map[string]any {
	out := map[string]any{}
	maps.Copy(out, FieldsOf(e.err))
	maps.Copy(out, e.fields)
	return out
}

// LogValue renders the error as a slog group of its message, codes, and fields.
func (e *Error) LogValue() slog.Value {
	attrs := []slog.Attr{slog.String("msg", e.Error())}
	if codes := e.Codes(); len(codes) > 0 {
		attrs = append(attrs, slog.String("codes", strings.Join(codes.Strings(), ",")))
	}
	fields := e.Fields()
	for _, k := range slices.Sorted(maps.Keys(fields)) {
		attrs = append(attrs, slog.Any(k, fields[k]))
	}
	return slog.GroupValue(attrs...)
}

// MarshalJSON encodes the error as {"message", "codes", "fields"}.
func (e *Error) MarshalJSON() ([]byte, error) {
	out := map[string]any{"message": e.Error()}
	if codes := e.Codes(); len(codes) > 0 {
		out["codes"] = codes.Strings()
	}
	if fields := e.Fields(); len(fields) > 0 {
		out["fields"] = fields
	}
	return json.Marshal(out)
}

// toFields turns alternating key-value arguments into a map. A non-string key is formatted
// with fmt.Sprint, and an unpaired trailing value is stored under "!BADKEY" as slog does.
func toFields(kv []any) map[string]any {
	if len(kv) == 0 {
		return nil
	}
	out := make(map[string]any, len(kv)/2+1)
	for i := 0; i < len(kv); i += 2 {
		if i+1 == len(kv) {
			out[badKey] = kv[i]
			break
		}
		key, ok := kv[i].(string)
		if !ok {
			key = fmt.Sprint(kv[i])
		}
		out[key] = kv[i+1]
	}
	return out
}

var (
	_ error          = (*Error)(nil)
	_ slog.LogValuer = (*Error)(nil)
	_ json.Marshaler = (*Error)(nil)
)
