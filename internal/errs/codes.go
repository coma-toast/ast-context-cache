package errs

// Generic codes shared across packages. Feature packages define their own, more specific codes
// (e.g. handoff_depth_exceeded) as Code constants alongside their errors.
const (
	CodeInvalidInput  Code = "invalid_input"
	CodeNotFound      Code = "not_found"
	CodeExpired       Code = "expired"
	CodeLimitExceeded Code = "limit_exceeded"
	CodeConflict      Code = "conflict"
	CodeDisabled      Code = "disabled"
	CodeUnsupported   Code = "unsupported"
	CodeInternal      Code = "internal"
)
