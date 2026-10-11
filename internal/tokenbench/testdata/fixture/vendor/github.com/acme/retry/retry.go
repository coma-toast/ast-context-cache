// Package retry is a vendored copy of github.com/acme/retry.
package retry

import "time"

// Do calls fn until it succeeds or attempts run out, sleeping with Backoff between tries.
func Do(attempts int, fn func() error) error {
	var err error
	for i := 0; i < attempts; i++ {
		if err = fn(); err == nil {
			return nil
		}
		time.Sleep(Backoff(i))
	}
	return err
}

// Backoff is the exponential delay before retry attempt n.
func Backoff(n int) time.Duration {
	return time.Duration(1<<n) * 100 * time.Millisecond
}
