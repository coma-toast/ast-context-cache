//go:build !darwin

package watcher

// No native recursive backend outside macOS: inotify costs a watch (not a
// descriptor) per directory, so fsnotify is already cheap there.
const nativeBackendName = ""

func newNativeBackend(root string) (backend, error) { return nil, nil }
