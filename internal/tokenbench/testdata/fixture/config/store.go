package config

import "os"

// Store fetches raw configuration blobs by key.
type Store interface {
	Get(key string) ([]byte, error)
}

// FileStore reads configuration blobs from a directory.
type FileStore struct {
	Dir string
}

// Get reads the file named key under the store directory.
func (f *FileStore) Get(key string) ([]byte, error) {
	return os.ReadFile(f.Dir + "/" + key)
}

// MemoryStore keeps configuration blobs in a map, for tests and defaults.
type MemoryStore struct {
	Blobs map[string][]byte
}

// Get returns the blob stored under key, or nil.
func (m *MemoryStore) Get(key string) ([]byte, error) {
	return m.Blobs[key], nil
}
