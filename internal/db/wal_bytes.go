package db

// IndexWalBytes returns on-disk WAL size for index.db.
func IndexWalBytes() int64 {
	return statWalBytes(indexDBPath())
}

// UsageWalBytes returns on-disk WAL size for usage.db.
func UsageWalBytes() int64 {
	return statWalBytes(usageDBPath())
}

// ContextWalBytes returns on-disk WAL size for context.db.
func ContextWalBytes() int64 {
	return statWalBytes(contextDBPath())
}
