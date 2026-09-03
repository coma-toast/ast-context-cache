package sys

// DiskIO holds sampled host disk throughput (primary internal block device).
type DiskIO struct {
	ReadMBps  float64
	WriteMBps float64
}

// SSDHealth summarizes the health and identity of the disk holding a given path
// (typically the data directory — the internal boot SSD by default, or a moved
// external/USB drive), best-effort per OS and per what the drive/enclosure exposes.
type SSDHealth struct {
	Available     bool
	Device        string
	Model         string
	SmartStatus   string
	SmartSource   string // "diskutil", "smartctl", or "" if no SMART data was available
	Protocol      string
	Capacity      string
	FreeSpace     string
	IsExternal    bool
	SolidState    bool
	TrimSupport   bool
	WearUsedPct   int     // NVMe PERCENTAGE_USED; -1 if unknown
	SparePct      int     // NVMe AVAILABLE_SPARE; -1 if unknown
	DataWrittenTB float64 // lifetime host writes in TB; -1 if unknown
	TemperatureC  float64 // Celsius; -1 if unknown
}
