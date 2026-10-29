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

// DiskInfo is one disk on the dashboard's disk health card; the dashboard's DiskInfo type reads
// the same field names. Pointer fields are omitted when the source does not report them, so
// the card shows nothing rather than a zero. Host marks a disk of the machine this instance runs
// on as a guest (an LXC container or VM), read from that host's smartctl_exporter.
type DiskInfo struct {
	Device               string   `json:"Device"`
	Model                string   `json:"Model,omitempty"`
	Serial               string   `json:"Serial,omitempty"`
	Protocol             string   `json:"Protocol,omitempty"`
	SmartType            string   `json:"SmartType,omitempty"`
	SmartStatus          string   `json:"SmartStatus,omitempty"`
	SmartSource          string   `json:"SmartSource,omitempty"`
	Capacity             string   `json:"Capacity,omitempty"`
	SolidState           *bool    `json:"SolidState,omitempty"`
	Host                 bool     `json:"Host,omitempty"`
	WarningLevel         string   `json:"WarningLevel,omitempty"`
	WarningMsg           string   `json:"WarningMsg,omitempty"`
	WearUsedPct          *int     `json:"WearUsedPct,omitempty"`
	SparePct             *int     `json:"SparePct,omitempty"`
	DataWrittenTB        *float64 `json:"DataWrittenTB,omitempty"`
	TemperatureC         *float64 `json:"TemperatureC,omitempty"`
	PowerOnHours         *int     `json:"PowerOnHours,omitempty"`
	PowerCycles          *int     `json:"PowerCycles,omitempty"`
	MediaErrors          *int     `json:"MediaErrors,omitempty"`
	CriticalWarning      *int     `json:"CriticalWarning,omitempty"`
	ReallocatedSectors   *int     `json:"ReallocatedSectors,omitempty"`
	PendingSectors       *int     `json:"PendingSectors,omitempty"`
	OfflineUncorrectable *int     `json:"OfflineUncorrectable,omitempty"`
	UDMACRCErrors        *int     `json:"UDMACRCErrors,omitempty"`
	SASGrownDefects      *int     `json:"SASGrownDefects,omitempty"`
}
