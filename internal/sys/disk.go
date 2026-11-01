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

// DiskInfo describes a single physical disk and its SMART health data, as
// reported by the platform source (diskutil, smartctl, or the Proxmox API).
// Pointer fields are omitted in JSON when the source does not report them.
// The Dashboard's disk card reads the same field names. Host marks a disk of
// the machine this instance runs on as a guest (an LXC container or VM),
// read from that host's smartctl_exporter.
type DiskInfo struct {
	Device               string   `json:"Device"`
	Model                string   `json:"Model,omitempty"`
	Serial               string   `json:"Serial,omitempty"`
	Protocol             string   `json:"Protocol,omitempty"`
	SmartType            string   `json:"SmartType,omitempty"`
	SmartStatus          string   `json:"SmartStatus,omitempty"`
	SmartSource          string   `json:"SmartSource,omitempty"`
	Capacity             string   `json:"Capacity,omitempty"`
	FreeSpace            string   `json:"FreeSpace,omitempty"`
	IsExternal           bool     `json:"IsExternal,omitempty"`
	SolidState           *bool    `json:"SolidState,omitempty"`
	TrimSupport          bool     `json:"TrimSupport,omitempty"`
	Host                 bool     `json:"Host,omitempty"`
	DataDirDisk          bool     `json:"DataDirDisk,omitempty"`
	WarningLevel         string   `json:"WarningLevel,omitempty"`
	WarningMsg           string   `json:"WarningMsg,omitempty"`
	WearUsedPct          *int     `json:"WearUsedPct,omitempty"`
	SparePct             *int     `json:"SparePct,omitempty"`
	DataWrittenTB        *float64 `json:"DataWrittenTB,omitempty"`
	TemperatureC         *float64 `json:"TemperatureC,omitempty"`
	PowerOnHours         *int     `json:"PowerOnHours,omitempty"`
	PowerCycles          *int     `json:"PowerCycles,omitempty"`
	UnsafeShutdowns      *int     `json:"UnsafeShutdowns,omitempty"`
	MediaErrors          *int     `json:"MediaErrors,omitempty"`
	CriticalWarning      *int     `json:"CriticalWarning,omitempty"`
	ReallocatedSectors   *int     `json:"ReallocatedSectors,omitempty"`
	PendingSectors       *int     `json:"PendingSectors,omitempty"`
	OfflineUncorrectable *int     `json:"OfflineUncorrectable,omitempty"`
	UDMACRCErrors        *int     `json:"UDMACRCErrors,omitempty"`
	SSDLifeLeft          *int     `json:"SSDLifeLeft,omitempty"`
	SASGrownDefects      *int     `json:"SASGrownDefects,omitempty"`
	SASUncorrectedErrors *int     `json:"SASUncorrectedErrors,omitempty"`
	// Proxmox-specific (filled by the Proxmox source, empty elsewhere)
	ZFSPool        string `json:"ZFSPool,omitempty"`
	ZFSHealth      string `json:"ZFSHealth,omitempty"`
	ZFSCapacityPct int    `json:"ZFSCapacityPct,omitempty"`
	ZFSFragPct     int    `json:"ZFSFragPct,omitempty"`
	ProxmoxNode    string `json:"ProxmoxNode,omitempty"`
}
