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
// Integer counters use -1 when the device does not report them; SmartSource
// names the tool that produced the SMART verdict.
type DiskInfo struct {
	Device       string // "/dev/nvme0n1" (Linux), "disk0" (macOS), or the Proxmox disk name
	Model        string
	Serial       string
	Protocol     string // "nvme", "sata", "sas", "scsi"
	SmartType    string // smartctl device type: "nvme", "sat", "usbsunplus", ...
	SmartStatus  string // "Verified", "Failing", "Unknown", "Not Supported"
	SmartSource  string // "diskutil", "smartctl", "proxmox", ""
	Capacity     string
	FreeSpace    string
	IsExternal   bool
	SolidState   bool
	TrimSupport  bool
	DataDirDisk  bool   // whether this disk holds the data dir
	WarningLevel string // "", "ok", "warning", "critical"
	WarningMsg   string // human-readable warning (e.g. permission denied)

	// SMART attributes (-1 when unknown)
	WearUsedPct          int // NVMe PERCENTAGE_USED
	SparePct             int // NVMe AVAILABLE_SPARE
	DataWrittenTB        float64
	TemperatureC         float64
	PowerOnHours         int
	PowerCycles          int
	UnsafeShutdowns      int
	MediaErrors          int // NVMe media errors
	CriticalWarning      int // NVMe critical warning bitmask
	ReallocatedSectors   int // attr 5
	PendingSectors       int // attr 197
	OfflineUncorrectable int // attr 198
	UDMACRCErrors        int // attr 199
	SSDLifeLeft          int // attr 231/233/202 (vendor dependent)
	SASGrownDefects      int // scsi_grown_defect_list
	SASUncorrectedErrors int // scsi error counter log total

	// Proxmox-specific (filled by the Proxmox source, empty elsewhere)
	ZFSPool        string // pool name if on ZFS
	ZFSHealth      string // pool health
	ZFSCapacityPct int    // pool capacity percent
	ZFSFragPct     int    // pool fragmentation percent
	ProxmoxNode    string // which Proxmox node this disk is on
}
