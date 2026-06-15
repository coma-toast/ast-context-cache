package sys

// DiskIO holds sampled host disk throughput (primary internal block device).
type DiskIO struct {
	ReadMBps  float64
	WriteMBps float64
}

// SSDHealth summarizes boot/internal SSD SMART and identity (best-effort per OS).
type SSDHealth struct {
	Available      bool
	Device         string
	Model          string
	SmartStatus    string
	Protocol       string
	Capacity       string
	SolidState     bool
	TrimSupport    bool
	WearUsedPct    int     // NVMe PERCENTAGE_USED; -1 if unknown
	SparePct       int     // NVMe AVAILABLE_SPARE; -1 if unknown
	DataWrittenTB  float64 // lifetime host writes in TB; -1 if unknown
	TemperatureC   float64 // Celsius; -1 if unknown
}
