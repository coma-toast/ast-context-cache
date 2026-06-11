package sys

// DiskIO holds sampled host disk throughput (primary internal block device).
type DiskIO struct {
	ReadMBps  float64
	WriteMBps float64
}

// SSDHealth summarizes boot/internal SSD SMART and identity (best-effort per OS).
type SSDHealth struct {
	Available   bool
	Device      string
	Model       string
	SmartStatus string
	Protocol    string
	Capacity    string
	SolidState  bool
	TrimSupport bool
}
