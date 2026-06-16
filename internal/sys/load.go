package sys

// LoadAvg holds host load averages (1/5/15 minute).
type LoadAvg struct {
	Available bool
	Load1     float64
	Load5     float64
	Load15    float64
}

// HostLoadAverage returns system load averages when supported on this platform.
func HostLoadAverage() LoadAvg {
	return hostLoadAverage()
}
