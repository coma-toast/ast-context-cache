//go:build linux

package sys

import (
	"os"
	"strconv"
	"strings"
)

func hostLoadAverage() LoadAvg {
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return LoadAvg{}
	}
	fields := strings.Fields(string(data))
	if len(fields) < 3 {
		return LoadAvg{}
	}
	load1, err1 := strconv.ParseFloat(fields[0], 64)
	load5, err2 := strconv.ParseFloat(fields[1], 64)
	load15, err3 := strconv.ParseFloat(fields[2], 64)
	if err1 != nil || err2 != nil || err3 != nil {
		return LoadAvg{}
	}
	return LoadAvg{Available: true, Load1: load1, Load5: load5, Load15: load15}
}
