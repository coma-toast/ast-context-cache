//go:build darwin

package sys

import (
	"encoding/binary"

	"golang.org/x/sys/unix"
)

const darwinLoadScale = 2048.0

func hostLoadAverage() LoadAvg {
	b, err := unix.SysctlRaw("vm.loadavg")
	if err != nil || len(b) < 12 {
		return LoadAvg{}
	}
	load1 := float64(binary.LittleEndian.Uint32(b[0:4])) / darwinLoadScale
	load5 := float64(binary.LittleEndian.Uint32(b[4:8])) / darwinLoadScale
	load15 := float64(binary.LittleEndian.Uint32(b[8:12])) / darwinLoadScale
	return LoadAvg{Available: true, Load1: load1, Load5: load5, Load15: load15}
}
