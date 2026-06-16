//go:build !darwin && !linux

package sys

func hostLoadAverage() LoadAvg {
	return LoadAvg{}
}
