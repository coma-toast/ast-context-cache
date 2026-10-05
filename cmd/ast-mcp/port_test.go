package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEnvPort(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  int
	}{
		{name: "unset uses default", value: "", want: 7821},
		{name: "valid port", value: "7999", want: 7999},
		{name: "surrounding space", value: " 8001 ", want: 8001},
		{name: "not a number", value: "abc", want: 7821},
		{name: "zero", value: "0", want: 7821},
		{name: "negative", value: "-1", want: 7821},
		{name: "out of range", value: "70000", want: 7821},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(envMCPPort, tt.value)
			assert.Equal(t, tt.want, envPort(envMCPPort, 7821))
		})
	}
}
