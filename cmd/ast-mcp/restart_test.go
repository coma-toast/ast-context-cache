package main

import (
	"errors"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDoRestartExits1WhenExecReturnsError(t *testing.T) {
	var exitCode int = -1
	fakeExit := func(code int) { exitCode = code }

	fakeExec := func(exe string, argv []string, env []string) error {
		return errors.New("exec failed")
	}

	var startCalled bool
	fakeStartChild := func(exe string, args []string, env []string, attr *os.ProcAttr) (*os.Process, error) {
		startCalled = true
		return nil, nil
	}

	doRestart("/fake/exe", fakeExec, fakeStartChild, fakeExit)

	assert.Equal(t, 1, exitCode)
	assert.False(t, startCalled)
}

func TestDoRestartTriggersFallbackWhenExecHangs(t *testing.T) {
	var exitCode int = -1
	fakeExit := func(code int) { exitCode = code }

	// Fake exec that blocks forever (simulating hang)
	execChan := make(chan struct{})
	fakeExec := func(exe string, argv []string, env []string) error {
		<-execChan
		return nil
	}

	var startCalled bool
	fakeStartChild := func(exe string, args []string, env []string, attr *os.ProcAttr) (*os.Process, error) {
		startCalled = true
		return nil, nil
	}

	// Override restartWatchdogTimeout temporarily for fast test
	origTimeout := restartWatchdogTimeout
	restartWatchdogTimeout = 10 * time.Millisecond
	t.Cleanup(func() { restartWatchdogTimeout = origTimeout })

	doRestart("/fake/exe", fakeExec, fakeStartChild, fakeExit)

	assert.Equal(t, 0, exitCode)
	assert.True(t, startCalled, "fallback child should be started on watchdog timeout")

	// Clean up hanging goroutine
	close(execChan)
}
