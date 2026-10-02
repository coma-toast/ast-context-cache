package embedder

import "github.com/coma-toast/ast-context-cache/internal/db/dbtest"

// A test that calls Reload starts the connectivity probe, which runs until
// stopped and reads settings and package vars later tests set.
func init() { dbtest.WaitFor(stopConnectivityProbeForTest) }

// stopConnectivityProbeForTest stops the connectivity probe and waits for its
// goroutine to return.
func stopConnectivityProbeForTest() {
	stopConnectivityProbe()
	probeLoops.Wait()
}
