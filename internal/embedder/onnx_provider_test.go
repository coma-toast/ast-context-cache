package embedder

import "testing"

func TestResolveONNXProvider(t *testing.T) {
	cases := []struct {
		value, goos, want string
	}{
		{"", "darwin", onnxProviderCoreML},
		{"auto", "darwin", onnxProviderCoreML},
		{"", "linux", onnxProviderCPU},
		{"AUTO", "linux", onnxProviderCPU},
		{"cpu", "darwin", onnxProviderCPU},
		{" CoreML ", "linux", onnxProviderCoreML},
		{"cuda", "linux", onnxProviderCUDA},
		{"tpu", "darwin", onnxProviderCPU},
	}
	for _, c := range cases {
		if got := resolveONNXProvider(c.value, c.goos); got != c.want {
			t.Errorf("resolveONNXProvider(%q, %q) = %q, want %q", c.value, c.goos, got, c.want)
		}
	}
}

func TestNewONNXSessionOptionsCPUNeedsNone(t *testing.T) {
	opts, err := newONNXSessionOptions(onnxProviderCPU, "")
	if err != nil || opts != nil {
		t.Fatalf("cpu options = %v, %v; want nil, nil", opts, err)
	}
}

func TestONNXProviderIsAnEmbedSetting(t *testing.T) {
	if !IsSettingKey(onnxProviderSetting) {
		t.Fatalf("%s is not an embed setting key", onnxProviderSetting)
	}
}
