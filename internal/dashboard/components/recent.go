package components

type RecentQuery struct {
	Timestamp        string
	TimestampTitle   string
	ToolName         string
	Query            string
	Mode             string
	Budget           int
	Saved            int
	DedupTokensSaved int
	Project          string
	DurationMs       float64
	CpuMs            float64
	Error            string
	Event            string
	File             string
	FileTitle        string
}

type RecentLogLine struct {
	Timestamp    string
	Level        string
	Message      string
	Raw          string
	MsgTruncated bool
}

type LogViewOpts struct {
	TailLines    int
	MaxLineChars int
}
