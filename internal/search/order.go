package search

// LessScored orders hits best first: score descending, then file, start_line and
// name ascending, so tied scores always come back in the same order.
func LessScored(a, b ScoredResult) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	if fa, fb := dataString(a.Data, "file"), dataString(b.Data, "file"); fa != fb {
		return fa < fb
	}
	if la, lb := dataInt(a.Data, "start_line"), dataInt(b.Data, "start_line"); la != lb {
		return la < lb
	}
	return dataString(a.Data, "name") < dataString(b.Data, "name")
}

// LessVector orders vector matches best first: similarity descending, then
// SourceFile, Name and ID ascending.
func LessVector(a *VectorEntry, simA float64, b *VectorEntry, simB float64) bool {
	if simA != simB {
		return simA > simB
	}
	if a.SourceFile != b.SourceFile {
		return a.SourceFile < b.SourceFile
	}
	if a.Name != b.Name {
		return a.Name < b.Name
	}
	return a.ID < b.ID
}

func dataString(d map[string]interface{}, key string) string {
	s, _ := d[key].(string)
	return s
}

func dataInt(d map[string]interface{}, key string) int {
	switch v := d[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 0
}
