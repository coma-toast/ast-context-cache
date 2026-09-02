package db

// InvalidateSummariesForFile removes cached LLM summaries after the file is re-indexed.
func InvalidateSummariesForFile(filePath, projectPath string) {
	if filePath == "" || projectPath == "" {
		return
	}
	if conn, err := IndexReader(); err == nil {
		conn.Exec("DELETE FROM summaries WHERE file_path = ? AND project_path = ?", filePath, projectPath)
	}
}
