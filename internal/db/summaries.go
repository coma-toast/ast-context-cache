package db

const deleteSummariesForFileQuery = "DELETE FROM summaries WHERE file_path = ? AND project_path = ?"

// InvalidateSummariesForFile removes cached LLM summaries after the file is re-indexed.
func InvalidateSummariesForFile(filePath, projectPath string) {
	if filePath == "" || projectPath == "" {
		return
	}
	if conn, err := IndexReader(); err == nil {
		conn.Exec(deleteSummariesForFileQuery, filePath, projectPath)
	}
}
