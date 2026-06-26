package db

// InvalidateSummariesForFile removes cached LLM summaries after the file is re-indexed.
func InvalidateSummariesForFile(filePath, projectPath string) {
	if filePath == "" || projectPath == "" {
		return
	}
	IndexDB.Exec("DELETE FROM summaries WHERE file_path = ? AND project_path = ?", filePath, projectPath)
}
