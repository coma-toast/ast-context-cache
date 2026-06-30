package dashboard

import "github.com/coma-toast/ast-context-cache/internal/db"

func usageDBReady() bool { return db.DB != nil }
func indexDBReady() bool { return db.IndexDB != nil }
func contextDBReady() bool { return db.ContextDB != nil }
