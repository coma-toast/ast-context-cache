package handoff

import "github.com/coma-toast/ast-context-cache/internal/db"

const (
	selectChildLineageQuery = `SELECT h.parent_session_id, COALESCE(c.project_path, h.project_path, '')
		FROM handoff_children c JOIN handoffs h ON h.ref = c.handoff_ref WHERE c.child_session_id = ?`
	selectTreeLineageQuery = `SELECT root_session_id, COALESCE(project_path, '') FROM handoff_trees WHERE tree_id = ?`
)

// lineage is where a session sits for lifecycle logs: its tree's root, or its handoff's parent,
// and the project.
type lineage struct {
	parent  SessionID
	project string
}

// lifecycleArgs are the keys every lifecycle event carries (OB-6): the tree and handoff through
// their LogValuers, the parent and child session ids, and the project path, then extra.
// Empty values are left out, so a root session's claim has no child_session.
func lifecycleArgs(tree TreeID, ref HandoffRef, parent, child SessionID, project string, extra ...any) []any {
	args := make([]any, 0, 8+len(extra))
	if tree != "" {
		args = append(args, tree.Attr())
	}
	if ref != "" {
		args = append(args, ref.Attr())
	}
	if parent != "" {
		args = append(args, "parent_session", string(parent))
	}
	if child != "" {
		args = append(args, "child_session", string(child))
	}
	if project != "" {
		args = append(args, "project_path", project)
	}
	return append(args, extra...)
}

// sessionEventArgs are lifecycleArgs for an event by sid, a session of te's tree, reading its
// parent and project through q. A failed read only leaves those keys out.
func sessionEventArgs(q rowQuerier, sid SessionID, te treeEntry, extra ...any) []any {
	if !te.isChild {
		l := treeLineage(q, te.tree)
		return lifecycleArgs(te.tree, "", sid, "", l.project, extra...)
	}
	var l lineage
	if q != nil {
		_ = q.QueryRow(selectChildLineageQuery, string(sid)).Scan(&l.parent, &l.project)
	}
	return lifecycleArgs(te.tree, te.handoff, l.parent, sid, l.project, extra...)
}

// contextReader is db.ContextDB as a rowQuerier, nil (not a nil *sql.DB) before it is open.
func contextReader() rowQuerier {
	if db.ContextDB == nil {
		return nil
	}
	return db.ContextDB
}

// treeLineage reads tree's root session and project; zero when the read fails.
func treeLineage(q rowQuerier, tree TreeID) lineage {
	var l lineage
	if q != nil {
		_ = q.QueryRow(selectTreeLineageQuery, string(tree)).Scan(&l.parent, &l.project)
	}
	return l
}
