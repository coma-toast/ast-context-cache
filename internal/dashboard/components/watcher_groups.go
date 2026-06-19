package components

import (
	"sort"
	"strings"
)

// WatcherSpaceGroup is a WTG workspace with its repo checkouts.
type WatcherSpaceGroup struct {
	Space    string
	Watchers []WatcherInfo
}

// GroupWatchers splits watchers into local (non-WTG) repos and space-grouped worktrees.
func GroupWatchers(watchers []WatcherInfo) (local []WatcherInfo, spaces []WatcherSpaceGroup) {
	bySpace := map[string][]WatcherInfo{}
	for _, w := range watchers {
		if strings.TrimSpace(w.Workspace) == "" {
			local = append(local, w)
			continue
		}
		bySpace[w.Workspace] = append(bySpace[w.Workspace], w)
	}
	sortWatchers(local)
	spaceNames := make([]string, 0, len(bySpace))
	for name := range bySpace {
		spaceNames = append(spaceNames, name)
	}
	sort.Strings(spaceNames)
	for _, name := range spaceNames {
		items := bySpace[name]
		sortWatchers(items)
		spaces = append(spaces, WatcherSpaceGroup{Space: name, Watchers: items})
	}
	return local, spaces
}

func sortWatchers(ws []WatcherInfo) {
	sort.Slice(ws, func(i, j int) bool {
		li, lj := strings.ToLower(ws[i].Label), strings.ToLower(ws[j].Label)
		if li != lj {
			return li < lj
		}
		return ws[i].ProjectPath < ws[j].ProjectPath
	})
}
