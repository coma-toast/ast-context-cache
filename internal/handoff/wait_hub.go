package handoff

import "sync"

// waitHub lets collect long-poll a tree (FI-6): each tree has one channel that is closed, and
// replaced, whenever one of its children changes status.
type waitHub struct {
	mu    sync.Mutex
	chans map[TreeID]chan struct{}
}

func newWaitHub() *waitHub {
	return &waitHub{chans: map[TreeID]chan struct{}{}}
}

// wait returns a channel that is closed at tree's next notify. Take it before reading the
// state being waited on, so a change between the read and the wait isn't missed.
func (w *waitHub) wait(tree TreeID) <-chan struct{} {
	w.mu.Lock()
	defer w.mu.Unlock()
	ch, ok := w.chans[tree]
	if !ok {
		ch = make(chan struct{})
		w.chans[tree] = ch
	}
	return ch
}

// notify wakes everyone waiting on tree.
func (w *waitHub) notify(tree TreeID) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if ch, ok := w.chans[tree]; ok {
		close(ch)
		delete(w.chans, tree)
	}
}
