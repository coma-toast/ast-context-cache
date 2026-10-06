export type ToastSeverity = 'success' | 'error' | 'info'

export interface StackedToast {
  key: number
  message: string
  severity: ToastSeverity
  // Toasts in one group share a single hold: a newer toast takes it over and
  // the older one falls back to a normal timer.
  group?: string
  held: boolean
  durationMs: number
}

export interface ToastStackState {
  visible: StackedToast[]
  queued: StackedToast[]
}

export const MAX_VISIBLE_TOASTS = 10
export const TOAST_DURATION_MS = 4000
// A held toast still closes after this long, in case its release never
// arrives (a dropped WebSocket, a server restart).
export const MAX_HOLD_MS = 120_000

export const emptyToastStack: ToastStackState = { visible: [], queued: [] }

const unholdGroup = (list: StackedToast[], group: string | undefined) =>
  group === undefined ? list : list.map((t) => (t.group === group && t.held ? { ...t, held: false, durationMs: TOAST_DURATION_MS } : t))

export function pushToast(state: ToastStackState, toast: StackedToast): ToastStackState {
  const visible = unholdGroup(state.visible, toast.group)
  const queued = unholdGroup(state.queued, toast.group)
  return visible.length < MAX_VISIBLE_TOASTS ? { visible: [...visible, toast], queued } : { visible, queued: [...queued, toast] }
}

// releaseGroup lets a group's held toast close delayMs after it is shown (or
// from now, when it already is).
export function releaseGroup(state: ToastStackState, group: string, delayMs: number): ToastStackState {
  const release = (list: StackedToast[]) => list.map((t) => (t.group === group && t.held ? { ...t, held: false, durationMs: delayMs } : t))
  return { visible: release(state.visible), queued: release(state.queued) }
}

export function removeToast(state: ToastStackState, key: number): ToastStackState {
  const visible = state.visible.filter((t) => t.key !== key)
  if (visible.length === state.visible.length) return { visible, queued: state.queued.filter((t) => t.key !== key) }
  const free = MAX_VISIBLE_TOASTS - visible.length
  return { visible: [...visible, ...state.queued.slice(0, free)], queued: state.queued.slice(free) }
}

// shouldHoldIndexToast holds an indexing toast open while its project still
// has work queued. drainGen guards the race where the drain message beats a
// toast flushed before it: such a toast carries an older generation.
export function shouldHoldIndexToast(pending: number, drainGen: number, lastDrainGen: number | undefined): boolean {
  return pending > 0 && drainGen >= (lastDrainGen ?? 0)
}
