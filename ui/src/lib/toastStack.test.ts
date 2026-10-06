import { describe, expect, it } from 'vitest'

import {
  emptyToastStack,
  MAX_VISIBLE_TOASTS,
  pushToast,
  releaseGroup,
  removeToast,
  shouldHoldIndexToast,
  TOAST_DURATION_MS,
  type StackedToast,
  type ToastStackState,
} from './toastStack'

const toast = (key: number, extra: Partial<StackedToast> = {}): StackedToast => ({
  key,
  message: `t${key}`,
  severity: 'info',
  held: false,
  durationMs: TOAST_DURATION_MS,
  ...extra,
})

const pushAll = (n: number, state: ToastStackState = emptyToastStack) => {
  for (let i = 0; i < n; i++) state = pushToast(state, toast(i))
  return state
}

describe('pushToast', () => {
  it('shows up to the cap at once and queues the rest in order', () => {
    const s = pushAll(MAX_VISIBLE_TOASTS + 2)
    expect(s.visible.map((t) => t.key)).toEqual([...Array(MAX_VISIBLE_TOASTS).keys()])
    expect(s.queued.map((t) => t.key)).toEqual([MAX_VISIBLE_TOASTS, MAX_VISIBLE_TOASTS + 1])
  })

  it('hands a group hold to the newest toast', () => {
    let s = pushToast(emptyToastStack, toast(1, { group: 'p', held: true }))
    s = pushToast(s, toast(2, { group: 'q', held: true }))
    s = pushToast(s, toast(3, { group: 'p', held: true }))
    expect(s.visible.map((t) => [t.key, t.held])).toEqual([[1, false], [2, true], [3, true]])
    expect(s.visible[0].durationMs).toBe(TOAST_DURATION_MS)
  })

  it('releases the old hold even when the newer toast is not held', () => {
    let s = pushToast(emptyToastStack, toast(1, { group: 'p', held: true }))
    s = pushToast(s, toast(2, { group: 'p' }))
    expect(s.visible.every((t) => !t.held)).toBe(true)
  })
})

describe('releaseGroup', () => {
  it('lets only that group close after the delay, visible or queued', () => {
    let s = pushAll(MAX_VISIBLE_TOASTS - 1)
    s = pushToast(s, toast(100, { group: 'p', held: true }))
    s = pushToast(s, toast(101, { group: 'q', held: true }))
    s = releaseGroup(s, 'q', 1000)
    expect(s.queued[0]).toMatchObject({ key: 101, held: false, durationMs: 1000 })
    expect(s.visible.at(-1)).toMatchObject({ key: 100, held: true })
  })
})

describe('removeToast', () => {
  it('promotes queued toasts into the freed slot', () => {
    const s = removeToast(pushAll(MAX_VISIBLE_TOASTS + 1), 0)
    expect(s.visible).toHaveLength(MAX_VISIBLE_TOASTS)
    expect(s.visible.at(-1)?.key).toBe(MAX_VISIBLE_TOASTS)
    expect(s.queued).toHaveLength(0)
  })

  it('drops a queued toast without touching the visible ones', () => {
    const s = removeToast(pushAll(MAX_VISIBLE_TOASTS + 1), MAX_VISIBLE_TOASTS)
    expect(s.visible).toHaveLength(MAX_VISIBLE_TOASTS)
    expect(s.queued).toHaveLength(0)
  })
})

describe('shouldHoldIndexToast', () => {
  it('holds while work is pending and no newer drain was seen', () => {
    expect(shouldHoldIndexToast(3, 0, undefined)).toBe(true)
    expect(shouldHoldIndexToast(3, 2, 2)).toBe(true)
  })

  it('does not hold an idle queue or a toast flushed before a drain already seen', () => {
    expect(shouldHoldIndexToast(0, 2, 2)).toBe(false)
    expect(shouldHoldIndexToast(3, 1, 2)).toBe(false)
  })
})
