import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { IndexHealth, QueueCapState, SettingsData } from '../api/types'

import { EmbedQueueSettingsSection } from './EmbedQueueSettingsSection'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const caps: QueueCapState[] = [
  { key: 'embed_queue_pending_cap', env: 'EMBED_QUEUE_PENDING_CAP', label: 'Retry (pending)', default: 128, hard_max: 1024, value: 128, used: 0, env_locked: false },
  { key: 'embed_queue_high_cap', env: 'EMBED_QUEUE_HIGH_CAP', label: 'Priority', default: 128, hard_max: 1024, value: 128, used: 0, env_locked: false },
  { key: 'embed_queue_low_cap', env: 'EMBED_QUEUE_LOW_CAP', label: 'Background', default: 2048, hard_max: 16384, value: 4096, used: 0, env_locked: true },
]

let container: HTMLDivElement
let root: Root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

const render = (save: (key: string, value: string) => Promise<void>, health?: Partial<IndexHealth>) =>
  act(() => {
    root.render(
      <EmbedQueueSettingsSection
        data={{ EmbedQueueCaps: caps } as SettingsData}
        indexHealth={health as IndexHealth | undefined}
        save={save}
      />,
    )
  })

const input = (key: string) => container.querySelector<HTMLInputElement>(`input[aria-label="${key}"]`)!

const blurWith = (key: string, value: string) =>
  act(() => {
    const el = input(key)
    el.value = value
    el.dispatchEvent(new FocusEvent('focusout', { bubbles: true }))
  })

describe('EmbedQueueSettingsSection', () => {
  it('renders one control per cap with live depth and the env lock', () => {
    render(vi.fn(), { EmbedPendingQueued: 2, EmbedHighQueued: 64, EmbedLowQueued: 1000 })
    expect(input('embed_queue_pending_cap').value).toBe('128')
    expect(input('embed_queue_low_cap').value).toBe('4096')
    expect(input('embed_queue_low_cap').disabled).toBe(true)
    expect(input('embed_queue_high_cap').disabled).toBe(false)
    const text = container.textContent ?? ''
    expect(text).toContain('Set by EMBED_QUEUE_LOW_CAP')
    expect(text).toContain('64 queued now · max 1024')
    expect(text).toContain('64 / 128 queued')
    expect(text).toContain('1000 / 4096 queued')
  })

  it('saves a valid changed value on blur', () => {
    const save = vi.fn().mockResolvedValue(undefined)
    render(save)
    blurWith('embed_queue_high_cap', '256')
    expect(save).toHaveBeenCalledWith('embed_queue_high_cap', '256')
  })

  it('does not save an unchanged value', () => {
    const save = vi.fn().mockResolvedValue(undefined)
    render(save)
    blurWith('embed_queue_high_cap', '128')
    expect(save).not.toHaveBeenCalled()
  })

  it('rejects out-of-range values without saving', () => {
    const save = vi.fn().mockResolvedValue(undefined)
    render(save)
    blurWith('embed_queue_pending_cap', '2000')
    expect(save).not.toHaveBeenCalled()
    expect(container.textContent).toContain('Must be 1–1024')
    blurWith('embed_queue_pending_cap', '0')
    expect(save).not.toHaveBeenCalled()
  })

  it('shows a fallback when caps are missing', () => {
    act(() => root.render(<EmbedQueueSettingsSection data={{} as SettingsData} save={vi.fn()} />))
    expect(container.textContent).toContain('Embed queue limits unavailable.')
  })
})
