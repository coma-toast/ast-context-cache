import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { IndexHealth, QueueCapState, SettingsData } from '../api/types'
import { buttonByText, clickAndSettle, mountHarness, rejection, typeInto, type Harness } from '../test/settingsHarness'

import { EmbedQueueSettingsSection } from './EmbedQueueSettingsSection'

const caps: QueueCapState[] = [
  { key: 'embed_queue_pending_cap', env: 'EMBED_QUEUE_PENDING_CAP', label: 'Retry (pending)', default: 128, hard_max: 1024, value: 128, used: 0, env_locked: false },
  { key: 'embed_queue_high_cap', env: 'EMBED_QUEUE_HIGH_CAP', label: 'Priority', default: 128, hard_max: 1024, value: 128, used: 0, env_locked: false },
  { key: 'embed_queue_low_cap', env: 'EMBED_QUEUE_LOW_CAP', label: 'Background', default: 2048, hard_max: 16384, value: 4096, used: 0, env_locked: true },
]

let h: Harness

beforeEach(() => {
  h = mountHarness()
})

afterEach(() => h.unmount())

const render = (health?: Partial<IndexHealth>, withCaps = caps) =>
  h.render(<EmbedQueueSettingsSection data={{ EmbedQueueCaps: withCaps } as SettingsData} indexHealth={health as IndexHealth | undefined} />)

const input = (key: string) => h.container.querySelector<HTMLInputElement>(`input[aria-label="${key}"]`)!
const save = () => buttonByText(h.container, 'Save')
const discard = () => buttonByText(h.container, 'Discard')

describe('EmbedQueueSettingsSection', () => {
  it('renders one control per cap with live depth and the env lock', () => {
    render({ EmbedPendingQueued: 2, EmbedHighQueued: 64, EmbedLowQueued: 1000 })
    expect(input('embed_queue_pending_cap').value).toBe('128')
    expect(input('embed_queue_low_cap').value).toBe('4096')
    expect(input('embed_queue_low_cap').disabled).toBe(true)
    expect(input('embed_queue_high_cap').disabled).toBe(false)
    const text = h.container.textContent ?? ''
    expect(text).toContain('Set by EMBED_QUEUE_LOW_CAP')
    expect(text).toContain('64 queued now · max 1024')
    expect(text).toContain('64 / 128 queued')
    expect(text).toContain('1000 / 4096 queued')
  })

  it('enables Save only for a changed value and sends just the changed keys', async () => {
    render()
    expect(save().disabled).toBe(true)
    expect(discard().disabled).toBe(true)
    typeInto(input('embed_queue_high_cap'), '128')
    expect(save().disabled).toBe(true)
    typeInto(input('embed_queue_high_cap'), '256')
    expect(save().disabled).toBe(false)
    expect(h.container.textContent).toContain('Unsaved changes')
    expect(h.markDirty).toHaveBeenLastCalledWith('embed-queue', true)
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenCalledTimes(1)
    expect(h.saveBatch).toHaveBeenCalledWith({ embed_queue_high_cap: '256' })
    expect(h.refresh).toHaveBeenCalledTimes(1)
    expect(h.markDirty).toHaveBeenLastCalledWith('embed-queue', false)
  })

  it('keeps an edit when the settings refetch lands mid-typing', () => {
    render()
    typeInto(input('embed_queue_pending_cap'), '300')
    render({ EmbedPendingQueued: 5 }, caps.map((c) => ({ ...c })))
    expect(input('embed_queue_pending_cap').value).toBe('300')
  })

  it('discards edits back to the saved value', () => {
    render()
    typeInto(input('embed_queue_pending_cap'), '300')
    act(() => discard().click())
    expect(input('embed_queue_pending_cap').value).toBe('128')
    expect(save().disabled).toBe(true)
  })

  it('blocks out-of-range values in the browser without saving', async () => {
    render()
    typeInto(input('embed_queue_pending_cap'), '2000')
    await clickAndSettle(save())
    expect(h.saveBatch).not.toHaveBeenCalled()
    expect(h.container.textContent).toContain('Must be 1–1024')
    expect(input('embed_queue_pending_cap').value).toBe('2000')
  })

  it('shows a server rejection on the field and keeps the draft', async () => {
    h.saveBatch.mockRejectedValueOnce(rejection('embed_queue_high_cap is set by EMBED_QUEUE_HIGH_CAP', { embed_queue_high_cap: 'embed_queue_high_cap is set by EMBED_QUEUE_HIGH_CAP' }))
    render()
    typeInto(input('embed_queue_high_cap'), '256')
    await clickAndSettle(save())
    expect(h.container.textContent).toContain('embed_queue_high_cap is set by EMBED_QUEUE_HIGH_CAP')
    expect(input('embed_queue_high_cap').getAttribute('aria-invalid')).toBe('true')
    expect(input('embed_queue_high_cap').value).toBe('256')
    expect(save().disabled).toBe(false)
    expect(h.refresh).not.toHaveBeenCalled()
  })

  it('shows a fallback when caps are missing', () => {
    h.render(<EmbedQueueSettingsSection data={{} as SettingsData} />)
    expect(h.container.textContent).toContain('Embed queue limits unavailable.')
  })
})
