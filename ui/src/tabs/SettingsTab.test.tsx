import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fixtureSettings } from '../storybook/fixtures'
import { buttonByText, clickAndSettle, typeInto } from '../test/settingsHarness'

const saveSettings = vi.fn(async (settings: Record<string, string>) => ({ status: 'ok', values: settings }))

// Everything else the page fetches stays pending, so only the save path runs.
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, api: new Proxy({}, { get: (_t, key) => (key === 'saveSettings' ? saveSettings : () => new Promise(() => {})) }) }
})

const { SettingsTab } = await import('./SettingsTab')

let container: HTMLDivElement
let root: Root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
  saveSettings.mockClear()
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

const idleField = () =>
  [...container.querySelectorAll<HTMLInputElement>('input')].find((el) => el.closest('.MuiFormControl-root')?.querySelector('label')?.textContent?.startsWith('Idle unload'))!

const leavePage = () => {
  const e = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(e)
  return e.defaultPrevented
}

describe('SettingsTab', () => {
  it('warns before leaving while a section is unsaved, and saves only that section', async () => {
    const onRefresh = vi.fn(async () => {})
    act(() => root.render(<SettingsTab data={fixtureSettings} mcpTier={null} onRefresh={onRefresh} />))
    expect(leavePage()).toBe(false)
    typeInto(idleField(), '45')
    expect(leavePage()).toBe(true)
    expect(container.textContent).toContain('Performance •')
    const card = container.querySelector<HTMLElement>('#settings-performance')!
    await clickAndSettle(buttonByText(card, 'Save'))
    expect(saveSettings).toHaveBeenCalledTimes(1)
    expect(saveSettings).toHaveBeenCalledWith({ idle_unload_minutes: '45' })
    expect(onRefresh).toHaveBeenCalledTimes(1)
    expect(leavePage()).toBe(false)
    expect(container.textContent).not.toContain('Performance •')
  })
})
