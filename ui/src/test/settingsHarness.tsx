import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { vi } from 'vitest'

import type { ApiError, SettingsBatchResult } from '../api/types'
import { SettingsSaveContext, type SettingsSaveApi } from '../hooks/useSettingsDraft'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

/** A mounted test root plus the save API the sections see. */
export interface Harness {
  container: HTMLDivElement
  saveBatch: ReturnType<typeof vi.fn<(changes: Record<string, string>) => Promise<SettingsBatchResult>>>
  refresh: ReturnType<typeof vi.fn<() => Promise<void>>>
  markDirty: ReturnType<typeof vi.fn<(section: string, dirty: boolean) => void>>
  render: (ui: ReactNode) => void
  unmount: () => void
}

export const mountHarness = (): Harness => {
  const container = document.createElement('div')
  document.body.appendChild(container)
  const root: Root = createRoot(container)
  const saveBatch = vi.fn(async (changes: Record<string, string>): Promise<SettingsBatchResult> => ({ status: 'ok', values: changes }))
  const refresh = vi.fn(async () => {})
  const markDirty = vi.fn()
  const api: SettingsSaveApi = { saveBatch, refresh, markDirty }
  return {
    container,
    saveBatch,
    refresh,
    markDirty,
    render: (ui) => act(() => root.render(<SettingsSaveContext.Provider value={api}>{ui}</SettingsSaveContext.Provider>)),
    unmount: () => {
      act(() => root.unmount())
      container.remove()
    },
  }
}

/** The error a 4xx settings save rejects with. */
export const rejection = (message: string, fields: Record<string, string>): ApiError =>
  Object.assign(new Error(message), { code: '', repreview: false, fields })

/** Types into a React-controlled input or textarea. */
export const typeInto = (input: HTMLInputElement | HTMLTextAreaElement, value: string) =>
  act(() => {
    const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
    Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(input, value)
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })

export const buttonByText = (container: HTMLElement, text: string) =>
  [...container.querySelectorAll('button')].find((b) => b.textContent === text)!

/** Clicks a button and lets the save promise chain settle. */
export const clickAndSettle = (button: HTMLButtonElement) =>
  act(async () => {
    button.click()
  })
