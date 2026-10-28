import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SettingsData } from '../api/types'
import { InstanceSettingsSection, OVERRIDE_KEY } from './InstanceSettingsSection'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

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

const render = (data: Partial<SettingsData>, save: (key: string, value: string) => Promise<void>) =>
  act(() => root.render(<InstanceSettingsSection data={data as SettingsData} save={save} />))

const nameInput = () => container.querySelector<HTMLInputElement>('input[type="text"], input:not([type])')!
const checkbox = () => container.querySelector<HTMLInputElement>('input[type="checkbox"]')
const saveButton = () => [...container.querySelectorAll('button')].find((b) => b.textContent === 'Save')!

/** Types into a React-controlled input. */
const typeInto = (input: HTMLInputElement, value: string) =>
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value)
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })

const envSet: Partial<SettingsData> = {
  InstanceName: '',
  InstanceNameEffective: 'fromenv',
  InstanceNameSource: 'env',
  InstanceNameExternal: 'fromenv',
  InstanceNameExternalSource: 'env',
  InstanceNameLocked: true,
  InstanceNameOverride: false,
}

describe('InstanceSettingsSection', () => {
  it('edits freely with no env or flag name, and shows no override box', () => {
    const save = vi.fn(async () => {})
    render({ InstanceName: 'jd', InstanceNameEffective: 'jd', InstanceNameSource: 'settings' }, save)
    expect(checkbox()).toBeNull()
    expect(nameInput().disabled).toBe(false)
    expect(nameInput().value).toBe('jd')
    typeInto(nameInput(), 'slide')
    act(() => saveButton().click())
    expect(save).toHaveBeenCalledWith('instance_name', 'slide')
  })

  it('runs the override checkbox flow for an env-set name', () => {
    const save = vi.fn(async () => {})
    // Env set, override off: env value read-only, box unchecked.
    render(envSet, save)
    expect(nameInput().disabled).toBe(true)
    expect(nameInput().value).toBe('fromenv')
    expect(saveButton().disabled).toBe(true)
    expect(checkbox()!.checked).toBe(false)
    expect(container.textContent).toContain('Override the environment value')
    expect(container.textContent).toContain('other browsers must log in again')

    // Checking persists the override.
    act(() => checkbox()!.click())
    expect(save).toHaveBeenLastCalledWith(OVERRIDE_KEY, 'true')

    // Server now reports the override with no stored name: field enabled, starts at the effective name.
    render({ ...envSet, InstanceNameLocked: false, InstanceNameOverride: true }, save)
    expect(checkbox()!.checked).toBe(true)
    expect(nameInput().disabled).toBe(false)
    expect(nameInput().value).toBe('fromenv')
    expect(saveButton().disabled).toBe(true)
    typeInto(nameInput(), 'dash')
    act(() => saveButton().click())
    expect(save).toHaveBeenLastCalledWith('instance_name', 'dash')

    // Saved: the override name is live. Unchecking reverts to the env value.
    render({ ...envSet, InstanceName: 'dash', InstanceNameEffective: 'dash', InstanceNameSource: 'override', InstanceNameLocked: false, InstanceNameOverride: true }, save)
    expect(nameInput().value).toBe('dash')
    expect(container.textContent).toContain('Active: dash (from override)')
    act(() => checkbox()!.click())
    expect(save).toHaveBeenLastCalledWith(OVERRIDE_KEY, 'false')

    // Override off with a stored name: env value shown read-only again.
    render({ ...envSet, InstanceName: 'dash' }, save)
    expect(nameInput().disabled).toBe(true)
    expect(nameInput().value).toBe('fromenv')
  })

  it('names the -name flag when the flag sets the name', () => {
    render({ ...envSet, InstanceNameSource: 'flag', InstanceNameExternalSource: 'flag' }, vi.fn(async () => {}))
    expect(container.textContent).toContain('Override the -name flag value')
    expect(container.textContent).toContain('Set by the -name flag')
  })
})
