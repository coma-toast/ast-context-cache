import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { SettingsData } from '../api/types'
import { OVERRIDE_KEY } from '../lib/settingsSections'
import { buttonByText, clickAndSettle, mountHarness, rejection, typeInto, type Harness } from '../test/settingsHarness'
import { InstanceSettingsSection } from './InstanceSettingsSection'

let h: Harness

beforeEach(() => {
  h = mountHarness()
})

afterEach(() => h.unmount())

const render = (data: Partial<SettingsData>) => h.render(<InstanceSettingsSection data={data as SettingsData} />)

const nameInput = () => h.container.querySelector<HTMLInputElement>('input[type="text"]:not([aria-label]), input:not([type]):not([aria-label])')!
const checkbox = () => h.container.querySelector<HTMLInputElement>('input[type="checkbox"]')
const save = () => buttonByText(h.container, 'Save')

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
  it('edits freely with no env or flag name, and shows no override box', async () => {
    render({ InstanceName: 'jd', InstanceNameEffective: 'jd', InstanceNameSource: 'settings' })
    expect(checkbox()).toBeNull()
    expect(nameInput().disabled).toBe(false)
    expect(nameInput().value).toBe('jd')
    expect(save().disabled).toBe(true)
    typeInto(nameInput(), 'slide')
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenCalledWith({ instance_name: 'slide' })
  })

  it('saves the override box and the new name together', async () => {
    // Env set, override off: env value read-only, box unchecked.
    render(envSet)
    expect(nameInput().disabled).toBe(true)
    expect(nameInput().value).toBe('fromenv')
    expect(save().disabled).toBe(true)
    expect(checkbox()!.checked).toBe(false)
    expect(h.container.textContent).toContain('Override the environment value')
    expect(h.container.textContent).toContain('other browsers must log in again')

    // Checking the box is a draft: the field unlocks at the current name, nothing is sent yet.
    act(() => checkbox()!.click())
    expect(h.saveBatch).not.toHaveBeenCalled()
    expect(checkbox()!.checked).toBe(true)
    expect(nameInput().disabled).toBe(false)
    expect(nameInput().value).toBe('fromenv')
    typeInto(nameInput(), 'dash')
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenLastCalledWith({ instance_name: 'dash', [OVERRIDE_KEY]: 'true' })

    // Saved: the override name is live. Unchecking is a draft that reverts to the env value.
    render({ ...envSet, InstanceName: 'dash', InstanceNameEffective: 'dash', InstanceNameSource: 'override', InstanceNameLocked: false, InstanceNameOverride: true })
    expect(nameInput().value).toBe('dash')
    expect(h.container.textContent).toContain('Active: dash (from override)')
    act(() => checkbox()!.click())
    expect(nameInput().disabled).toBe(true)
    expect(nameInput().value).toBe('fromenv')
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenLastCalledWith({ [OVERRIDE_KEY]: 'false' })
  })

  it('shows a rejected name inline and never reports it saved', async () => {
    h.saveBatch.mockRejectedValueOnce(rejection('invalid instance_name: bad char', { instance_name: 'invalid instance_name: bad char' }))
    render({ InstanceName: 'jd', InstanceNameEffective: 'jd', InstanceNameSource: 'settings' })
    typeInto(nameInput(), 'bad/name')
    await clickAndSettle(save())
    expect(h.container.textContent).toContain('invalid instance_name: bad char')
    expect(nameInput().value).toBe('bad/name')
    expect(h.refresh).not.toHaveBeenCalled()
  })

  it('names the -name flag when the flag sets the name', () => {
    render({ ...envSet, InstanceNameSource: 'flag', InstanceNameExternalSource: 'flag' })
    expect(h.container.textContent).toContain('Override the -name flag value')
    expect(h.container.textContent).toContain('Set by the -name flag')
  })
})
