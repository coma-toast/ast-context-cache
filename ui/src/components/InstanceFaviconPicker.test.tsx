import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SettingsData } from '../api/types'
import { checkFaviconFile, FAVICON_KEYS, FAVICON_MAX_BYTES } from '../lib/favicon'
import { buttonByText, clickAndSettle, mountHarness, typeInto, type Harness } from '../test/settingsHarness'
import { InstanceSettingsSection } from './InstanceSettingsSection'

let h: Harness

beforeEach(() => {
  h = mountHarness()
})

afterEach(() => h.unmount())

// The picker lives in the General card and shares its draft.
const render = (data: Partial<SettingsData>) => h.render(<InstanceSettingsSection data={data as SettingsData} />)

const picker = () => h.container.querySelector<HTMLElement>('[data-testid="favicon-picker"]')!
const radio = (value: string) => h.container.querySelector<HTMLInputElement>(`input[type="radio"][value="${value}"]`)!
const badgeInput = () => h.container.querySelector<HTMLInputElement>('input[aria-label="Favicon badge"]')
const colorInput = () => h.container.querySelector<HTMLInputElement>('input[type="color"]')
const checkbox = () => picker().querySelector<HTMLInputElement>('input[type="checkbox"]')
const fileInput = () => h.container.querySelector<HTMLInputElement>('[data-testid="favicon-file"]')!
const save = () => buttonByText(h.container, 'Save')

const chooseFile = (file: File) =>
  act(async () => {
    Object.defineProperty(fileInput(), 'files', { value: [file], configurable: true })
    fileInput().dispatchEvent(new Event('change', { bubbles: true }))
  })

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

describe('InstanceFaviconPicker', () => {
  it('shows the current favicon and switches modes on Save', async () => {
    render({ InstanceFaviconMode: '', InstanceFaviconURL: '/dashboard/favicon?v=abc' })
    expect(picker().querySelector('img')!.getAttribute('src')).toBe('/dashboard/favicon?v=abc')
    expect(radio('default').checked).toBe(true)
    expect(colorInput()).toBeNull()
    expect(checkbox()).toBeNull()
    act(() => radio('color').click())
    expect(h.saveBatch).not.toHaveBeenCalled()
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenLastCalledWith({ [FAVICON_KEYS.mode]: 'color', [FAVICON_KEYS.color]: '#863bff' })
  })

  it('saves a color and a badge together in color mode', async () => {
    render({ InstanceFaviconMode: 'color', InstanceFaviconColor: '#863bff', InstanceFaviconBadge: '' })
    expect(save().disabled).toBe(true)
    typeInto(colorInput()!, '#00aa00')
    typeInto(badgeInput()!, 'M')
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenCalledTimes(1)
    expect(h.saveBatch).toHaveBeenCalledWith({ [FAVICON_KEYS.color]: '#00aa00', [FAVICON_KEYS.badge]: 'M' })
  })

  it('uploads a small PNG right away and rejects big or wrong-type files without saving', async () => {
    render({ InstanceFaviconMode: 'default' })
    await chooseFile(new File([new Uint8Array(FAVICON_MAX_BYTES + 1)], 'big.png', { type: 'image/png' }))
    expect(h.container.textContent).toContain('big.png is 65 KB; the limit is 64 KB')
    await chooseFile(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    expect(h.container.textContent).toContain('photo.jpg is not a PNG, SVG or ICO file')
    expect(h.saveBatch).not.toHaveBeenCalled()

    await chooseFile(new File([PNG], 'icon.png', { type: 'image/png' }))
    await vi.waitFor(() => expect(h.saveBatch).toHaveBeenCalled())
    const [changes] = h.saveBatch.mock.calls[0]
    expect(Object.keys(changes)).toEqual([FAVICON_KEYS.upload])
    expect(changes[FAVICON_KEYS.upload].startsWith('data:image/png;base64,')).toBe(true)
    await vi.waitFor(() => expect(h.refresh).toHaveBeenCalled())
    expect(h.container.querySelector('[role="alert"]')).toBeNull()
  })

  it('picks the stored upload without asking for a file again', async () => {
    render({ InstanceFaviconMode: 'default', InstanceFaviconUploadType: 'image/png', InstanceFaviconUploadBytes: 2048 })
    expect(h.container.textContent).toContain('Upload (image/png, 2 KB)')
    act(() => radio('upload').click())
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenLastCalledWith({ [FAVICON_KEYS.mode]: 'upload' })
  })

  it('locks to the env color until the override box is checked', async () => {
    const env: Partial<SettingsData> = { InstanceFaviconExternalColor: '#ff0000', InstanceFaviconLocked: true, InstanceFaviconMode: 'upload', InstanceFaviconBadge: 'D' }
    render(env)
    expect(radio('color').checked).toBe(true)
    expect(radio('default').disabled).toBe(true)
    expect(colorInput()!.value).toBe('#ff0000')
    expect(colorInput()!.disabled).toBe(true)
    expect(badgeInput()!.value).toBe('')
    expect(h.container.textContent).toContain('Set by AST_INSTANCE_FAVICON_COLOR')
    expect(checkbox()!.checked).toBe(false)

    // Checking the box unlocks the stored settings in the draft.
    act(() => checkbox()!.click())
    expect(radio('upload').checked).toBe(true)
    expect(radio('default').disabled).toBe(false)
    expect(h.container.textContent).toContain('Overrides AST_INSTANCE_FAVICON_COLOR')
    await clickAndSettle(save())
    expect(h.saveBatch).toHaveBeenLastCalledWith({ [FAVICON_KEYS.override]: 'true' })
  })
})

describe('checkFaviconFile', () => {
  it.each([
    ['icon.svg', 'image/svg+xml', 10, ''],
    ['favicon.ico', '', 10, ''],
    ['favicon.ico', 'image/vnd.microsoft.icon', 10, ''],
    ['limit.png', 'image/png', FAVICON_MAX_BYTES, ''],
    ['over.png', 'image/png', FAVICON_MAX_BYTES + 1, 'the limit is 64 KB'],
    ['page.html', 'text/html', 10, 'not a PNG, SVG or ICO'],
  ])('%s (%s, %d bytes)', (name, type, size, want) => {
    const got = checkFaviconFile(new File([new Uint8Array(size)], name, { type }))
    if (want) expect(got).toContain(want)
    else expect(got).toBe('')
  })
})
