import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SettingsData } from '../api/types'
import { checkFaviconFile, FAVICON_KEYS, FAVICON_MAX_BYTES } from '../lib/favicon'
import { InstanceFaviconPicker } from './InstanceFaviconPicker'

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
  act(() => root.render(<InstanceFaviconPicker data={data as SettingsData} save={save} />))

const radio = (value: string) => container.querySelector<HTMLInputElement>(`input[type="radio"][value="${value}"]`)!
const badgeInput = () => container.querySelector<HTMLInputElement>('input[aria-label="Favicon badge"]')
const colorInput = () => container.querySelector<HTMLInputElement>('input[type="color"]')
const button = (label: string) => [...container.querySelectorAll('button')].find((b) => b.textContent === label)
const checkbox = () => container.querySelector<HTMLInputElement>('input[type="checkbox"]')
const fileInput = () => container.querySelector<HTMLInputElement>('[data-testid="favicon-file"]')!

const setValue = (input: HTMLInputElement, value: string) =>
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value)
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })

const chooseFile = (file: File) =>
  act(async () => {
    Object.defineProperty(fileInput(), 'files', { value: [file], configurable: true })
    fileInput().dispatchEvent(new Event('change', { bubbles: true }))
  })

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

describe('InstanceFaviconPicker', () => {
  it('shows the current favicon and switches modes', () => {
    const save = vi.fn(async () => {})
    render({ InstanceFaviconMode: '', InstanceFaviconURL: '/dashboard/favicon?v=abc' }, save)
    expect(container.querySelector('img')!.getAttribute('src')).toBe('/dashboard/favicon?v=abc')
    expect(radio('default').checked).toBe(true)
    expect(colorInput()).toBeNull()
    expect(checkbox()).toBeNull()
    act(() => radio('color').click())
    expect(save).toHaveBeenLastCalledWith(FAVICON_KEYS.mode, 'color')
  })

  it('applies a color and a badge in color mode', async () => {
    const save = vi.fn(async () => {})
    render({ InstanceFaviconMode: 'color', InstanceFaviconColor: '#863bff', InstanceFaviconBadge: '' }, save)
    expect(button('Apply')!.disabled).toBe(true)
    setValue(colorInput()!, '#00aa00')
    setValue(badgeInput()!, 'M')
    await act(async () => button('Apply')!.click())
    expect(save).toHaveBeenCalledWith(FAVICON_KEYS.color, '#00aa00')
    expect(save).toHaveBeenCalledWith(FAVICON_KEYS.badge, 'M')
  })

  it('uploads a small PNG and rejects big or wrong-type files without saving', async () => {
    const save = vi.fn(async () => {})
    render({ InstanceFaviconMode: 'default' }, save)
    await chooseFile(new File([new Uint8Array(FAVICON_MAX_BYTES + 1)], 'big.png', { type: 'image/png' }))
    expect(container.textContent).toContain('big.png is 65 KB; the limit is 64 KB')
    await chooseFile(new File(['x'], 'photo.jpg', { type: 'image/jpeg' }))
    expect(container.textContent).toContain('photo.jpg is not a PNG, SVG or ICO file')
    expect(save).not.toHaveBeenCalled()

    await chooseFile(new File([PNG], 'icon.png', { type: 'image/png' }))
    await vi.waitFor(() => expect(save).toHaveBeenCalled())
    const [key, value] = save.mock.calls[0] as unknown as [string, string]
    expect(key).toBe(FAVICON_KEYS.upload)
    expect(value.startsWith('data:image/png;base64,')).toBe(true)
    expect(container.querySelector('[role="alert"]')).toBeNull()
  })

  it('picks the stored upload without asking for a file again', () => {
    const save = vi.fn(async () => {})
    render({ InstanceFaviconMode: 'default', InstanceFaviconUploadType: 'image/png', InstanceFaviconUploadBytes: 2048 }, save)
    expect(container.textContent).toContain('Upload (image/png, 2 KB)')
    act(() => radio('upload').click())
    expect(save).toHaveBeenLastCalledWith(FAVICON_KEYS.mode, 'upload')
  })

  it('locks to the env color until the override box is checked', () => {
    const save = vi.fn(async () => {})
    const env: Partial<SettingsData> = { InstanceFaviconExternalColor: '#ff0000', InstanceFaviconLocked: true, InstanceFaviconMode: 'upload', InstanceFaviconBadge: 'D' }
    render(env, save)
    expect(radio('color').checked).toBe(true)
    expect(radio('default').disabled).toBe(true)
    expect(colorInput()!.value).toBe('#ff0000')
    expect(colorInput()!.disabled).toBe(true)
    expect(badgeInput()!.value).toBe('')
    expect(container.textContent).toContain('Set by AST_INSTANCE_FAVICON_COLOR')
    expect(checkbox()!.checked).toBe(false)
    act(() => checkbox()!.click())
    expect(save).toHaveBeenLastCalledWith(FAVICON_KEYS.override, 'true')

    // Override on: the stored settings come back and are editable.
    render({ ...env, InstanceFaviconLocked: false, InstanceFaviconOverride: true }, save)
    expect(radio('upload').checked).toBe(true)
    expect(radio('default').disabled).toBe(false)
    expect(container.textContent).toContain('Overrides AST_INSTANCE_FAVICON_COLOR')
    act(() => checkbox()!.click())
    expect(save).toHaveBeenLastCalledWith(FAVICON_KEYS.override, 'false')
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
