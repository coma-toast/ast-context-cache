import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { SettingsData } from '../api/types'
import { buttonByText, clickAndSettle, mountHarness, type Harness, typeInto } from '../test/settingsHarness'
import { EmbeddingSection, HostDiskSection, PerformanceSection, WatcherSection } from './SettingsFormSections'

let h: Harness

beforeEach(() => {
  h = mountHarness()
})

afterEach(() => h.unmount())

const embedData = (extra: Partial<SettingsData> = {}) =>
  ({ EmbedBackend: 'onnx', EmbedHTTPURL: '', EmbedHTTPBearer: '', EmbedOpenAIAPIKey: '', EmbedActiveBackend: 'onnx', EmbedActiveModel: 'm', ...extra }) as SettingsData

const field = (label: string) =>
  [...h.container.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>('input, textarea')].find((el) => el.closest('.MuiFormControl-root')?.querySelector('label')?.textContent?.startsWith(label))!

describe('EmbeddingSection', () => {
  it('saves backend, URL and bearer in one batch', async () => {
    h.render(<EmbeddingSection data={embedData()} />)
    act(() => {
      h.container.querySelector<HTMLElement>('[role="combobox"]')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }))
    })
    act(() => [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((o) => o.textContent === 'http')!.click())
    typeInto(field('HTTP URL'), 'http://embed.local/embed')
    typeInto(field('HTTP bearer'), 'tok')
    await clickAndSettle(buttonByText(h.container, 'Save'))
    expect(h.saveBatch).toHaveBeenCalledTimes(1)
    expect(h.saveBatch).toHaveBeenCalledWith({ EMBED_BACKEND: 'http', EMBED_HTTP_URL: 'http://embed.local/embed', EMBED_HTTP_BEARER: 'tok' })
  })

  it('leaves a stored secret alone when blank and sends "" only after Clear', async () => {
    h.render(<EmbeddingSection data={embedData({ EmbedOpenAIAPIKeySet: true })} />)
    expect(field('OpenAI API key').value).toBe('')
    expect(h.container.textContent).toContain('Saved. Leave blank to keep it')
    expect(buttonByText(h.container, 'Save').disabled).toBe(true)
    typeInto(field('OpenAI API key'), 'x')
    typeInto(field('OpenAI API key'), '')
    expect(buttonByText(h.container, 'Save').disabled).toBe(true)
    act(() => buttonByText(h.container, 'Clear').click())
    expect(h.container.textContent).toContain('Removed when you save')
    await clickAndSettle(buttonByText(h.container, 'Save'))
    expect(h.saveBatch).toHaveBeenCalledWith({ EMBED_OPENAI_API_KEY: '' })
  })
})

describe('WatcherSection', () => {
  it('toggles .log indexing as a draft', async () => {
    h.render(<WatcherSection data={{ WatcherIgnoreGlobs: '[]', IndexLogFiles: false } as SettingsData} />)
    const toggle = h.container.querySelector<HTMLInputElement>('input[type="checkbox"]')!
    act(() => toggle.click())
    expect(h.saveBatch).not.toHaveBeenCalled()
    await clickAndSettle(buttonByText(h.container, 'Save'))
    expect(h.saveBatch).toHaveBeenCalledWith({ index_log_files: 'true' })
  })
})

describe('PerformanceSection', () => {
  const auxLabels = () => ['Max aux embed workers', 'Aux embed workers', 'Aux embed backend'].filter((l) => h.container.textContent?.includes(l))

  it('hides the aux controls when the aux embedder is unavailable', () => {
    h.render(<PerformanceSection data={{ EmbedWorkerMax: 15, AuxAvailable: false } as SettingsData} />)
    expect(auxLabels()).toHaveLength(0)
  })

  it('shows them when available, and for servers that do not report availability', () => {
    h.render(<PerformanceSection data={{ EmbedWorkerMax: 15, AuxAvailable: true } as SettingsData} />)
    expect(auxLabels()).toHaveLength(3)
    h.render(<PerformanceSection data={{ EmbedWorkerMax: 15 } as SettingsData} />)
    expect(auxLabels()).toHaveLength(3)
  })
})

describe('HostDiskSection', () => {
  it('saves the URL as one key', async () => {
    h.render(<HostDiskSection data={{} as SettingsData} />)
    expect(buttonByText(h.container, 'Save').disabled).toBe(true)
    typeInto(field('Host disk health URL'), 'http://192.168.1.10:9633/metrics')
    await clickAndSettle(buttonByText(h.container, 'Save'))
    expect(h.saveBatch).toHaveBeenCalledWith({ host_disk_health_url: 'http://192.168.1.10:9633/metrics' })
  })

  it.each([
    ['192.168.1.10:9633/metrics', 'Must start with http:// or https://'],
    ['http://root:pw@pve:9633/metrics', 'Must not contain a user name or password'],
  ])('blocks %s before sending it', async (value, msg) => {
    h.render(<HostDiskSection data={{} as SettingsData} />)
    typeInto(field('Host disk health URL'), value)
    await clickAndSettle(buttonByText(h.container, 'Save'))
    expect(h.saveBatch).not.toHaveBeenCalled()
    expect(h.container.textContent).toContain(msg)
  })

  it('clears a saved URL with an empty value', async () => {
    h.render(<HostDiskSection data={{ HostDiskHealthURL: 'http://pve:9633/metrics' } as SettingsData} />)
    expect(field('Host disk health URL').value).toBe('http://pve:9633/metrics')
    typeInto(field('Host disk health URL'), '')
    await clickAndSettle(buttonByText(h.container, 'Save'))
    expect(h.saveBatch).toHaveBeenCalledWith({ host_disk_health_url: '' })
  })

  it('keeps the setup steps and the guide link next to the field', () => {
    h.render(<HostDiskSection data={{} as SettingsData} />)
    expect(h.container.textContent).toContain('How to set this up')
    expect(h.container.textContent).toContain('grep smartctl_device')
    const link = [...h.container.querySelectorAll('a')].find((a) => a.href.endsWith('docs/host-disk-health.md'))
    expect(link).toBeDefined()
  })
})
