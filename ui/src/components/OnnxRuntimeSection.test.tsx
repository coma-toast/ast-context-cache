import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ONNXRuntimeStatus, SettingsData } from '../api/types'

const installONNXRuntime = vi.fn()
const restartNow = vi.fn()
vi.mock('../api/client', () => ({ api: { installONNXRuntime: () => installONNXRuntime(), restartNow: () => restartNow() } }))

import { OnnxRuntimeSection } from './OnnxRuntimeSection'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const missing: ONNXRuntimeStatus = {
  available: false,
  reason: 'ONNX Runtime library not found at /usr/lib/libonnxruntime.so (default)',
  loaded: false,
  path: '/usr/lib/libonnxruntime.so',
  source: 'default',
  managed: false,
  pinned: '1.31.0',
  update_available: false,
  restart_required: false,
  install_supported: true,
}

const unavailable = {
  AuxAvailable: false,
  AuxUnavailableReason: missing.reason,
  EmbedAuxBackend: 'onnx',
  ONNXRuntime: missing,
} as SettingsData

let container: HTMLDivElement
let root: Root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
  installONNXRuntime.mockReset()
  restartNow.mockReset()
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

const text = () => container.textContent ?? ''
const button = (label: string) =>
  Array.from(container.querySelectorAll('button')).find((b) => b.textContent === label) as HTMLButtonElement | undefined

describe('OnnxRuntimeSection', () => {
  it('shows the unavailable card with an install button, then offers a restart', async () => {
    const onRefresh = vi.fn()
    installONNXRuntime.mockResolvedValue({ status: 'installed', version: '1.31.0', restart_required: true })
    act(() => root.render(<OnnxRuntimeSection data={unavailable} onRefresh={onRefresh} />))
    expect(text()).toContain('Local (aux) embedder unavailable: ONNX Runtime library not found')
    const install = button('Install ONNX Runtime')
    expect(install).toBeDefined()
    await act(async () => install!.click())
    expect(installONNXRuntime).toHaveBeenCalledTimes(1)
    expect(onRefresh).toHaveBeenCalled()
    expect(text()).toContain('ONNX Runtime 1.31.0 is installed. Restart ast-mcp to load it.')
    restartNow.mockResolvedValue({ status: 'restarting' })
    await act(async () => button('Restart now')!.click())
    expect(restartNow).toHaveBeenCalledTimes(1)
  })

  it('explains the env var when there is no managed build', () => {
    act(() => root.render(<OnnxRuntimeSection data={{ ...unavailable, ONNXRuntime: { ...missing, install_supported: false } }} onRefresh={vi.fn()} />))
    expect(button('Install ONNX Runtime')).toBeUndefined()
    expect(text()).toContain('set ONNXRUNTIME_LIB')
  })

  it('shows the installed version and an update button', () => {
    const ort: ONNXRuntimeStatus = {
      ...missing,
      available: true,
      reason: undefined,
      version: '1.30.0',
      path: '/data/lib/onnxruntime-1.30.0/libonnxruntime.so',
      source: 'managed',
      managed: true,
      installed: '1.30.0',
      update_available: true,
    }
    act(() => root.render(<OnnxRuntimeSection data={{ AuxAvailable: true, ONNXRuntime: ort } as SettingsData} onRefresh={vi.fn()} />))
    expect(text()).toContain('ONNX Runtime 1.30.0 · managed')
    expect(button('Update to 1.31.0')).toBeDefined()
    expect(button('Install ONNX Runtime')).toBeUndefined()
  })
})
