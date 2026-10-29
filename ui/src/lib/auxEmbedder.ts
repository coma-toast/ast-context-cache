import type { SettingsData } from '../api/types'

/**
 * Whether Settings hides the aux embedder controls: the aux backend is onnx and ONNX Runtime
 * can't load. Older servers don't send AuxAvailable, so only an explicit false hides them.
 */
export const auxUnavailable = (data: Pick<SettingsData, 'AuxAvailable'>) => data.AuxAvailable === false
