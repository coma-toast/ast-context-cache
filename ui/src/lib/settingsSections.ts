import type { ChangeEvent } from 'react'
import type { SxProps, Theme } from '@mui/material/styles'

import type { SettingsData } from '../api/types'
import type { SettingsDraft } from '../hooks/useSettingsDraft'
import { FAVICON_KEYS } from './favicon'
import { SECRET_SET } from './settingsDraft'

/** Card sx for a Settings section: the border turns amber while it has unsaved edits. */
export const settingsCardSx = (dirty: boolean): SxProps<Theme> => ({
  mb: 2,
  scrollMarginTop: { xs: 120, md: 120 },
  borderColor: dirty ? 'warning.main' : undefined,
})

/** TextField props bound to one draft key: value, change handler and the inline error. */
export const draftField = (draft: SettingsDraft, key: string, helper?: string) => ({
  value: draft.value(key),
  onChange: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => draft.set(key, e.target.value),
  error: !!draft.errors[key],
  helperText: draft.errors[key] ?? helper,
})

/** Settings key for the instance name. */
export const NAME_KEY = 'instance_name'

/** Settings key that lets the dashboard name win over -name / AST_INSTANCE_NAME. */
export const OVERRIDE_KEY = 'instance_name_override'

/**
 * The General card's saved values. With an external name and no stored one, the name starts as
 * the current name so checking the override box does not look like a rename.
 */
export const generalSaved = (data: SettingsData): Record<string, string> => ({
  [NAME_KEY]: data.InstanceNameExternal ? data.InstanceName || data.InstanceNameEffective || '' : (data.InstanceName ?? ''),
  [OVERRIDE_KEY]: String(!!data.InstanceNameOverride),
  [FAVICON_KEYS.mode]: data.InstanceFaviconMode || 'default',
  [FAVICON_KEYS.color]: data.InstanceFaviconColor ?? '',
  [FAVICON_KEYS.badge]: data.InstanceFaviconBadge ?? '',
  [FAVICON_KEYS.override]: String(!!data.InstanceFaviconOverride),
})

/** The Performance card's saved values; the probe interval only when the server reports it. */
export const performanceSaved = (data: SettingsData): Record<string, string> => ({
  idle_unload_minutes: String(data.IdleUnloadMinutes),
  embed_worker_max: String(data.EmbedWorkerMax ?? 15),
  embed_aux_worker_max: String(data.EmbedAuxWorkerMax ?? 10),
  EMBED_AUX_WORKERS: String(data.EmbedAuxWorkers ?? 0),
  EMBED_AUX_BACKEND: data.EmbedAuxBackend || 'onnx',
  ...(data.EmbedProbeIntervalSec != null ? { embed_probe_interval_seconds: String(data.EmbedProbeIntervalSec) } : {}),
})

export const virtualContextSaved = (data: SettingsData): Record<string, string> => ({
  context_max_notes_session: String(data.ContextMaxNotesSession),
  context_max_tokens_session: String(data.ContextMaxTokensSession),
  context_max_notes_global: String(data.ContextMaxNotesGlobal),
  context_max_tokens_global: String(data.ContextMaxTokensGlobal),
})

/** Embedding secrets: the API only says whether they are set. */
export const EMBED_SECRET_KEYS = ['EMBED_HTTP_BEARER', 'EMBED_OPENAI_API_KEY']

export const embeddingSaved = (data: SettingsData): Record<string, string> => ({
  EMBED_BACKEND: data.EmbedBackend || 'onnx',
  EMBED_HTTP_URL: data.EmbedHTTPURL ?? '',
  EMBED_HTTP_BEARER: data.EmbedHTTPBearerSet ? SECRET_SET : '',
  EMBED_OPENAI_API_KEY: data.EmbedOpenAIAPIKeySet ? SECRET_SET : '',
})

export const watcherSaved = (data: SettingsData): Record<string, string> => ({
  watcher_ignore_globs: data.WatcherIgnoreGlobs ?? '',
  index_log_files: String(!!data.IndexLogFiles),
})

export const retentionSaved = (data: SettingsData): Record<string, string> => ({
  query_retention_max_age_days: String(data.QueryRetentionMaxAgeDays),
})

export const storageSaved = (data: SettingsData): Record<string, string> => ({
  transcript_usage_ingest: String(!!data.TranscriptUsageIngest),
})
