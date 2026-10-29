import { createContext, useContext, useEffect, useState } from 'react'

import { api } from '../api/client'
import type { ApiError, SettingsBatchResult } from '../api/types'
import { useToast } from '../context/ToastContext'
import { changedKeys, draftChanges, draftValue, editsAfterSave, SECRET_CLEARED, SECRET_SET } from '../lib/settingsDraft'

/** What a Settings section needs from the page to save a draft. */
export interface SettingsSaveApi {
  /** Saves the changes in one request; rejects with ApiError.fields naming the refused keys. */
  saveBatch: (changes: Record<string, string>) => Promise<SettingsBatchResult>
  /** Reloads settings after a save; the draft keeps showing the saved values until it settles. */
  refresh: () => void | Promise<void>
  /** Reports whether a section has unsaved edits (for the leave-page warning and the nav). */
  markDirty: (section: string, dirty: boolean) => void
}

const standalone: SettingsSaveApi = { saveBatch: (changes) => api.saveSettings(changes), refresh: () => {}, markDirty: () => {} }

/** Provided by SettingsTab; sections rendered alone (stories, tests) save straight to the API. */
export const SettingsSaveContext = createContext<SettingsSaveApi>(standalone)

/** Reports a section's unsaved state to the page, and clears it when the section unmounts. */
export const useReportDirty = (section: string, dirty: boolean) => {
  const { markDirty } = useContext(SettingsSaveContext)
  useEffect(() => {
    markDirty(section, dirty)
  }, [markDirty, section, dirty])
  useEffect(() => () => markDirty(section, false), [markDirty, section])
}

export interface SettingsDraft {
  /** The value to show in a field. Secrets show only what was typed. */
  value: (key: string) => string
  set: (key: string, value: string) => void
  /** Marks a set secret for removal on save. */
  clearSecret: (key: string) => void
  /** Whether a secret is stored on the server and not marked for removal. */
  secretSet: (key: string) => boolean
  dirty: boolean
  /** The server's (or the client check's) message for each refused key. */
  errors: Record<string, string>
  saving: boolean
  save: () => Promise<void>
  discard: () => void
}

interface DraftOptions {
  /** Password fields: blank means unchanged; clearSecret sends "". */
  secrets?: string[]
  /** Client-side checks run before sending; any message blocks the save. */
  validate?: (changes: Record<string, string>) => Record<string, string>
}

/**
 * Holds a Settings section's unsaved edits over the values the server last returned. Save sends
 * the changed keys as one batch, keeps the draft and shows per-field errors when the server
 * refuses it, and never reports "saved" for a rejected value. Discard drops the edits.
 */
export const useSettingsDraft = (section: string, label: string, saved: Record<string, string>, options: DraftOptions = {}): SettingsDraft => {
  const ctx = useContext(SettingsSaveContext)
  const { showToast } = useToast()
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [committed, setCommitted] = useState<Record<string, string>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const secrets = new Set(options.secrets ?? [])
  const changed = changedKeys(saved, edits, committed, secrets)
  const dirty = changed.length > 0
  useReportDirty(section, dirty)

  const set = (key: string, value: string) => {
    setEdits((prev) => ({ ...prev, [key]: value }))
    setErrors((prev) => {
      const { [key]: _drop, ...rest } = prev
      return rest
    })
  }
  const discard = () => {
    setEdits({})
    setErrors({})
  }
  const save = async () => {
    if (!dirty || saving) return
    const changes = draftChanges(changed, edits)
    const invalid = options.validate?.(changes) ?? {}
    if (Object.keys(invalid).length > 0) {
      setErrors(invalid)
      showToast(`${label} not saved: fix the highlighted fields`, 'error')
      return
    }
    setSaving(true)
    try {
      const res = await ctx.saveBatch(changes)
      const stored: Record<string, string> = { ...changes, ...(res.values ?? {}) }
      for (const k of Object.keys(changes)) if (secrets.has(k)) stored[k] = changes[k] ? SECRET_SET : ''
      setCommitted((prev) => ({ ...prev, ...stored }))
      setEdits((prev) => editsAfterSave(prev, changes))
      setErrors({})
      if (res.reload_error) showToast(`${label} saved, but the embedder did not reload: ${res.reload_error}`, 'error')
      else showToast(`${label} saved`, 'success')
      try {
        await ctx.refresh()
      } finally {
        setCommitted({})
      }
    } catch (e) {
      const fields = (e as Partial<ApiError>).fields ?? {}
      setErrors(fields)
      showToast(`${label} not saved: ${e instanceof Error ? e.message : String(e)}`, 'error')
    } finally {
      setSaving(false)
    }
  }
  return {
    value: (key) => draftValue(key, saved, edits, committed, secrets),
    set,
    clearSecret: (key) => set(key, SECRET_CLEARED),
    secretSet: (key) => edits[key] !== SECRET_CLEARED && (committed[key] ?? saved[key]) === SECRET_SET,
    dirty,
    errors,
    saving,
    save,
    discard,
  }
}
