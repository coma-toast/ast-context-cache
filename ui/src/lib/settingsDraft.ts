/**
 * Draft logic for a Settings section: the saved values come from the server, edits hold only the
 * keys the user touched, so a settings refetch never overwrites what is being typed.
 */

/** Edit value of a secret field the user explicitly cleared (sent as ""). */
export const SECRET_CLEARED = '\u0000cleared'

/** Saved value of a secret that is set on the server; the browser never sees the real one. */
export const SECRET_SET = '\u0000set'

/** The value a field shows: the edit, else the last committed save, else the saved value. */
export const draftValue = (
  key: string,
  saved: Record<string, string>,
  edits: Record<string, string>,
  committed: Record<string, string>,
  secrets: ReadonlySet<string>,
): string => {
  if (secrets.has(key)) {
    const v = edits[key]
    return v === undefined || v === SECRET_CLEARED ? '' : v
  }
  return edits[key] ?? committed[key] ?? saved[key] ?? ''
}

/**
 * Keys whose edit differs from what is saved. A secret is changed when a new value was typed
 * (blank means unchanged) or when it was set and the user cleared it.
 */
export const changedKeys = (
  saved: Record<string, string>,
  edits: Record<string, string>,
  committed: Record<string, string>,
  secrets: ReadonlySet<string>,
): string[] =>
  Object.keys(edits)
    .filter((k) => {
      const v = edits[k]
      if (secrets.has(k)) return v === SECRET_CLEARED ? (committed[k] ?? saved[k]) === SECRET_SET : v !== ''
      return v !== (committed[k] ?? saved[k] ?? '')
    })
    .sort()

/** The request body for the changed keys: a cleared secret is sent as "". */
export const draftChanges = (keys: string[], edits: Record<string, string>): Record<string, string> =>
  Object.fromEntries(keys.map((k) => [k, edits[k] === SECRET_CLEARED ? '' : edits[k]]))

/** Edits left after a save: keys edited again while the request was in flight stay. */
export const editsAfterSave = (edits: Record<string, string>, sent: Record<string, string>): Record<string, string> =>
  Object.fromEntries(
    Object.entries(edits).filter(([k, v]) => !(k in sent) || (v === SECRET_CLEARED ? sent[k] !== '' : v !== sent[k])),
  )
