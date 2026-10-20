// Per-instance localStorage scoping (JD-283).
//
// Two ast-context-cache instances can share one origin (e.g. behind Tailscale
// Serve or a reverse proxy on port 443), so every key is prefixed with the
// instance scope: `astcache_<scope>_<key>`. The scope comes from the settings
// payload (`InstanceScope`); until it loads, the last-known scope cached in the
// unscoped bootstrap key is used so first paint already reads the right slot.

let currentScope = ''
try {
  currentScope = localStorage.getItem('astcache_instance_scope') || ''
} catch {
  /* storage unavailable */
}

export function setInstanceScope(scope: string) {
  currentScope = scope || ''
  try {
    localStorage.setItem('astcache_instance_scope', currentScope)
  } catch {
    /* ignore */
  }
}

export function scopedKey(key: string): string {
  return currentScope ? `astcache_${currentScope}_${key}` : `astcache_${key}`
}

export function getScopedItem(key: string): string | null {
  try {
    const namespaced = localStorage.getItem(scopedKey(key))
    if (namespaced !== null) return namespaced
    // One-time migration from the pre-scoping unprefixed key.
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

export function setScopedItem(key: string, value: string) {
  try {
    localStorage.setItem(scopedKey(key), value)
  } catch {
    /* ignore */
  }
}
