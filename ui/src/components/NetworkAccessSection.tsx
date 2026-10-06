import { useCallback, useEffect, useRef, useState } from 'react'

import { Alert, Box, Button, Card, CardContent, Chip, Stack, TextField, Tooltip, Typography } from '@mui/material'

import type { NetworkKey, NetworkState, NetworkUpdate } from '../api/types'

import { useToast } from '../context/ToastContext'

import { api } from '../api/client'
import {
  generateToken,
  listenerChips,
  needsTokenWarning,
  networkLockReason,
  parseList,
  validateAddrs,
  validateHosts,
} from '../lib/network'

interface NetworkAccessSectionProps {
  /** Changes whenever App reloads settings (WebSocket `settings` panel, saves), so this refetches too. */
  refreshKey?: unknown
}

type ListKey = Extract<NetworkKey, 'listen_extra_addrs' | 'trusted_hosts'>

const LIST_FIELDS: { key: ListKey; label: string; placeholder: string; validate: (l: string[]) => string }[] = [
  {
    key: 'listen_extra_addrs',
    label: 'Extra listen addresses',
    placeholder: '100.101.102.103',
    validate: validateAddrs,
  },
  {
    key: 'trusted_hosts',
    label: 'Trusted hostnames',
    placeholder: 'my-laptop.tailnet-name.ts.net',
    validate: validateHosts,
  },
]

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e))

/**
 * Settings card for reaching the server beyond loopback (e.g. over Tailscale): extra listen
 * addresses opened live, hostnames trusted for Host/Origin checks, and the write-only access token
 * remote clients must present. Env-locked fields are read-only.
 */
export const NetworkAccessSection = ({ refreshKey }: NetworkAccessSectionProps) => {
  const { showToast } = useToast()
  const [net, setNet] = useState<NetworkState | null>(null)
  const [loadError, setLoadError] = useState('')
  const [drafts, setDrafts] = useState<Partial<Record<ListKey, string>>>({})
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<ListKey, string>>>({})
  const [busy, setBusy] = useState(false)
  const [enteringToken, setEnteringToken] = useState(false)
  const [tokenInput, setTokenInput] = useState('')
  const [generated, setGenerated] = useState('')
  // A refetch landing mid-save would overwrite the response with the pre-save state.
  const busyRef = useRef(false)

  const load = useCallback(async () => {
    try {
      const res = await api.network()
      if (busyRef.current) return
      setNet(res)
      setLoadError('')
    } catch (e) {
      setLoadError(errorText(e))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, refreshKey])

  const update = async (body: NetworkUpdate, success: string): Promise<boolean> => {
    busyRef.current = true
    setBusy(true)
    try {
      setNet(await api.setNetwork(body))
      showToast(success, 'success')
      return true
    } catch (e) {
      showToast(errorText(e), 'error')
      return false
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  const saved = (key: ListKey) => (net?.[key] ?? []).join('\n')
  const draft = (key: ListKey) => drafts[key] ?? saved(key)

  const saveList = async (key: ListKey, validate: (l: string[]) => string) => {
    const list = parseList(draft(key))
    const problem = validate(list)
    setFieldErrors((prev) => ({ ...prev, [key]: problem }))
    if (problem) return
    if (await update({ [key]: list.join('\n') }, 'Network access updated')) {
      setDrafts(({ [key]: _drop, ...rest }) => rest)
    }
  }

  const generate = async () => {
    const token = generateToken()
    if (await update({ remote_access_token: token }, 'Access token generated')) setGenerated(token)
  }

  const saveEnteredToken = async () => {
    if (await update({ remote_access_token: tokenInput.trim() }, 'Access token set')) {
      setTokenInput('')
      setEnteringToken(false)
      setGenerated('')
    }
  }

  const clearToken = async () => {
    if (await update({ remote_access_token: '' }, 'Access token cleared')) setGenerated('')
  }

  const copyGenerated = async () => {
    try {
      // navigator.clipboard only exists in secure contexts, which a plain-HTTP tailnet URL is not.
      await navigator.clipboard.writeText(generated)
      showToast('Token copied', 'success')
    } catch {
      showToast('Copy unavailable here; select the token and copy it manually', 'info')
    }
  }

  const tokenLock = networkLockReason(net, 'remote_access_token')
  const chips = listenerChips(net?.listeners)

  return (
    <Card variant="outlined" id="settings-network" sx={{ mb: 2, scrollMarginTop: { xs: 120, md: 120 } }}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          Network access
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1.5 }}>
          Both servers always listen on loopback. Add more addresses (such as this machine&apos;s Tailscale IP) to reach
          them from other devices; changes apply immediately. Loopback clients never need the access token.
        </Typography>
        {loadError && (
          <Alert severity="error" sx={{ mb: 1.5 }}>
            Network settings unavailable: {loadError}
          </Alert>
        )}
        {!net && !loadError && <Typography color="text.secondary">Loading network settings…</Typography>}
        {net && (
          <Stack spacing={2}>
            {needsTokenWarning(net) && (
              <Alert severity="warning" data-testid="network-token-warning">
                No access token is set: anyone who can reach this address can use the server. Generate a token below.
              </Alert>
            )}
            {net.base_wildcard && (
              <Alert severity="info">
                The base listen address ({net.base_listen || 'all interfaces'}) already covers every interface, so extra
                addresses are not opened separately; they still count as trusted hosts.
              </Alert>
            )}

            {LIST_FIELDS.map(({ key, label, placeholder, validate }) => {
              const lock = networkLockReason(net, key)
              const dirty = draft(key) !== saved(key)
              const helper =
                fieldErrors[key] ||
                lock ||
                (key === 'listen_extra_addrs'
                  ? (
                      <>
                        One IP per line. Find your Tailscale IP with <code>tailscale ip -4</code>.
                      </>
                    )
                  : 'Names browsers use to open the dashboard, e.g. the MagicDNS name. Extra addresses are trusted automatically.')
              return (
                <Box key={key}>
                  <Tooltip title={lock} disableHoverListener={!lock} disableFocusListener={!lock}>
                    {/* A disabled field fires no pointer events, so the box carries the tooltip. */}
                    <Box>
                      <TextField
                        label={label}
                        multiline
                        minRows={2}
                        fullWidth
                        size="small"
                        placeholder={placeholder}
                        disabled={!!lock || busy}
                        value={draft(key)}
                        error={!!fieldErrors[key]}
                        helperText={helper}
                        onChange={(e) => {
                          const value = e.target.value
                          setDrafts((prev) => ({ ...prev, [key]: value }))
                          setFieldErrors((prev) => ({ ...prev, [key]: '' }))
                        }}
                        slotProps={{ htmlInput: { 'aria-label': key, spellCheck: false } }}
                      />
                    </Box>
                  </Tooltip>
                  {!lock && (
                    <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                      <Button size="small" variant="outlined" disabled={!dirty || busy} onClick={() => void saveList(key, validate)}>
                        Save
                      </Button>
                      {dirty && (
                        <Button
                          size="small"
                          disabled={busy}
                          onClick={() => {
                            setDrafts(({ [key]: _drop, ...rest }) => rest)
                            setFieldErrors((prev) => ({ ...prev, [key]: '' }))
                          }}
                        >
                          Revert
                        </Button>
                      )}
                    </Stack>
                  )}
                </Box>
              )
            })}

            <Box>
              <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                Listener status
              </Typography>
              {chips.length === 0 ? (
                <Typography variant="caption" color="text.secondary">
                  No extra listeners.
                </Typography>
              ) : (
                <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
                  {chips.map((c) => (
                    <Tooltip key={c.key} title={c.tooltip}>
                      <Chip
                        label={c.label}
                        size="small"
                        color={c.color}
                        variant="outlined"
                        data-testid={`listener-${c.key}`}
                        sx={{ fontFamily: 'monospace' }}
                      />
                    </Tooltip>
                  ))}
                </Stack>
              )}
            </Box>

            <Box>
              <Stack direction="row" spacing={1} sx={{ alignItems: 'center', mb: 0.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  Access token
                </Typography>
                <Chip
                  label={net.token_set ? 'Set' : 'Not set'}
                  size="small"
                  color={net.token_set ? 'success' : 'default'}
                  variant="outlined"
                  data-testid="network-token-status"
                />
              </Stack>
              <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}>
                {tokenLock || (
                  <>
                    Remote MCP clients send it as <code>Authorization: Bearer &lt;token&gt;</code>; remote browsers sign in
                    once at <code>/login</code>. It is never shown again after it is saved.
                  </>
                )}
              </Typography>
              <Tooltip title={tokenLock} disableHoverListener={!tokenLock} disableFocusListener={!tokenLock}>
                <span>
                  <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
                    <Button size="small" variant="outlined" disabled={!!tokenLock || busy} onClick={() => void generate()}>
                      Generate
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      disabled={!!tokenLock || busy}
                      onClick={() => setEnteringToken((v) => !v)}
                    >
                      Set
                    </Button>
                    <Button
                      size="small"
                      color="error"
                      variant="outlined"
                      disabled={!!tokenLock || busy || !net.token_set}
                      onClick={() => void clearToken()}
                    >
                      Clear
                    </Button>
                  </Stack>
                </span>
              </Tooltip>
              {enteringToken && !tokenLock && (
                <Stack direction="row" spacing={1} sx={{ mt: 1.5, alignItems: 'flex-start' }}>
                  <TextField
                    label="New access token"
                    type="password"
                    size="small"
                    autoComplete="new-password"
                    value={tokenInput}
                    onChange={(e) => setTokenInput(e.target.value)}
                    sx={{ minWidth: 280 }}
                  />
                  <Button size="small" variant="contained" disabled={!tokenInput.trim() || busy} onClick={() => void saveEnteredToken()}>
                    Save
                  </Button>
                </Stack>
              )}
              {generated && (
                <Alert
                  severity="success"
                  sx={{ mt: 1.5 }}
                  onClose={() => setGenerated('')}
                  action={
                    <Button color="inherit" size="small" onClick={() => void copyGenerated()}>
                      Copy
                    </Button>
                  }
                >
                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    New token saved. Copy it now; it is not shown again.
                  </Typography>
                  <Typography
                    variant="body2"
                    data-testid="network-generated-token"
                    sx={{ fontFamily: 'monospace', wordBreak: 'break-all', userSelect: 'all' }}
                  >
                    {generated}
                  </Typography>
                </Alert>
              )}
            </Box>
          </Stack>
        )}
      </CardContent>
    </Card>
  )
}
