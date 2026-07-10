import { useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import type { Project, SettingsData } from '../api/types'
import { api } from '../api/client'
import { useToast } from '../context/ToastContext'
import { formatNum } from '../api/client'

const SECTIONS = [
  { id: 'performance', label: 'Performance' },
  { id: 'virtual', label: 'Virtual context' },
  { id: 'embedding', label: 'Embedding' },
  { id: 'watcher', label: 'Watcher' },
  { id: 'retention', label: 'Retention' },
  { id: 'projects', label: 'Projects' },
  { id: 'agents', label: 'Agents' },
  { id: 'mcp', label: 'MCP tier' },
]

export function SettingsTab({
  data,
  onRefresh,
  mcpTier,
}: {
  data: SettingsData | null
  mcpTier: { tier: string; tools_json_path: string; tools_json_exists: boolean } | null
  onRefresh: () => void
}) {
  const { showToast } = useToast()
  const [linkChild, setLinkChild] = useState<Record<string, string>>({})

  if (!data) return <Typography color="text.secondary">Loading settings…</Typography>

  const save = async (key: string, value: string) => {
    try {
      await api.saveSetting(key, value)
      showToast('Setting saved', 'success')
      onRefresh()
    } catch (e) {
      showToast(String(e), 'error')
    }
  }

  const projects = data.Projects || []
  const linkable = (parent: Project) =>
    projects.filter((p) => p.Path !== parent.Path && p.LinkedParent === '' && !parent.LinkedChildren?.includes(p.Path))

  return (
    <Box>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{
        mb: 3,
        position: 'sticky',
        top: { xs: 56, md: 56 },
        zIndex: 10,
        py: 1,
        bgcolor: 'rgba(13,17,23,0.85)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}>
        {SECTIONS.map((s) => (
          <Chip key={s.id} label={s.label} component="a" href={`#settings-${s.id}`} clickable variant="outlined" size="small" />
        ))}
      </Stack>

      <Card variant="outlined" id="settings-performance" sx={{ mb: 2, scrollMarginTop: { xs: 120, md: 120 } }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Performance
          </Typography>
          <TextField
            label="Idle unload (minutes)"
            type="number"
            size="small"
            defaultValue={data.IdleUnloadMinutes}
            onBlur={(e) => save('idle_unload_minutes', e.target.value)}
            sx={{ mr: 2, mb: 1 }}
          />
        </CardContent>
      </Card>

      <Card variant="outlined" id="settings-virtual" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Virtual context
          </Typography>
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            <TextField label="Max notes / session" type="number" size="small" defaultValue={data.ContextMaxNotesSession} onBlur={(e) => save('context_max_notes_session', e.target.value)} />
            <TextField label="Max tokens / session" type="number" size="small" defaultValue={data.ContextMaxTokensSession} onBlur={(e) => save('context_max_tokens_session', e.target.value)} />
            <TextField label="Max notes global" type="number" size="small" defaultValue={data.ContextMaxNotesGlobal} onBlur={(e) => save('context_max_notes_global', e.target.value)} />
            <TextField label="Max tokens global" type="number" size="small" defaultValue={data.ContextMaxTokensGlobal} onBlur={(e) => save('context_max_tokens_global', e.target.value)} />
          </Stack>
          <Box sx={{ mt: 2, p: 2, border: '1px solid', borderColor: 'error.dark', borderRadius: 1 }}>
            <Typography variant="subtitle2" color="error">
              Danger zone
            </Typography>
            <Button color="error" variant="outlined" size="small" sx={{ mt: 1 }} onClick={async () => {
              try {
                await api.flushContextAll()
                showToast('Flushed all virtual context', 'success')
              } catch (e) {
                showToast(String(e), 'error')
              }
            }}>
              Flush all virtual context
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" id="settings-embedding" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Embedding backend
          </Typography>
          <Chip label={`Active: ${data.EmbedActiveBackend} / ${data.EmbedActiveModel}`} size="small" sx={{ mb: 2 }} />
          <Stack spacing={2}>
            <FormControl size="small" sx={{ minWidth: 160 }}>
              <InputLabel>Backend</InputLabel>
              <Select label="Backend" defaultValue={data.EmbedBackend || 'onnx'} onChange={(e) => save('EMBED_BACKEND', e.target.value)}>
                {['onnx', 'ollama', 'http', 'openai', 'docker'].map((b) => (
                  <MenuItem key={b} value={b}>
                    {b}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField label="HTTP URL" size="small" fullWidth defaultValue={data.EmbedHTTPURL} onBlur={(e) => save('EMBED_HTTP_URL', e.target.value)} />
            <TextField label="HTTP bearer" size="small" fullWidth type="password" defaultValue={data.EmbedHTTPBearer} onBlur={(e) => save('EMBED_HTTP_BEARER', e.target.value)} />
            <TextField label="OpenAI API key" size="small" fullWidth type="password" defaultValue={data.EmbedOpenAIAPIKey} onBlur={(e) => save('EMBED_OPENAI_API_KEY', e.target.value)} />
            <Button variant="outlined" size="small" onClick={async () => {
              try {
                await api.embedderTest()
                showToast('Embedder test OK', 'success')
              } catch (e) {
                showToast(String(e), 'error')
              }
            }}>
              Test embedder
            </Button>
          </Stack>
        </CardContent>
      </Card>

      <Card variant="outlined" id="settings-watcher" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            File watcher & indexing
          </Typography>
          <TextField label="Watcher ignore globs (JSON)" multiline minRows={2} fullWidth size="small" defaultValue={data.WatcherIgnoreGlobs} onBlur={(e) => save('watcher_ignore_globs', e.target.value)} sx={{ mb: 1 }} />
          <Button size="small" onClick={() => save('index_log_files', data.IndexLogFiles ? 'false' : 'true')}>
            Index .log files: {data.IndexLogFiles ? 'On' : 'Off'}
          </Button>
        </CardContent>
      </Card>

      <Card variant="outlined" id="settings-retention" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Retention
          </Typography>
          <TextField label="Query retention max age (days)" type="number" size="small" defaultValue={data.QueryRetentionMaxAgeDays} onBlur={(e) => save('query_retention_max_age_days', e.target.value)} />
        </CardContent>
      </Card>

      <Card variant="outlined" id="settings-projects" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Projects
          </Typography>
          {data.ProjectsLoading && <Typography color="text.secondary">Loading…</Typography>}
          {projects.length === 0 && !data.ProjectsLoading && (
            <Typography color="text.secondary">No indexed projects — run index_files via MCP</Typography>
          )}
          <Stack spacing={2}>
            {projects.map((p) => (
              <Box key={p.Path} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1, minWidth: 0 }}>
                <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', md: 'flex-start' }} gap={2}>
                  <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Typography fontWeight={600}>{p.Label}</Typography>
                    <Typography
                      variant="caption"
                      fontFamily="monospace"
                      display="block"
                      sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={p.Path}
                    >
                      {p.Path}
                    </Typography>
                    {(p.SymbolCount > 0 || p.QueryCount > 0) && (
                      <Typography variant="caption" color="text.secondary">
                        {formatNum(p.SymbolCount)} symbols · {formatNum(p.FileCount)} files · {formatNum(p.QueryCount)} queries
                      </Typography>
                    )}
                    {p.LinkedParent && (
                      <Typography variant="caption" display="block">
                        Linked under: {p.LinkedParent}
                      </Typography>
                    )}
                    {p.LinkedChildren?.map((c) => (
                      <Stack key={c} direction="row" spacing={1} alignItems="center">
                        <Typography variant="caption" fontFamily="monospace">
                          {c}
                        </Typography>
                        <Button size="small" onClick={async () => {
                          try {
                            await api.unlinkProject(p.Path, c)
                            showToast('Unlinked', 'success')
                            onRefresh()
                          } catch (e) {
                            showToast(String(e), 'error')
                          }
                        }}>
                          Unlink
                        </Button>
                      </Stack>
                    ))}
                  </Box>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ flexShrink: 0 }}>
                    <Button size="small" onClick={async () => {
                      try {
                        await api.pinProject(p.Path, !p.Pinned)
                        showToast(p.Pinned ? 'Unpinned' : 'Pinned', 'success')
                        onRefresh()
                      } catch (e) {
                        showToast(String(e), 'error')
                      }
                    }}>
                      {p.Pinned ? 'Unpin' : 'Pin'}
                    </Button>
                    <Button size="small" color="warning" onClick={async () => {
                      try {
                        await api.resetProject(p.Path)
                        showToast('Reset', 'success')
                        onRefresh()
                      } catch (e) {
                        showToast(String(e), 'error')
                      }
                    }}>
                      Reset
                    </Button>
                    <Button size="small" color="error" onClick={async () => {
                      if (!confirm(`Delete ${p.Label}?`)) return
                      try {
                        await api.deleteWatcher(p.Path)
                        showToast('Deleted', 'success')
                        onRefresh()
                      } catch (e) {
                        showToast(String(e), 'error')
                      }
                    }}>
                      Delete
                    </Button>
                  </Stack>
                </Stack>
                {!p.LinkedParent && linkable(p).length > 0 && (
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
                    <FormControl size="small" sx={{ minWidth: 200 }}>
                      <InputLabel>Link subproject</InputLabel>
                      <Select
                        label="Link subproject"
                        value={linkChild[p.Path] || ''}
                        onChange={(e) => setLinkChild({ ...linkChild, [p.Path]: e.target.value })}
                      >
                        {linkable(p).map((c) => (
                          <MenuItem key={c.Path} value={c.Path}>
                            {c.Label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Button
                      size="small"
                      disabled={!linkChild[p.Path]}
                      onClick={async () => {
                        try {
                          await api.linkProject(p.Path, linkChild[p.Path])
                          showToast('Linked', 'success')
                          setLinkChild({ ...linkChild, [p.Path]: '' })
                          onRefresh()
                        } catch (e) {
                          showToast(String(e), 'error')
                        }
                      }}
                    >
                      Link
                    </Button>
                    <Typography variant="caption" color="text.secondary">
                      Auto-link on index; search fans out to children
                    </Typography>
                  </Stack>
                )}
              </Box>
            ))}
          </Stack>
        </CardContent>
      </Card>

      <Card variant="outlined" id="settings-agents" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Agent integration
          </Typography>
          <Stack spacing={2}>
            {data.Agents?.map((a) => (
              <Box key={a.Type}>
                <Typography fontWeight={500}>{a.Name}</Typography>
                <Typography variant="caption" color="text.secondary" display="block">
                  {a.Description}
                </Typography>
                <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                  <Button size="small" variant={a.GlobalInstalled ? 'contained' : 'outlined'} onClick={async () => {
                    try {
                      if (a.GlobalInstalled) await api.agentUninstall(a.Type, true)
                      else await api.agentInstall(a.Type, true)
                      showToast('Updated agent config', 'success')
                      onRefresh()
                    } catch (e) {
                      showToast(String(e), 'error')
                    }
                  }}>
                    Global {a.GlobalInstalled ? '✓' : 'Install'}
                  </Button>
                  <Button size="small" variant={a.ProjectInstalled ? 'contained' : 'outlined'} onClick={async () => {
                    try {
                      if (a.ProjectInstalled) await api.agentUninstall(a.Type, false)
                      else await api.agentInstall(a.Type, false)
                      showToast('Updated agent config', 'success')
                      onRefresh()
                    } catch (e) {
                      showToast(String(e), 'error')
                    }
                  }}>
                    Project {a.ProjectInstalled ? '✓' : 'Install'}
                  </Button>
                </Stack>
              </Box>
            ))}
          </Stack>
        </CardContent>
      </Card>

      {mcpTier && (
        <Card variant="outlined" id="settings-mcp">
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              MCP tool tier
            </Typography>
            <Typography variant="body2">Effective tier: {mcpTier.tier}</Typography>
            <Typography variant="caption" fontFamily="monospace" display="block">
              {mcpTier.tools_json_path} {mcpTier.tools_json_exists ? '(exists)' : '(missing)'}
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  )
}
