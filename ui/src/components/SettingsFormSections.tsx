import type { ReactNode } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  FormControl,
  FormControlLabel,
  FormHelperText,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material'

import { api } from '../api/client'
import { auxUnavailable } from '../lib/auxEmbedder'
import { OnnxRuntimeSection } from './OnnxRuntimeSection'
import type { SettingsData } from '../api/types'
import { useToast } from '../context/ToastContext'
import { useSettingsDraft, type SettingsDraft } from '../hooks/useSettingsDraft'
import { SECRET_SET } from '../lib/settingsDraft'
import {
  draftField,
  EMBED_SECRET_KEYS,
  embeddingSaved,
  performanceSaved,
  retentionSaved,
  settingsCardSx,
  virtualContextSaved,
  watcherSaved,
} from '../lib/settingsSections'
import { SettingsSaveBar } from './SettingsSaveBar'

const EMBED_BACKENDS = ['onnx', 'ollama', 'http', 'openai', 'docker']

/** A settings card around a draft: amber border while dirty, Save / Discard at the bottom. */
const DraftCard = ({ id, title, draft, children }: { id: string; title: string; draft: SettingsDraft; children: ReactNode }) => (
  <Card variant="outlined" id={`settings-${id}`} sx={settingsCardSx(draft.dirty)}>
    <CardContent>
      <Typography variant="subtitle1" gutterBottom>
        {title}
      </Typography>
      {children}
      <SettingsSaveBar draft={draft} label={title} />
    </CardContent>
  </Card>
)

/** A select bound to one draft key, with the key's error under it. */
const DraftSelect = ({ draft, k, label, options, minWidth = 160 }: { draft: SettingsDraft; k: string; label: string; options: [string, string][]; minWidth?: number }) => (
  <FormControl size="small" sx={{ minWidth }} error={!!draft.errors[k]}>
    <InputLabel id={`${k}-label`}>{label}</InputLabel>
    <Select labelId={`${k}-label`} label={label} value={draft.value(k)} onChange={(e) => draft.set(k, String(e.target.value))}>
      {options.map(([value, text]) => (
        <MenuItem key={value} value={value}>
          {text}
        </MenuItem>
      ))}
    </Select>
    {draft.errors[k] && <FormHelperText>{draft.errors[k]}</FormHelperText>}
  </FormControl>
)

/** An on/off setting stored as "true" / "false". */
const DraftSwitch = ({ draft, k, label }: { draft: SettingsDraft; k: string; label: string }) => (
  <>
    <FormControlLabel
      control={<Switch size="small" checked={draft.value(k) === 'true'} onChange={(e) => draft.set(k, String(e.target.checked))} />}
      label={label}
    />
    {draft.errors[k] && <FormHelperText error>{draft.errors[k]}</FormHelperText>}
  </>
)

/**
 * A write-only secret: the field starts blank and blank means unchanged. When one is stored,
 * Clear marks it for removal on Save.
 */
const SecretField = ({ draft, k, label, saved }: { draft: SettingsDraft; k: string; label: string; saved: Record<string, string> }) => {
  const cleared = saved[k] === SECRET_SET && !draft.secretSet(k) && draft.value(k) === ''
  const helper = cleared ? 'Removed when you save' : draft.secretSet(k) ? 'Saved. Leave blank to keep it, or type a new one.' : 'Not set'
  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: 'flex-start' }}>
      <TextField
        label={label}
        size="small"
        fullWidth
        type="password"
        autoComplete="new-password"
        placeholder={draft.secretSet(k) ? '••••••••' : ''}
        {...draftField(draft, k, helper)}
      />
      {draft.secretSet(k) && (
        <Button size="small" color="warning" onClick={() => draft.clearSecret(k)} sx={{ mt: 0.5 }}>
          Clear
        </Button>
      )}
    </Stack>
  )
}

export const PerformanceSection = ({ data }: { data: SettingsData }) => {
  const draft = useSettingsDraft('performance', 'Performance', performanceSaved(data))
  return (
    <DraftCard id="performance" title="Performance" draft={draft}>
      <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }} useFlexGap>
        <TextField label="Idle unload (minutes)" type="number" size="small" {...draftField(draft, 'idle_unload_minutes')} />
        <TextField
          label="Max embed workers"
          type="number"
          size="small"
          slotProps={{ htmlInput: { min: 1, max: 64 } }}
          {...draftField(draft, 'embed_worker_max', 'Primary pool +/− cap')}
        />
        {!auxUnavailable(data) && (
          <>
            <TextField
              label="Max aux embed workers"
              type="number"
              size="small"
              slotProps={{ htmlInput: { min: 1, max: 32 } }}
              {...draftField(draft, 'embed_aux_worker_max', 'Aux catch-up pool cap')}
            />
            <TextField
              label="Aux embed workers"
              type="number"
              size="small"
              slotProps={{ htmlInput: { min: 0, max: Number(draft.value('embed_aux_worker_max')) || 10 } }}
              {...draftField(draft, 'EMBED_AUX_WORKERS', 'Also adjustable on Overview')}
            />
            <DraftSelect
              draft={draft}
              k="EMBED_AUX_BACKEND"
              label="Aux embed backend"
              minWidth={180}
              options={[
                ['onnx', 'onnx (local)'],
                ['http', 'http'],
                ['ollama', 'ollama'],
                ['openai', 'openai / litellm'],
                ['docker', 'docker (DMR)'],
              ]}
            />
          </>
        )}
        {data.EmbedProbeIntervalSec != null && (
          <TextField
            label="Probe interval (sec)"
            type="number"
            size="small"
            slotProps={{ htmlInput: { min: 5, max: 600 } }}
            {...draftField(draft, 'embed_probe_interval_seconds')}
          />
        )}
      </Stack>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
        Aux controls appear on the embeddings card when the aux backend differs from the primary backend.
      </Typography>
    </DraftCard>
  )
}

export const VirtualContextSection = ({ data }: { data: SettingsData }) => {
  const { showToast } = useToast()
  const draft = useSettingsDraft('virtual', 'Virtual context', virtualContextSaved(data))
  return (
    <DraftCard id="virtual" title="Virtual context" draft={draft}>
      <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }} useFlexGap>
        <TextField label="Max notes / session" type="number" size="small" {...draftField(draft, 'context_max_notes_session')} />
        <TextField label="Max tokens / session" type="number" size="small" {...draftField(draft, 'context_max_tokens_session')} />
        <TextField label="Max notes global" type="number" size="small" {...draftField(draft, 'context_max_notes_global')} />
        <TextField label="Max tokens global" type="number" size="small" {...draftField(draft, 'context_max_tokens_global')} />
      </Stack>
      <Box sx={{ mt: 2, p: 2, border: '1px solid', borderColor: 'error.dark', borderRadius: 1 }}>
        <Typography variant="subtitle2" color="error">
          Danger zone
        </Typography>
        <Button
          color="error"
          variant="outlined"
          size="small"
          sx={{ mt: 1 }}
          onClick={async () => {
            if (!confirm('Flush ALL virtual context across every session and project? This is irreversible.')) return
            try {
              await api.flushContextAll()
              showToast('Flushed all virtual context', 'success')
            } catch (e) {
              showToast(String(e), 'error')
            }
          }}
        >
          Flush all virtual context
        </Button>
      </Box>
    </DraftCard>
  )
}

/**
 * Embedding backend: the backend and its URL / credentials save together, so the embedder
 * reloads once with a complete configuration instead of after each field.
 */
export const EmbeddingSection = ({ data, onRefresh }: { data: SettingsData; onRefresh?: () => void }) => {
  const { showToast } = useToast()
  const saved = embeddingSaved(data)
  const draft = useSettingsDraft('embedding', 'Embedding backend', saved, { secrets: EMBED_SECRET_KEYS })
  return (
    <DraftCard id="embedding" title="Embedding backend" draft={draft}>
      <Chip label={`Active: ${data.EmbedActiveBackend} / ${data.EmbedActiveModel}`} size="small" sx={{ mb: 2 }} />
      <OnnxRuntimeSection data={data} onRefresh={onRefresh ?? (() => {})} />
      <Stack spacing={2}>
        <DraftSelect draft={draft} k="EMBED_BACKEND" label="Backend" options={EMBED_BACKENDS.map((b) => [b, b])} />
        <TextField label="HTTP URL" size="small" fullWidth {...draftField(draft, 'EMBED_HTTP_URL')} />
        <SecretField draft={draft} k="EMBED_HTTP_BEARER" label="HTTP bearer" saved={saved} />
        <SecretField draft={draft} k="EMBED_OPENAI_API_KEY" label="OpenAI API key" saved={saved} />
        <Box>
          <Button
            variant="outlined"
            size="small"
            disabled={draft.dirty}
            onClick={async () => {
              try {
                await api.embedderTest()
                showToast('Embedder test OK', 'success')
              } catch (e) {
                showToast(String(e), 'error')
              }
            }}
          >
            Test embedder
          </Button>
          {draft.dirty && (
            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
              Save first: the test uses the saved settings.
            </Typography>
          )}
        </Box>
      </Stack>
    </DraftCard>
  )
}

export const WatcherSection = ({ data }: { data: SettingsData }) => {
  const draft = useSettingsDraft('watcher', 'File watcher & indexing', watcherSaved(data))
  return (
    <DraftCard id="watcher" title="File watcher & indexing" draft={draft}>
      <TextField label="Watcher ignore globs (JSON)" multiline minRows={2} fullWidth size="small" sx={{ mb: 1 }} {...draftField(draft, 'watcher_ignore_globs')} />
      <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}>
        Applies to every project. Each repo's .gitignore (nested), .stignore and .astignore are also honored, and
        per-project excludes live under Projects → Excludes. Excluded directories are pruned, not walked.
      </Typography>
      <DraftSwitch draft={draft} k="index_log_files" label="Index .log files" />
    </DraftCard>
  )
}

export const RetentionSection = ({ data }: { data: SettingsData }) => {
  const draft = useSettingsDraft('retention', 'Retention', retentionSaved(data))
  return (
    <DraftCard id="retention" title="Retention" draft={draft}>
      <TextField label="Query retention max age (days)" type="number" size="small" {...draftField(draft, 'query_retention_max_age_days')} />
    </DraftCard>
  )
}

export { DraftSwitch }

