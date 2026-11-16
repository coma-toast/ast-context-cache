import { useCallback, useState } from 'react'
import { Box, Chip, Stack, Tab, Tabs, Typography } from '@mui/material'
import { formatNum } from '../api/client'
import type { MemoryData, MemorySubTab } from '../api/types'
import { DocSourcesSubtab } from '../components/kb/DocSourcesSubtab'
import { KBDetailDrawer } from '../components/kb/KBDetailDrawer'
import { KBExplorerSubtab } from '../components/kb/KBExplorerSubtab'
import { KBPreferencesSubtab } from '../components/kb/KBPreferencesSubtab'
import { KBTimelineSubtab } from '../components/kb/KBTimelineSubtab'
import { VirtualContextSubtab } from '../components/kb/VirtualContextSubtab'

export interface MemoryTabProps {
  data: MemoryData | null
  onRefresh: () => void
  docSourcesPage: number
  onDocSourcesPageChange: (page: number) => void
  projectId?: string
  kbRefreshNonce?: number
  subTab?: MemorySubTab
  selectedRef?: string
  onSubTabChange?: (sub: MemorySubTab) => void
  onSelectRef?: (ref: string) => void
}

const SUB_TABS: { id: MemorySubTab; label: string }[] = [
  { id: 'explorer', label: 'Explorer' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'preferences', label: 'Preferences' },
  { id: 'virtual-context', label: 'Virtual context' },
  { id: 'doc-sources', label: 'Doc sources' },
]

export function MemoryTab({
  data,
  onRefresh,
  docSourcesPage,
  onDocSourcesPageChange,
  projectId,
  kbRefreshNonce = 0,
  subTab,
  selectedRef,
  onSubTabChange,
  onSelectRef,
}: MemoryTabProps) {
  const [internalSubTab, setInternalSubTab] = useState<MemorySubTab>('explorer')
  const [internalSelectedRef, setInternalSelectedRef] = useState('')
  const [localRefreshNonce, setLocalRefreshNonce] = useState(0)

  const activeSubTab = subTab ?? internalSubTab
  const activeRef = selectedRef !== undefined ? selectedRef : internalSelectedRef
  const effectiveRefreshNonce = kbRefreshNonce + localRefreshNonce

  const handleSubTabChange = useCallback(
    (next: MemorySubTab) => {
      if (subTab === undefined) {
        setInternalSubTab(next)
      }
      onSubTabChange?.(next)
    },
    [subTab, onSubTabChange],
  )

  const handleSelectRef = useCallback(
    (ref: string) => {
      const normalized = ref.trim()
      if (selectedRef === undefined) {
        setInternalSelectedRef(normalized)
      }
      onSelectRef?.(normalized)
    },
    [selectedRef, onSelectRef],
  )

  const handleKBMutated = useCallback(() => {
    setLocalRefreshNonce((n) => n + 1)
    onRefresh()
  }, [onRefresh])

  return (
    <Box data-testid="memory-tab">
      <Stack
        direction="row"
        spacing={1.5}
        useFlexGap
        sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', mb: 1.5 }}
      >
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
            Memory
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Knowledge Base hub
          </Typography>
        </Box>

        {data && (
          <Stack direction="row" spacing={0.75} useFlexGap sx={{ flexWrap: 'wrap', alignItems: 'center' }}>
            <Chip size="small" variant="outlined" label={`Facts: ${formatNum(data.ActiveFacts ?? 0)}`} />
            <Chip size="small" variant="outlined" label={`Preferences: ${formatNum(data.ActivePreferences ?? 0)}`} />
            <Chip size="small" variant="outlined" label={`Events: ${formatNum(data.ActiveEvents ?? 0)}`} />
            <Chip
              size="small"
              variant="outlined"
              label={`Config changes: ${formatNum(data.ActiveConfigChanges ?? 0)}`}
            />
            <Chip size="small" variant="outlined" label={`Discoveries: ${formatNum(data.ActiveDiscoveries ?? 0)}`} />
            <Chip
              size="small"
              variant={(data.QuarantinedCount ?? 0) > 0 ? 'filled' : 'outlined'}
              color={(data.QuarantinedCount ?? 0) > 0 ? 'warning' : 'default'}
              label={`Quarantined: ${formatNum(data.QuarantinedCount ?? 0)}`}
            />
          </Stack>
        )}
      </Stack>

      <Tabs
        value={activeSubTab}
        onChange={(_e, next) => handleSubTabChange(next as MemorySubTab)}
        variant="scrollable"
        scrollButtons="auto"
        aria-label="Memory sections"
        sx={{ mb: 2, borderBottom: '1px solid', borderColor: 'divider' }}
      >
        {SUB_TABS.map((t) => (
          <Tab key={t.id} value={t.id} label={t.label} />
        ))}
      </Tabs>

      {activeSubTab === 'explorer' && (
        <KBExplorerSubtab
          projectId={projectId}
          kbRefreshNonce={effectiveRefreshNonce}
          selectedRef={activeRef}
          onSelectRef={handleSelectRef}
        />
      )}

      {activeSubTab === 'timeline' && (
        <KBTimelineSubtab
          projectId={projectId}
          kbRefreshNonce={effectiveRefreshNonce}
          onSelectRef={handleSelectRef}
        />
      )}

      {activeSubTab === 'preferences' && (
        <KBPreferencesSubtab
          projectId={projectId}
          kbRefreshNonce={effectiveRefreshNonce}
          onSelectRef={handleSelectRef}
        />
      )}

      {activeSubTab === 'virtual-context' && (
        <VirtualContextSubtab
          data={data}
          onRefresh={onRefresh}
          projectId={projectId}
          kbRefreshNonce={effectiveRefreshNonce}
        />
      )}

      {activeSubTab === 'doc-sources' && (
        <DocSourcesSubtab
          data={data}
          onRefresh={onRefresh}
          docSourcesPage={docSourcesPage}
          onDocSourcesPageChange={onDocSourcesPageChange}
        />
      )}

      <KBDetailDrawer
        refId={activeRef}
        onClose={() => handleSelectRef('')}
        onSelectRef={handleSelectRef}
        onMutated={handleKBMutated}
      />
    </Box>
  )
}