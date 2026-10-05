import { useState } from 'react'

import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'

import DeleteSweepIcon from '@mui/icons-material/DeleteSweep'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'

import type { HandoffTreeView, HandoffTreesResponse } from '../api/types'

import { useToast } from '../context/ToastContext'

import { api } from '../api/client'
import { chartColors } from '../lib/chartColors'
import {
  buildHandoffNodes,
  claimsLabel,
  formatAge,
  formatRepeatRate,
  HANDOFF_STATUSES,
  sortTrees,
  STATUS_COLOR,
  statusCounts,
  type ChildNode,
  type HandoffNode,
} from '../lib/handoffTree'
import { formatStat } from './HealthBar'

const MONO = '"JetBrains Mono", ui-monospace, monospace'
const INDENT = 2.5

interface HandoffTreesCardProps {
  data: HandoffTreesResponse | null
  /** Called after a flush so the parent reloads the trees. */
  onChanged?: () => void
}

/**
 * Handoff trees (OB-4): each tree collapses to a header row with its status counts, usage, and
 * claims; expanded, its handoffs list their children with tokens delivered and saved, repeat
 * rate, claims, and last activity, and a child's own handoffs nest beneath it.
 */
export const HandoffTreesCard = ({ data, onChanged }: HandoffTreesCardProps) => {
  const { showToast } = useToast()
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [flushTarget, setFlushTarget] = useState<HandoffTreeView | null>(null)
  const [flushing, setFlushing] = useState(false)

  const flush = async (tree: HandoffTreeView) => {
    setFlushing(true)
    try {
      const res = await api.flushHandoffTree(tree.tree_id)
      const { handoffs, children } = res.flushed
      showToast(`Flushed tree: ${handoffs} handoff${handoffs === 1 ? '' : 's'}, ${children} child session${children === 1 ? '' : 's'}`, 'success')
      onChanged?.()
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Flush failed', 'error')
    } finally {
      setFlushing(false)
      setFlushTarget(null)
    }
  }

  const trees = data ? sortTrees(data.trees) : []
  return (
    <Card variant="outlined" data-testid="handoff-trees-card">
      <CardContent sx={{ pb: '16px !important' }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', mb: 1, flexWrap: 'wrap' }} useFlexGap>
          <Typography variant="subtitle2">Handoff trees</Typography>
          {data && (
            <Typography variant="caption" color="text.secondary">
              {trees.length} tree{trees.length === 1 ? '' : 's'} · 24h child repeat-search rate{' '}
              {Math.round(data.repeat_search_ratio_24h * 100)}%
            </Typography>
          )}
        </Stack>
        {!data && (
          <Typography variant="body2" color="text.secondary">
            Loading handoff trees…
          </Typography>
        )}
        {data && trees.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No handoff trees. A tree appears when an agent hands work to a subagent with the handoff tool.
          </Typography>
        )}
        {trees.length > 0 && (
          <List dense disablePadding>
            {trees.map((tree, i) => {
              const open = expanded[tree.tree_id] ?? i === 0
              return (
                <TreeRow
                  key={tree.tree_id}
                  tree={tree}
                  open={open}
                  onToggle={() => setExpanded((prev) => ({ ...prev, [tree.tree_id]: !open }))}
                  onFlush={() => setFlushTarget(tree)}
                  flushDisabled={flushing}
                />
              )
            })}
          </List>
        )}
      </CardContent>
      <Dialog open={flushTarget !== null} onClose={() => !flushing && setFlushTarget(null)} aria-labelledby="flush-tree-title">
        <DialogTitle id="flush-tree-title">Flush handoff tree?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This deletes tree <Box component="span" sx={{ fontFamily: MONO }}>{flushTarget?.tree_id}</Box> of{' '}
            <Box component="span" sx={{ fontFamily: MONO }}>{flushTarget?.root_session_id}</Box>: its{' '}
            {flushTarget?.handoffs.length ?? 0} handoff{flushTarget?.handoffs.length === 1 ? '' : 's'}, every child session's
            notes and results, the scratchpad, and claims. Memory promoted to the parent session stays. This can't be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFlushTarget(null)} disabled={flushing}>
            Cancel
          </Button>
          <Button color="error" variant="contained" disabled={flushing} onClick={() => flushTarget && void flush(flushTarget)}>
            {flushing ? 'Flushing…' : 'Flush tree'}
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  )
}

interface TreeRowProps {
  tree: HandoffTreeView
  open: boolean
  onToggle: () => void
  onFlush: () => void
  flushDisabled: boolean
}

const TreeRow = ({ tree, open, onToggle, onFlush, flushDisabled }: TreeRowProps) => {
  const counts = statusCounts(tree)
  const project = tree.project_path?.split('/').filter(Boolean).pop() ?? ''
  const claims = claimsLabel(tree.active_claims, tree.queued_claims)
  const usagePct = tree.tokens_max > 0 ? Math.round((tree.tokens_used / tree.tokens_max) * 100) : 0
  return (
    <Box component="li" sx={{ listStyle: 'none', borderTop: '1px solid', borderColor: 'divider' }} data-testid={`handoff-tree-${tree.tree_id}`}>
      <Stack direction="row" sx={{ alignItems: 'center' }}>
        <ListItemButton onClick={onToggle} aria-expanded={open} sx={{ px: 0.5, py: 0.75, minWidth: 0, flex: 1, gap: 1 }}>
          {open ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
          <Box sx={{ minWidth: 0, flex: 1 }}>
            <Typography noWrap sx={{ fontFamily: MONO, fontSize: 12.5 }} title={[tree.tree_id, tree.root_session_id, tree.project_path].filter(Boolean).join('\n')}>
              {tree.root_session_id}
              {project && (
                <Typography component="span" color="text.secondary" sx={{ fontFamily: 'inherit', fontSize: 'inherit', ml: 1 }}>
                  {project}
                </Typography>
              )}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }} noWrap>
              {formatStat(tree.tokens_used)} / {formatStat(tree.tokens_max)} tokens ({usagePct}%) · {tree.entries_used} / {tree.entries_max}{' '}
              entries · delivered {formatStat(tree.tokens_delivered)} · saved {formatStat(tree.tokens_saved)} · repeat{' '}
              {formatRepeatRate(tree.repeat_rate, tree.search_calls)} · active {formatAge(tree.last_access_at)} ago
            </Typography>
          </Box>
          <Stack direction="row" spacing={0.5} sx={{ flexShrink: 0, alignItems: 'center' }}>
            {HANDOFF_STATUSES.filter((s) => counts[s] > 0).map((s) => (
              <Chip key={s} size="small" color={STATUS_COLOR[s]} variant="outlined" label={`${counts[s]} ${s}`} sx={{ height: 20, fontSize: 11 }} />
            ))}
            {claims && <Chip size="small" variant="outlined" label={claims} sx={{ height: 20, fontSize: 11, borderColor: chartColors.orange, color: chartColors.orange }} />}
            {tree.expired && <Chip size="small" label="expired" sx={{ height: 20, fontSize: 11 }} />}
          </Stack>
        </ListItemButton>
        <Tooltip title="Flush this tree">
          <span>
            <IconButton size="small" color="error" aria-label={`Flush handoff tree ${tree.tree_id}`} disabled={flushDisabled} onClick={onFlush}>
              <DeleteSweepIcon fontSize="inherit" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
      <Collapse in={open} timeout="auto" unmountOnExit>
        <List dense disablePadding sx={{ pb: 1 }}>
          {buildHandoffNodes(tree).map((node) => (
            <HandoffNodeView key={node.handoff.handoff} node={node} level={1} />
          ))}
        </List>
      </Collapse>
    </Box>
  )
}

const HandoffNodeView = ({ node, level }: { node: HandoffNode; level: number }) => {
  const { handoff } = node
  return (
    <>
      <ListItem disableGutters sx={{ pl: level * INDENT, py: 0.25, gap: 1 }}>
        <Typography noWrap sx={{ fontSize: 12.5, fontWeight: 600, minWidth: 0, flex: 1 }} title={handoff.handoff}>
          {handoff.label || handoff.handoff}
          <Typography component="span" color="text.secondary" sx={{ fontFamily: MONO, fontSize: 11, fontWeight: 400, ml: 1 }}>
            {handoff.handoff}
          </Typography>
        </Typography>
        <Chip size="small" variant="outlined" label={handoff.mode} sx={{ height: 18, fontSize: 10.5 }} />
        <Chip size="small" variant="outlined" label={`depth ${handoff.depth}`} sx={{ height: 18, fontSize: 10.5 }} />
      </ListItem>
      {node.children.length === 0 && (
        <ListItem disableGutters sx={{ pl: (level + 1) * INDENT, py: 0 }}>
          <Typography variant="caption" color="text.secondary">
            Not opened yet
          </Typography>
        </ListItem>
      )}
      {node.children.map((c) => (
        <ChildNodeView key={c.child.session_id} node={c} level={level + 1} />
      ))}
    </>
  )
}

const ChildNodeView = ({ node, level }: { node: ChildNode; level: number }) => {
  const { child } = node
  const claims = claimsLabel(child.active_claims, child.queued_claims)
  return (
    <>
      <ListItem disableGutters sx={{ pl: level * INDENT, py: 0.25, alignItems: 'flex-start', gap: 1 }} data-testid={`handoff-child-${child.session_id}`}>
        <Chip size="small" color={STATUS_COLOR[child.status]} label={child.status} sx={{ height: 18, fontSize: 10.5, mt: 0.25, width: 80 }} />
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography noWrap sx={{ fontFamily: MONO, fontSize: 12 }} title={child.session_id}>
            {child.session_id}
            {child.label && (
              <Typography component="span" color="text.secondary" sx={{ fontFamily: 'inherit', fontSize: 'inherit', ml: 1 }}>
                {child.label}
              </Typography>
            )}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }} noWrap>
            delivered {formatStat(child.tokens_delivered)} / {formatStat(child.tokens_available)} · saved{' '}
            <Box component="span" sx={{ color: chartColors.green }}>
              {formatStat(child.tokens_saved)}
            </Box>{' '}
            · repeat {formatRepeatRate(child.repeat_rate, child.search_calls)} ({child.repeat_calls}/{child.search_calls}) · active{' '}
            {formatAge(child.last_activity_at)} ago
          </Typography>
          {child.summary && (
            <Typography variant="caption" sx={{ display: 'block' }} noWrap title={[child.result_ref, child.summary].filter(Boolean).join('\n')}>
              {child.summary}
            </Typography>
          )}
        </Box>
        {claims && <Chip size="small" variant="outlined" label={claims} sx={{ height: 18, fontSize: 10.5, mt: 0.25, borderColor: chartColors.orange, color: chartColors.orange }} />}
      </ListItem>
      {node.handoffs.map((h) => (
        <HandoffNodeView key={h.handoff.handoff} node={h} level={level + 1} />
      ))}
    </>
  )
}
