import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material'
import FolderIcon from '@mui/icons-material/Folder'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import type { BrowseDirEntry } from '../api/types'
import { api } from '../api/client'

// A browser can't hand a web page a real filesystem path (native "choose folder"
// dialogs don't exist for web content, and <input type=file webkitdirectory> only
// exposes relative paths) — so instead of a native file picker, this browses the
// server's own filesystem over /api/browse-dir and returns the absolute path the
// backend actually sees, which is what the Move action needs.
export function DirectoryPicker({
  open,
  initialPath,
  onClose,
  onSelect,
}: {
  open: boolean
  initialPath?: string
  onClose: () => void
  onSelect: (path: string) => void
}) {
  const [path, setPath] = useState('')
  const [parent, setParent] = useState<string | undefined>(undefined)
  const [entries, setEntries] = useState<BrowseDirEntry[]>([])
  const [shortcuts, setShortcuts] = useState<BrowseDirEntry[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const load = async (target?: string) => {
    setLoading(true)
    try {
      const r = await api.browseDir(target)
      setPath(r.path)
      setParent(r.parent)
      setEntries(r.entries || [])
      setShortcuts(r.shortcuts || [])
      setError(r.error || '')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not list directory')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open) void load(initialPath)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Choose a directory</DialogTitle>
      <DialogContent dividers>
        <Stack direction="row" spacing={1} sx={{ mb: 1.5 }} flexWrap="wrap" useFlexGap>
          {shortcuts.map((s) => (
            <Chip key={s.path} label={s.name} size="small" variant="outlined" onClick={() => void load(s.path)} />
          ))}
        </Stack>
        <Typography variant="caption" color="text.secondary" component="div" sx={{ mb: 1, fontFamily: 'ui-monospace, monospace', wordBreak: 'break-all' }}>
          {path || '…'}
        </Typography>
        {error && (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            {error}
          </Alert>
        )}
        <Box sx={{ maxHeight: 320, overflowY: 'auto', border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
          <List dense disablePadding>
            {parent !== undefined && parent !== '' && (
              <ListItemButton onClick={() => void load(parent)}>
                <ArrowUpwardIcon fontSize="small" sx={{ mr: 1.5, opacity: 0.7 }} />
                <ListItemText primary=".." />
              </ListItemButton>
            )}
            {entries.map((e) => (
              <ListItemButton key={e.path} onClick={() => void load(e.path)}>
                <FolderIcon fontSize="small" sx={{ mr: 1.5, opacity: 0.7 }} />
                <ListItemText primary={e.name} />
              </ListItemButton>
            ))}
            {!loading && entries.length === 0 && parent === undefined && !error && (
              <Box sx={{ p: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  No subdirectories here.
                </Typography>
              </Box>
            )}
          </List>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={!path} onClick={() => path && onSelect(path)}>
          Select this folder
        </Button>
      </DialogActions>
    </Dialog>
  )
}
