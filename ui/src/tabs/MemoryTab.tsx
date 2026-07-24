import { useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  IconButton,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import RefreshIcon from '@mui/icons-material/Refresh'
import type { MemoryData } from '../api/types'
import { api, formatNum } from '../api/client'
import { useToast } from '../context/ToastContext'
import { StatCard } from '../components/HealthBar'

export function MemoryTab({ data, onRefresh }: { data: MemoryData | null; onRefresh: () => void }) {
  const { showToast } = useToast()
  const [docName, setDocName] = useState('')
  const [docUrl, setDocUrl] = useState('')
  const [docType, setDocType] = useState('markdown')
  const [packBusy, setPackBusy] = useState(false)

  if (!data) return <Typography color="text.secondary">Loading memory…</Typography>

  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Virtual context
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="Inventory tokens" value={formatNum(data.VirtualInventoryTokens)} sub={`${data.VirtualNotesCount} notes`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="30d utilization" value={`${data.VirtualUtilPct30d?.toFixed(0) || 0}%`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="Orphans" value={formatNum(data.VirtualOrphanCount)} />
        </Grid>
      </Grid>

      <Typography variant="overline" color="text.secondary">
        Knowledge base (doc sources)
      </Typography>
      <Card variant="outlined" sx={{ mt: 1, mb: 2 }}>
        <CardContent>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
            <TextField size="small" label="Name" value={docName} onChange={(e) => setDocName(e.target.value)} />
            <TextField size="small" label="URL" value={docUrl} onChange={(e) => setDocUrl(e.target.value)} sx={{ minWidth: 240 }} />
            <TextField size="small" label="Type" value={docType} onChange={(e) => setDocType(e.target.value)} />
            <Button
              variant="contained"
              size="small"
              onClick={async () => {
                try {
                  await api.addDocSource(docName, docUrl, docType)
                  showToast('Doc source added', 'success')
                  setDocName('')
                  setDocUrl('')
                  onRefresh()
                } catch (e) {
                  showToast(String(e), 'error')
                }
              }}
            >
              Add source
            </Button>
            <Button
              variant="outlined"
              size="small"
              disabled={packBusy}
              onClick={async () => {
                setPackBusy(true)
                try {
                  const r = await api.installDocPack()
                  showToast(`Starter pack: ${r.added ?? 0} sources queued`, 'success')
                  onRefresh()
                } catch (e) {
                  showToast(String(e), 'error')
                } finally {
                  setPackBusy(false)
                }
              }}
            >
              Add starter doc pack
            </Button>
          </Stack>
          {data.DocSources?.length === 0 ? (
            <Typography color="text.secondary">No doc sources — add one above or use Add starter doc pack</Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Age</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.DocSources?.map((d) => (
                  <TableRow key={d.ID}>
                    <TableCell>{d.Name}</TableCell>
                    <TableCell>{d.Type}</TableCell>
                    <TableCell>{d.Age}{d.Stale ? ' (stale)' : ''}</TableCell>
                    <TableCell align="right">
                      <IconButton size="small" aria-label="Refresh" onClick={async () => {
                        try {
                          await api.docSourceAction('refresh', d.ID)
                          showToast('Refreshing', 'info')
                          onRefresh()
                        } catch (e) {
                          showToast(String(e), 'error')
                        }
                      }}>
                        <RefreshIcon fontSize="small" />
                      </IconButton>
                      <IconButton size="small" aria-label="Delete" onClick={async () => {
                        try {
                          await api.docSourceAction('delete', d.ID)
                          showToast('Deleted', 'success')
                          onRefresh()
                        } catch (e) {
                          showToast(String(e), 'error')
                        }
                      }}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}
