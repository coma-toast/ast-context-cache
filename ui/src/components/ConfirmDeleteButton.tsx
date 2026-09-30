import { Button, IconButton, Tooltip } from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import { useEffect, useState } from 'react'

const DISARM_AFTER_MS = 5000

/**
 * Two-step delete without a second mouse trip: the first activation swaps in a focused
 * "Confirm" button, so Enter or Space confirms right away. Esc, moving focus elsewhere,
 * or waiting DISARM_AFTER_MS cancels.
 */
export function ConfirmDeleteButton({
  label,
  onConfirm,
  disabled,
  tooltip = 'Delete',
  variant = 'icon',
}: {
  label: string
  onConfirm: () => void
  disabled?: boolean
  tooltip?: string
  variant?: 'icon' | 'text'
}) {
  const [armed, setArmed] = useState(false)

  useEffect(() => {
    if (!armed) return
    const id = window.setTimeout(() => setArmed(false), DISARM_AFTER_MS)
    return () => window.clearTimeout(id)
  }, [armed])

  useEffect(() => {
    if (disabled) setArmed(false)
  }, [disabled])

  if (armed) {
    return (
      <Tooltip title="Enter or Space to confirm · Esc to cancel" open placement="top" arrow>
        <Button
          size="small"
          color="error"
          variant="contained"
          disableElevation
          autoFocus
          aria-label={`Confirm delete ${label}`}
          onClick={() => {
            setArmed(false)
            onConfirm()
          }}
          onBlur={() => setArmed(false)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              e.stopPropagation()
              setArmed(false)
            }
          }}
          sx={{ minWidth: 0, px: 1, py: 0.125, fontSize: 12, lineHeight: 1.5, whiteSpace: 'nowrap' }}
        >
          Confirm
        </Button>
      </Tooltip>
    )
  }

  if (variant === 'text') {
    return (
      <Button size="small" color="error" disabled={disabled} aria-label={`Delete ${label}`} onClick={() => setArmed(true)}>
        Delete
      </Button>
    )
  }

  return (
    <Tooltip title={tooltip}>
      <span>
        <IconButton size="small" color="error" aria-label={`Delete ${label}`} disabled={disabled} onClick={() => setArmed(true)}>
          <DeleteIcon fontSize="inherit" />
        </IconButton>
      </span>
    </Tooltip>
  )
}
