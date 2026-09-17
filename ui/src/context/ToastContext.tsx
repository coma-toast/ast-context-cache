import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Alert, Snackbar } from '@mui/material'

type ToastSeverity = 'success' | 'error' | 'info'

interface QueuedToast {
  key: number
  message: string
  severity: ToastSeverity
}

interface ToastContextValue {
  showToast: (message: string, severity?: ToastSeverity) => void
}

const ToastContext = createContext<ToastContextValue>({ showToast: () => {} })

let nextToastKey = 0

// A single toast slot used to overwrite whatever was showing — App.tsx fires
// one showToast per failed API call in a batch, so 2+ simultaneous failures
// meant only the last was ever seen. Now queues them and shows one at a time,
// matching MUI's own "consecutive snackbars" pattern.
export function ToastProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<QueuedToast[]>([])
  const [current, setCurrent] = useState<QueuedToast | null>(null)
  const [open, setOpen] = useState(false)

  const showToast = useCallback((message: string, severity: ToastSeverity = 'info') => {
    setQueue((q) => [...q, { key: nextToastKey++, message, severity }])
  }, [])

  useEffect(() => {
    if (!open && queue.length > 0) {
      setCurrent(queue[0])
      setQueue((q) => q.slice(1))
      setOpen(true)
    }
  }, [open, queue])

  const handleClose = (_e?: unknown, reason?: string) => {
    if (reason === 'clickaway') return
    setOpen(false)
  }

  const value = useMemo(() => ({ showToast }), [showToast])
  return (
    <ToastContext.Provider value={value}>
      {children}
      {current && (
        <Snackbar
          key={current.key}
          open={open}
          autoHideDuration={4000}
          onClose={handleClose}
          slotProps={{ transition: { onExited: () => setCurrent(null) } }}
        >
          <Alert severity={current.severity} variant="filled" onClose={() => handleClose()}>
            {current.message}
          </Alert>
        </Snackbar>
      )}
    </ToastContext.Provider>
  )
}

export function useToast() {
  return useContext(ToastContext)
}
