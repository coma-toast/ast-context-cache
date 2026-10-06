import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Alert, Box, Grow, Typography } from '@mui/material'

import {
  emptyToastStack,
  MAX_HOLD_MS,
  pushToast,
  releaseGroup,
  removeToast,
  TOAST_DURATION_MS,
  type StackedToast,
  type ToastSeverity,
} from '../lib/toastStack'

interface ToastOptions {
  group?: string
  hold?: boolean
}

interface ToastContextValue {
  showToast: (message: string, severity?: ToastSeverity, options?: ToastOptions) => void
  releaseToastGroup: (group: string, delayMs: number) => void
}

const ToastContext = createContext<ToastContextValue>({ showToast: () => {}, releaseToastGroup: () => {} })

let nextToastKey = 0

// Stacks up to MAX_VISIBLE_TOASTS toasts, each on its own timer, and queues
// the rest. A held toast (an indexing toast while its project's queue is
// still busy) stays until its group is released or MAX_HOLD_MS passes.
export function ToastProvider({ children }: { children: ReactNode }) {
  const [stack, setStack] = useState(emptyToastStack)

  const showToast = useCallback((message: string, severity: ToastSeverity = 'info', options: ToastOptions = {}) => {
    const toast: StackedToast = { key: nextToastKey++, message, severity, group: options.group, held: !!options.hold, durationMs: TOAST_DURATION_MS }
    setStack((s) => pushToast(s, toast))
  }, [])
  const releaseToastGroup = useCallback((group: string, delayMs: number) => setStack((s) => releaseGroup(s, group, delayMs)), [])
  const close = useCallback((key: number) => setStack((s) => removeToast(s, key)), [])

  const value = useMemo(() => ({ showToast, releaseToastGroup }), [showToast, releaseToastGroup])
  return (
    <ToastContext.Provider value={value}>
      {children}
      {stack.visible.length > 0 && (
        <Box
          role="region"
          aria-label="Notifications"
          sx={{ position: 'fixed', zIndex: (t) => t.zIndex.snackbar, bottom: 24, left: { xs: 16, sm: 24 }, right: { xs: 16, sm: 'auto' }, display: 'flex', flexDirection: 'column', gap: 1, maxWidth: { sm: 480 } }}
        >
          {stack.queued.length > 0 && (
            <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'flex-start', px: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
              +{stack.queued.length} more
            </Typography>
          )}
          {stack.visible.map((t) => (
            <ToastItem key={t.key} toast={t} onClose={close} />
          ))}
        </Box>
      )}
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onClose }: { toast: StackedToast; onClose: (key: number) => void }) {
  const { key, held, durationMs } = toast
  useEffect(() => {
    const id = window.setTimeout(() => onClose(key), held ? MAX_HOLD_MS : durationMs)
    return () => window.clearTimeout(id)
  }, [key, held, durationMs, onClose])
  return (
    <Grow in appear>
      <Alert severity={toast.severity} variant="filled" onClose={() => onClose(key)} sx={{ boxShadow: 3 }}>
        {toast.message}
      </Alert>
    </Grow>
  )
}

export function useToast() {
  return useContext(ToastContext)
}
