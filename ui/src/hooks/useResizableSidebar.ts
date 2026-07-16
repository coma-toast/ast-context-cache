import { useCallback, useEffect, useRef, useState } from 'react'

const STORAGE_KEY = 'dashboard-sidebar-width'
const DEFAULT = 240
const MIN = 180
const MAX = 420

function readStored(): number {
  try {
    const n = parseInt(localStorage.getItem(STORAGE_KEY) || '', 10)
    if (Number.isFinite(n) && n >= MIN && n <= MAX) return n
  } catch {
    /* ignore */
  }
  return DEFAULT
}

export function useResizableSidebar() {
  const [width, setWidth] = useState(readStored)
  const dragging = useRef(false)
  const widthRef = useRef(width)
  widthRef.current = width

  const onPointerDown = useCallback((e: PointerEvent & { currentTarget: EventTarget & { setPointerCapture: (id: number) => void } }) => {
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    e.preventDefault()
  }, [])

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!dragging.current) return
      const next = Math.min(MAX, Math.max(MIN, e.clientX))
      setWidth(next)
    }
    const onUp = () => {
      if (!dragging.current) return
      dragging.current = false
      try {
        localStorage.setItem(STORAGE_KEY, String(widthRef.current))
      } catch {
        /* ignore */
      }
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [])

  return { width, onPointerDown, min: MIN, max: MAX }
}
