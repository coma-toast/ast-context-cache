import type { ReactNode } from 'react'

export function StoryFrame({ children }: { children: ReactNode }) {
  return (
    <div
      className="story-frame"
      data-testid="story-frame"
      style={{ width: 1280, minHeight: 400, background: '#0d1117', padding: 16, boxSizing: 'border-box' }}
    >
      {children}
    </div>
  )
}
