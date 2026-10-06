import { useEffect } from 'react'
import type { Meta, StoryObj } from '@storybook/react-vite'

import { StoryFrame } from '../storybook/StoryFrame'
import { ToastProvider, useToast } from '../context/ToastContext'

const meta: Meta = {
  title: 'Dashboard/Toasts',
}

export default meta

type Story = StoryObj

function Burst({ count }: { count: number }) {
  const { showToast } = useToast()
  useEffect(() => {
    // Held toasts so the stack stays put for screenshots; live ones time out.
    for (let i = 0; i < count; i++) showToast(`Indexing · ast-context-cache: reindex · file_${i}.go`, 'info', { group: `story-${i}`, hold: true })
    showToast('Setting saved', 'success', { group: 'story-saved', hold: true })
  }, [count, showToast])
  return null
}

/** A burst of watcher toasts: ten stack at once and the rest wait behind a "+N more" count. */
export const IndexingBurst: Story = {
  render: () => (
    <StoryFrame>
      <ToastProvider>
        <Burst count={12} />
      </ToastProvider>
    </StoryFrame>
  ),
}
