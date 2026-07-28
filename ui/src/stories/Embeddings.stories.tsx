import type { Meta, StoryObj } from '@storybook/react-vite'
import { Box } from '@mui/material'
import { StoryFrame } from '../storybook/StoryFrame'
import { EmbeddingsPanel } from '../components/EmbeddingsPanel'
import { fixtureIndexHealth, fixtureIndexHealthDegraded } from '../storybook/fixtures'

const meta: Meta = {
  title: 'Dashboard/Embeddings',
}

export default meta

type Story = StoryObj

export const Healthy: Story = {
  render: () => (
    <StoryFrame>
      <Box sx={{ maxWidth: 520 }}>
        <EmbeddingsPanel data={fixtureIndexHealth} onRefresh={() => {}} />
      </Box>
    </StoryFrame>
  ),
}

export const Degraded: Story = {
  render: () => (
    <StoryFrame>
      <Box sx={{ maxWidth: 520 }}>
        <EmbeddingsPanel data={fixtureIndexHealthDegraded} onRefresh={() => {}} />
      </Box>
    </StoryFrame>
  ),
}
