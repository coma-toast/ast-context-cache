import type { Meta, StoryObj } from '@storybook/react-vite'
import { StoryFrame } from '../storybook/StoryFrame'
import { MemoryTab } from '../tabs/MemoryTab'
import { fixtureMemory, fixtureMemoryEmptyDocs } from '../storybook/fixtures'

const meta: Meta = {
  title: 'Dashboard/Memory',
}

export default meta

type Story = StoryObj

export const Healthy: Story = {
  render: () => (
    <StoryFrame>
      <MemoryTab data={fixtureMemory} onRefresh={() => {}} />
    </StoryFrame>
  ),
}

export const EmptyDocs: Story = {
  render: () => (
    <StoryFrame>
      <MemoryTab data={fixtureMemoryEmptyDocs} onRefresh={() => {}} />
    </StoryFrame>
  ),
}
