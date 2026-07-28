import type { Meta, StoryObj } from '@storybook/react-vite'
import { StoryFrame } from '../storybook/StoryFrame'
import { SettingsTab } from '../tabs/SettingsTab'
import { fixtureMcpTier, fixtureSettings } from '../storybook/fixtures'

const meta: Meta = {
  title: 'Dashboard/Settings',
}

export default meta

type Story = StoryObj

export const EmbeddingAndVirtual: Story = {
  render: () => (
    <StoryFrame>
      <SettingsTab data={fixtureSettings} mcpTier={fixtureMcpTier} onRefresh={() => {}} />
    </StoryFrame>
  ),
}
