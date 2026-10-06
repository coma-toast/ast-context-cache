import type { Meta, StoryObj } from '@storybook/react-vite'
import { StoryFrame } from '../storybook/StoryFrame'
import { SettingsTab } from '../tabs/SettingsTab'
import { FeaturesSection } from '../components/FeaturesSection'
import { HandoffSettingsSection } from '../components/HandoffSettingsSection'
import { NetworkAccessSection } from '../components/NetworkAccessSection'
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

/** Features card alone: defaults, a dashboard setting, an env-locked flag, and nested handoff children. */
export const Features: Story = {
  render: () => (
    <StoryFrame>
      <FeaturesSection />
    </StoryFrame>
  ),
}

/** Handoff limits card: every knob at its default, with max depth locked by its env var. */
export const Handoff: Story = {
  render: () => (
    <StoryFrame>
      <HandoffSettingsSection data={fixtureSettings} save={async () => {}} />
    </StoryFrame>
  ),
}

/** Network access card: extra addresses (one failing to bind), env-locked trusted hosts, no token yet. */
export const NetworkAccess: Story = {
  render: () => (
    <StoryFrame>
      <NetworkAccessSection />
    </StoryFrame>
  ),
}
