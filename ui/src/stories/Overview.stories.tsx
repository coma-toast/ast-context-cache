import type { Meta, StoryObj } from '@storybook/react-vite'
import { Box } from '@mui/material'
import { StoryFrame } from '../storybook/StoryFrame'
import { HealthBar } from '../components/HealthBar'
import { IndexHealthSection } from '../tabs/IndexHealthSection'
import { OverviewTab } from '../tabs/OverviewTab'
import { fixtureContextSessions, fixtureHealth, fixtureIndexHealth, fixtureStats, fixtureWeeklyDigest } from '../storybook/fixtures'

const meta: Meta = {
  title: 'Dashboard/Overview',
}

export default meta

type Story = StoryObj

export const Hero: Story = {
  render: () => (
    <StoryFrame>
      <Box sx={{ mb: 2 }}>
        <HealthBar health={fixtureHealth} />
      </Box>
      <IndexHealthSection data={fixtureIndexHealth} onRefresh={() => {}} />
      <OverviewTab stats={fixtureStats} weeklyDigest={fixtureWeeklyDigest} contextSessions={fixtureContextSessions} />
    </StoryFrame>
  ),
}

export const IndexRuntime: Story = {
  render: () => (
    <StoryFrame>
      <Box sx={{ mb: 2 }}>
        <HealthBar health={fixtureHealth} />
      </Box>
      <IndexHealthSection data={fixtureIndexHealth} onRefresh={() => {}} />
    </StoryFrame>
  ),
}
