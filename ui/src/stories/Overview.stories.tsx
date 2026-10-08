import type { Meta, StoryObj } from '@storybook/react-vite'
import { Box } from '@mui/material'
import { StoryFrame } from '../storybook/StoryFrame'
import { HealthBar } from '../components/HealthBar'
import { IndexHealthSection } from '../tabs/IndexHealthSection'
import { OverviewTab } from '../tabs/OverviewTab'
import { WatchersPanel } from '../components/WatchersPanel'
import { fixtureContextSessions, fixtureHandoffTrees, fixtureHealth, fixtureIndexHealth, fixtureStats, fixtureWeeklyDigest } from '../storybook/fixtures'

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
      <WatchersPanel watchers={fixtureIndexHealth.Watchers || []} onRefresh={() => {}} />
    </StoryFrame>
  ),
}

/** Token savings first: activity, the 7-day digest, virtual context, and handoff trees (the README hero). */
export const Savings: Story = {
  render: () => (
    <StoryFrame>
      <Box sx={{ mb: 2 }}>
        <HealthBar health={fixtureHealth} />
      </Box>
      <OverviewTab stats={fixtureStats} weeklyDigest={fixtureWeeklyDigest} contextSessions={fixtureContextSessions} handoffTrees={fixtureHandoffTrees} />
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
