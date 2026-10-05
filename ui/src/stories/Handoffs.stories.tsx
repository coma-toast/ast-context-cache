import type { Meta, StoryObj } from '@storybook/react-vite'

import { StoryFrame } from '../storybook/StoryFrame'
import { HandoffTreesCard } from '../components/HandoffTreesCard'
import { OverviewTab } from '../tabs/OverviewTab'
import {
  fixtureContextSessions,
  fixtureHandoffTrees,
  fixtureHandoffTreesEmpty,
  fixtureStats,
  fixtureWeeklyDigest,
} from '../storybook/fixtures'

const meta: Meta = {
  title: 'Dashboard/Handoffs',
}

export default meta

type Story = StoryObj

/** Overview with the handoff trees card under virtual context: a live tree with a nested handoff, and an expired tree. */
export const Overview: Story = {
  render: () => (
    <StoryFrame>
      <OverviewTab
        stats={fixtureStats}
        weeklyDigest={fixtureWeeklyDigest}
        contextSessions={fixtureContextSessions}
        handoffTrees={fixtureHandoffTrees}
      />
    </StoryFrame>
  ),
}

/** The card before any agent has handed off work. */
export const Empty: Story = {
  render: () => (
    <StoryFrame>
      <HandoffTreesCard data={fixtureHandoffTreesEmpty} />
    </StoryFrame>
  ),
}
