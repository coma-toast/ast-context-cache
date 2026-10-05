import type { Meta, StoryObj } from '@storybook/react-vite'

import { InstallerPreviewDialog, InstallerSection, REPREVIEW_NOTICE } from '../components/InstallerSection'
import { StoryFrame } from '../storybook/StoryFrame'
import { fixtureInstallerPlan } from '../storybook/fixtures'

const meta: Meta = {
  title: 'Dashboard/Installer',
}

export default meta

type Story = StoryObj

/**
 * Agent integration card: Cursor installed, Claude Code modified by the user (hooks shown),
 * VS Code with an unparseable config, JetBrains unsupported, and a legacy warning.
 */
export const Default: Story = {
  render: () => (
    <StoryFrame>
      <InstallerSection />
    </StoryFrame>
  ),
}

/** Cursor install preview after a stale plan was re-previewed: warning, skipped skill, colored diffs. */
export const PreviewOpen: Story = {
  render: () => (
    <StoryFrame>
      <InstallerSection />
      <InstallerPreviewDialog
        open
        targetName="Cursor"
        plan={fixtureInstallerPlan}
        notice={REPREVIEW_NOTICE}
        applying={false}
        onApply={() => {}}
        onClose={() => {}}
      />
    </StoryFrame>
  ),
}
