import type { Meta, StoryObj } from '@storybook/react-vite'

import type { UpdateCheckResult, UpdateStatus } from '../api/types'
import { UpdatesCard, type UpdatesCardProps } from '../components/UpdatesSection'
import { StoryFrame } from '../storybook/StoryFrame'

const meta: Meta = {
  title: 'Dashboard/Updates',
}

export default meta

type Story = StoryObj

const check: UpdateCheckResult = {
  current_version: '4.0.6',
  build: 'release',
  source_build: false,
  latest_version: '4.0.7',
  release_url: 'https://github.com/coma-toast/ast-context-cache/releases/tag/v4.0.7',
  published_at: '2026-10-06T12:00:00Z',
  asset_name: 'ast-context-cache_4.0.7_darwin_arm64.tar.gz',
  update_available: true,
}

const status = (overrides: Partial<UpdateStatus>): UpdateStatus => ({
  active: false,
  done: false,
  phase: '',
  error: '',
  started_at: '',
  finished_at: '',
  from_version: '4.0.6',
  to_version: '4.0.7',
  ...overrides,
})

const card = (props: Partial<UpdatesCardProps>) => (
  <StoryFrame>
    <UpdatesCard
      check={check}
      status={null}
      checking={false}
      starting={false}
      restarting={false}
      onCheck={() => {}}
      onUpdate={() => {}}
      onRestart={() => {}}
      {...props}
    />
  </StoryFrame>
)

/** A newer release is out for this release build. */
export const Available: Story = { render: () => card({}) }

/** Running a local source build: the release replaces it, with a warning. */
export const SourceBuild: Story = {
  render: () => card({ check: { ...check, current_version: '4.0.7', build: 'source', source_build: true } }),
}

/** Mid-update, verifying the downloaded archive. */
export const Installing: Story = { render: () => card({ status: status({ active: true, phase: 'verifying' }) }) }

/** Installed and waiting for a restart. */
export const Installed: Story = { render: () => card({ status: status({ done: true, phase: 'installed' }) }) }

/** Already on the latest release. */
export const UpToDate: Story = {
  render: () => card({ check: { ...check, current_version: '4.0.7', update_available: false } }),
}

/** The download failed verification. */
export const Failed: Story = {
  render: () => card({ status: status({ phase: 'error', error: 'downloaded archive does not match checksums.txt' }) }),
}
