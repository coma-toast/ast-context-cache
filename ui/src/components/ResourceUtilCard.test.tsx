import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { IndexHealth } from '../api/types'
import { mountHarness, type Harness } from '../test/settingsHarness'
import { HostDiskNotice } from './ResourceUtilCard'

let h: Harness

beforeEach(() => {
  h = mountHarness()
})

afterEach(() => h.unmount())

const now = Math.round(Date.now() / 1000)

describe('HostDiskNotice', () => {
  it.each<[string, Partial<IndexHealth>, string]>([
    [
      'a failed poll with a kept sample',
      { HostDiskURLSet: true, HostDiskError: 'host disk health URL answered 503', HostDiskSampledUnix: now - 300, Disks: [{ Device: 'sda', Host: true }] },
      "Can't read host disks: host disk health URL answered 503. Showing the last good sample from 5 min ago.",
    ],
    ['a failed first poll', { HostDiskURLSet: true, HostDiskError: 'connection refused' }, "Can't read host disks: connection refused"],
    ['a guest with no URL', { HostDiskGuest: true }, "Disks belong to the host; SMART isn't visible from here."],
  ])('explains %s', (_name, data, want) => {
    h.render(<HostDiskNotice data={data as IndexHealth} />)
    expect(h.container.textContent).toContain(want)
  })

  it.each<[string, Partial<IndexHealth>]>([
    ['a healthy host sample', { HostDiskGuest: true, HostDiskURLSet: true, HostDiskSampledUnix: now, Disks: [{ Device: 'sda', Host: true }] }],
    ['bare metal without a URL', {}],
  ])('says nothing for %s', (_name, data) => {
    h.render(<HostDiskNotice data={data as IndexHealth} />)
    expect(h.container.textContent).toBe('')
  })
})
