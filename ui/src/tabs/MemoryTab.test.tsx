import { ThemeProvider } from '@mui/material/styles'
import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  ContextNoteDetailResponse,
  ContextNoteListResponse,
  DocEntriesResponse,
  KBActionResponse,
  KBDetailResponse,
  KBEntry,
  KBListResponse,
  KBMutationResponse,
  MemoryData,
} from '../api/types'
import { ToastProvider } from '../context/ToastContext'
import { fixtureMemory } from '../storybook/fixtures'
import { typeInto } from '../test/settingsHarness'
import { dashboardTheme } from '../theme'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const activeFactEntry: KBEntry = {
  ref: 'mem_fact_01',
  kind: 'fact',
  scope: 'project',
  project_path: '/Users/demo/project',
  subject: 'SQLite WAL',
  predicate: 'checkpoint_mode',
  object: 'TRUNCATE',
  title: 'SQLite WAL checkpoint mode',
  body: 'Maintains WAL bounds.',
  tags: ['sqlite'],
  actor: 'agent',
  created_at: '2026-10-27T10:00:00Z',
  verified: false,
}

const quarantinedEntry: KBEntry = {
  ref: 'mem_quar_01',
  kind: 'fact',
  scope: 'project',
  project_path: '/Users/demo/project',
  subject: 'deploy_host',
  predicate: 'port',
  object: '8443',
  title: 'Quarantined host port note',
  body: 'Needs verification.',
  quarantined: 1,
  actor: 'agent',
  created_at: '2026-10-28T11:00:00Z',
}

const sampleKBList: KBListResponse = {
  entries: [activeFactEntry],
  total: 1,
  page: 1,
  per_page: 50,
  total_pages: 1,
  counts: {
    fact: 1,
    procedure: 0,
    preference: 0,
    event: 0,
    config_change: 0,
    discovery: 0,
    total: 1,
    quarantined: 1,
  },
}

const sampleContextNotes: ContextNoteListResponse = {
  items: [
    {
      ref: 'ctx_handoff_01',
      session_id: 'hof_abc.c1',
      project_path: '/Users/demo/project',
      label: 'Auth trace result',
      snippet: 'Tokens refresh in RefreshIfStale.',
      tags: 'auth,handoff',
      kind: 'handoff_result',
      virtual_tokens: 420,
      access_count: 2,
      revision: 2,
      created_at: '2026-10-28T09:00:00Z',
      last_accessed_at: '2026-10-28T10:00:00Z',
      handoff_tree_active: true,
    },
  ],
  total: 1,
  page: 1,
  per_page: 50,
  total_pages: 1,
}

const sampleContextDetail: ContextNoteDetailResponse = {
  note: {
    ref: 'ctx_handoff_01',
    session_id: 'hof_abc.c1',
    project_path: '/Users/demo/project',
    label: 'Auth trace result',
    content: '# Auth Trace\n\nUpdated finding.',
    tags: 'auth,handoff',
    kind: 'handoff_result',
    virtual_tokens: 420,
    access_count: 2,
    created_at: '2026-10-28T09:00:00Z',
    last_accessed_at: '2026-10-28T10:00:00Z',
    revision: 2,
  },
  revisions: [
    {
      revision: 1,
      content: '# Auth Trace\n\nInitial finding.',
      token_est: 300,
      op: 'replace',
      created_at: '2026-10-28T09:15:00Z',
    },
  ],
  active_handoff_tree: true,
}

const sampleDocEntries: DocEntriesResponse = {
  source_id: 1,
  entries: [
    {
      id: 10,
      source_id: 1,
      title: 'Untrusted Doc Page',
      path: '/docs/untrusted',
      content: '# Safe Heading\n\nHello <script>window.__xss = true; alert("xss")</script> world',
      updated_at: '2026-10-28T12:00:00Z',
    },
  ],
  total: 1,
  page: 1,
  per_page: 50,
  total_pages: 1,
}

const {
  kbListMock,
  kbGetEntryMock,
  kbCreateEntryMock,
  kbUpdateEntryMock,
  kbEntryActionMock,
  contextNotesListMock,
  contextNoteDetailMock,
  contextNoteDeleteMock,
  flushContextOrphansMock,
  docEntriesMock,
  docSourceActionMock,
} = vi.hoisted(() => ({
  kbListMock: vi.fn(),
  kbGetEntryMock: vi.fn(),
  kbCreateEntryMock: vi.fn(),
  kbUpdateEntryMock: vi.fn(),
  kbEntryActionMock: vi.fn(),
  contextNotesListMock: vi.fn(),
  contextNoteDetailMock: vi.fn(),
  contextNoteDeleteMock: vi.fn(),
  flushContextOrphansMock: vi.fn(),
  docEntriesMock: vi.fn(),
  docSourceActionMock: vi.fn(),
}))

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      kbList: kbListMock,
      kbGetEntry: kbGetEntryMock,
      kbCreateEntry: kbCreateEntryMock,
      kbUpdateEntry: kbUpdateEntryMock,
      kbEntryAction: kbEntryActionMock,
      contextNotesList: contextNotesListMock,
      contextNoteDetail: contextNoteDetailMock,
      contextNoteDelete: contextNoteDeleteMock,
      flushContextOrphans: flushContextOrphansMock,
      docEntries: docEntriesMock,
      docSourceAction: docSourceActionMock,
    },
  }
})

const { MemoryTab } = await import('./MemoryTab')

const testMemoryData: MemoryData = {
  ...fixtureMemory,
  ActiveFacts: 14,
  ActivePreferences: 8,
  ActiveEvents: 6,
  ActiveConfigChanges: 3,
  ActiveDiscoveries: 4,
  QuarantinedCount: 1,
}

let container: HTMLDivElement
let root: Root

function renderWithProviders(ui: ReactNode) {
  return act(async () => {
    root.render(
      <ThemeProvider theme={dashboardTheme}>
        <ToastProvider>{ui}</ToastProvider>
      </ThemeProvider>,
    )
  })
}

async function clickAndFlush(el: HTMLElement) {
  await act(async () => {
    el.click()
  })
}

function findButton(label: string, scope: ParentNode = document.body): HTMLButtonElement {
  const btn = [...scope.querySelectorAll('button')].find(
    (b) => b.textContent?.trim() === label || b.textContent?.includes(label),
  )
  if (!btn) {
    throw new Error(`Button "${label}" not found`)
  }
  return btn
}

function findInputByLabel(labelPrefix: string, scope: ParentNode = document.body): HTMLInputElement | HTMLTextAreaElement {
  const controls = [...scope.querySelectorAll('.MuiFormControl-root')]
  for (const ctrl of controls) {
    const lbl = ctrl.querySelector('label')?.textContent?.trim() ?? ''
    if (lbl.startsWith(labelPrefix)) {
      const input = ctrl.querySelector<HTMLInputElement | HTMLTextAreaElement>('input, textarea')
      if (input) return input
    }
  }
  throw new Error(`Input with label "${labelPrefix}" not found`)
}

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
  vi.clearAllMocks()
  kbListMock.mockImplementation(async () => sampleKBList)
  kbGetEntryMock.mockImplementation(
    async (ref: string): Promise<KBDetailResponse> => ({
      entry: ref === quarantinedEntry.ref ? quarantinedEntry : activeFactEntry,
      history: [ref === quarantinedEntry.ref ? quarantinedEntry : activeFactEntry],
      conflicts: [],
    }),
  )
  kbCreateEntryMock.mockImplementation(
    async (): Promise<KBMutationResponse> => ({
      entry: { ...activeFactEntry, ref: 'mem_created_01' },
      warnings: ['Possible secret redacted in body'],
      redacted_fields: ['body'],
    }),
  )
  kbUpdateEntryMock.mockImplementation(
    async (): Promise<KBMutationResponse> => ({
      entry: activeFactEntry,
      warnings: ['Possible secret redacted in body on edit'],
      redacted_fields: ['body'],
    }),
  )
  kbEntryActionMock.mockImplementation(
    async ({ ref, action }: { ref: string; action: KBActionResponse['action'] }): Promise<KBActionResponse> => ({
      status: 'ok',
      action,
      ref,
      entry: null,
    }),
  )
  contextNotesListMock.mockImplementation(async () => sampleContextNotes)
  contextNoteDetailMock.mockImplementation(async () => sampleContextDetail)
  contextNoteDeleteMock.mockImplementation(async (ref: string) => ({
    status: 'deleted',
    ref,
    virtual_tokens_freed: 420,
  }))
  flushContextOrphansMock.mockImplementation(async () => ({
    flushed_refs: 2,
    virtual_tokens_freed: 500,
    kept_recent: 0,
  }))
  docEntriesMock.mockImplementation(async () => sampleDocEntries)
  docSourceActionMock.mockImplementation(async () => ({ status: 'ok' }))
  delete (window as { __xss?: boolean }).__xss
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  delete (window as { __xss?: boolean }).__xss
})

describe('MemoryTab', () => {
  it('renders all 5 sub-tabs, summary chips, and switches between controlled/uncontrolled sub-tabs', async () => {
    const onSubTabChange = vi.fn()
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
        subTab="explorer"
        onSubTabChange={onSubTabChange}
      />,
    )

    expect(container.textContent).toContain('Memory')
    expect(container.textContent).toContain('Facts: 14')
    expect(container.textContent).toContain('Preferences: 8')
    expect(container.textContent).toContain('Events: 6')
    expect(container.textContent).toContain('Config changes: 3')
    expect(container.textContent).toContain('Discoveries: 4')
    expect(container.textContent).toContain('Quarantined: 1')

    const tabs = [...container.querySelectorAll('[role="tab"]')]
    expect(tabs.map((t) => t.textContent)).toEqual([
      'Explorer',
      'Timeline',
      'Preferences',
      'Virtual context',
      'Doc sources',
    ])

    await clickAndFlush(tabs[1] as HTMLElement)
    expect(onSubTabChange).toHaveBeenCalledWith('timeline')

    await clickAndFlush(tabs[3] as HTMLElement)
    expect(onSubTabChange).toHaveBeenCalledWith('virtual-context')

    // Rerender with controlled subTab="virtual-context"
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
        subTab="virtual-context"
        onSubTabChange={onSubTabChange}
      />,
    )
    expect(container.querySelector('[data-testid="virtual-context-subtab"]')).not.toBeNull()

    // Uncontrolled mode switching
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
      />,
    )
    const uncontrolledTabs = [...container.querySelectorAll('[role="tab"]')]
    await clickAndFlush(uncontrolledTabs[4] as HTMLElement)
    expect(container.querySelector('[data-testid="doc-sources-subtab"]')).not.toBeNull()
  })

  it('supports Explorer search/filter and displays secret redaction warning toast on create and edit', async () => {
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
      />,
    )

    expect(kbListMock).toHaveBeenCalled()

    // Filter by kind chip
    const factChip = [...container.querySelectorAll('.MuiChip-clickable')].find((el) =>
      el.textContent?.startsWith('Fact'),
    ) as HTMLElement
    await clickAndFlush(factChip)
    expect(kbListMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        kinds: ['fact'],
      }),
    )

    // Search with Enter key
    const searchInput = container.querySelector<HTMLInputElement>(
      'input[placeholder*="Search knowledge base"]',
    )!
    typeInto(searchInput, 'WAL')
    await act(async () => {
      searchInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
    })
    expect(kbListMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        q: 'WAL',
        kinds: ['fact'],
      }),
    )

    // Create entry returning secret redaction warning
    await clickAndFlush(findButton('New Entry', container))
    const dialog = document.body.querySelector('[role="dialog"]')!
    expect(dialog.textContent).toContain('New Knowledge Base Entry')

    typeInto(findInputByLabel('Subject', dialog), 'api_service')
    typeInto(findInputByLabel('Object', dialog), 'configured')
    typeInto(findInputByLabel('Body (Markdown)', dialog), 'token=sk_secret_123')

    await clickAndFlush(findButton('Create Entry', dialog))
    expect(kbCreateEntryMock).toHaveBeenCalledTimes(1)
    expect(document.body.textContent).toContain('Possible secret redacted in body')

    // Edit entry via detail drawer returning secret redaction warning
    await clickAndFlush(findButton('Edit', document.body))
    const editDialog = document.body.querySelector('[aria-labelledby="kb-entry-form-title"]')!
    expect(editDialog).not.toBeNull()
    expect(editDialog.textContent).toContain('Edit Knowledge Base Entry')
    await clickAndFlush(findButton('Save Changes', editDialog))
    expect(kbUpdateEntryMock).toHaveBeenCalledTimes(1)
    expect(document.body.textContent).toContain('Possible secret redacted in body on edit')
  })

  it('requires confirmation dialog before executing forget or reject_quarantine actions', async () => {
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
        selectedRef="mem_fact_01"
        onSelectRef={vi.fn()}
      />,
    )

    // Click Forget -> dialog must open, action must NOT run yet
    await clickAndFlush(findButton('Forget', document.body))
    expect(kbEntryActionMock).not.toHaveBeenCalled()
    expect(document.body.textContent).toContain('Forget knowledge base entry?')

    // Confirm Forget Entry
    await clickAndFlush(findButton('Forget Entry', document.body))
    expect(kbEntryActionMock).toHaveBeenCalledWith({
      ref: 'mem_fact_01',
      action: 'forget',
      verified: undefined,
    })

    kbEntryActionMock.mockClear()

    // Open quarantined entry -> click Reject -> dialog must open first
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
        selectedRef="mem_quar_01"
        onSelectRef={vi.fn()}
      />,
    )

    await clickAndFlush(findButton('Reject', document.body))
    expect(kbEntryActionMock).not.toHaveBeenCalled()
    expect(document.body.textContent).toContain('Reject quarantined entry?')

    await clickAndFlush(findButton('Reject Entry', document.body))
    expect(kbEntryActionMock).toHaveBeenCalledWith({
      ref: 'mem_quar_01',
      action: 'reject_quarantine',
      verified: undefined,
    })
  })

  it('renders VirtualContextSubtab ctx_* notes, revision viewer, and active-handoff warning on delete', async () => {
    const onRefresh = vi.fn()
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={onRefresh}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
        subTab="virtual-context"
        onSubTabChange={vi.fn()}
      />,
    )

    expect(contextNotesListMock).toHaveBeenCalled()
    expect(container.textContent).toContain('Inventory tokens')
    expect(container.textContent).toContain('30d utilization')
    expect(container.textContent).toContain('Orphans')
    expect(container.textContent).toContain('ctx_handoff_01')

    // Click row to open detail drawer
    const row = container.querySelector<HTMLElement>('[data-testid="context-note-row-ctx_handoff_01"]')!
    await clickAndFlush(row)
    expect(contextNoteDetailMock).toHaveBeenCalledWith('ctx_handoff_01')
    expect(document.body.textContent).toContain('Updated finding.')

    // Switch to Revisions tab inside drawer
    const revTab = [...document.body.querySelectorAll('[role="tab"]')].find((t) =>
      t.textContent?.includes('Revisions (1)'),
    ) as HTMLElement
    await clickAndFlush(revTab)
    expect(document.body.querySelector('[data-testid="revision-item-1"]')).not.toBeNull()
    expect(document.body.querySelector('[data-testid="revision-diff-viewer"]')).not.toBeNull()
    expect(document.body.textContent).toContain('Initial finding.')

    // Click Delete note -> confirmation dialog shows active handoff warning before calling API
    await clickAndFlush(findButton('Delete note', document.body))
    expect(contextNoteDeleteMock).not.toHaveBeenCalled()
    const warning = document.body.querySelector('[data-testid="active-handoff-warning"]')
    expect(warning).not.toBeNull()
    expect(warning?.textContent).toContain('active handoff tree')

    await clickAndFlush(findButton('Confirm delete', document.body))
    expect(contextNoteDeleteMock).toHaveBeenCalledWith('ctx_handoff_01')
    expect(onRefresh).toHaveBeenCalled()

    // Flush orphans requires confirmation dialog
    await clickAndFlush(findButton('Flush orphans', container))
    expect(flushContextOrphansMock).not.toHaveBeenCalled()
    await clickAndFlush(findButton('Confirm flush orphans', document.body))
    expect(flushContextOrphansMock).toHaveBeenCalledTimes(1)
  })

  it('opens cached doc_content viewer in DocSourcesSubtab and renders Markdown safely', async () => {
    await renderWithProviders(
      <MemoryTab
        data={testMemoryData}
        onRefresh={vi.fn()}
        docSourcesPage={1}
        onDocSourcesPageChange={vi.fn()}
        subTab="doc-sources"
        onSubTabChange={vi.fn()}
      />,
    )

    const row = container.querySelector<HTMLElement>('[data-testid="doc-source-row-1"]')!
    expect(row).not.toBeNull()
    await clickAndFlush(row)

    expect(docEntriesMock).toHaveBeenCalledWith(1, undefined, 1, 50)
    const reader = document.body.querySelector('[data-testid="doc-entry-reader"]')
    expect(reader).not.toBeNull()
    expect(reader?.querySelector('h1')?.textContent).toBe('Safe Heading')
    expect(reader?.textContent).toContain('Hello')
    expect(reader?.textContent).toContain('world')
    expect(document.body.querySelector('script')).toBeNull()
    expect((window as { __xss?: boolean }).__xss).toBeUndefined()
  })
})
