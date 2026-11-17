import { useEffect, useRef, type ReactNode } from 'react'
import type { Meta, StoryObj } from '@storybook/react-vite'
import { StoryFrame } from '../storybook/StoryFrame'
import { MemoryTab } from '../tabs/MemoryTab'
import { fixtureMemory, fixtureMemoryEmptyDocs } from '../storybook/fixtures'
import type {
  ContextNoteDetailResponse,
  ContextNoteListResponse,
  DocEntriesResponse,
  KBDetailResponse,
  KBEntry,
  KBListResponse,
  MemoryData,
} from '../api/types'

const storyMemoryData: MemoryData = {
  ...fixtureMemory,
  ActiveFacts: 14,
  ActiveProcedures: 5,
  ActivePreferences: 8,
  ActiveEvents: 6,
  ActiveConfigChanges: 3,
  ActiveDiscoveries: 4,
  QuarantinedCount: 1,
}

const storyKBEntries: KBEntry[] = [
  {
    ref: 'mem_fact_01',
    kind: 'fact',
    scope: 'project',
    project_path: '/Users/demo/project',
    subject: 'SQLite WAL',
    predicate: 'checkpoint_mode',
    object: 'TRUNCATE during maintenance',
    title: 'SQLite WAL checkpoint mode',
    body: 'Dashboard maintenance runs **TRUNCATE** checkpoints when WAL exceeds threshold.',
    tags: ['sqlite', 'storage'],
    actor: 'agent',
    token_est: 28,
    access_count: 11,
    created_at: '2026-10-27T10:15:00Z',
    verified: true,
  },
  {
    ref: 'mem_pref_01',
    kind: 'preference',
    scope: 'global',
    rule: 'Always run go test ./... before opening a PR.',
    title: 'Run full Go test suite before PR',
    body: 'Applies across all repositories unless a narrower fast-check is explicitly requested.',
    tags: ['workflow', 'testing'],
    actor: 'human',
    category: 'workflow',
    token_est: 24,
    access_count: 19,
    created_at: '2026-10-20T09:00:00Z',
    verified: true,
  },
  {
    ref: 'mem_event_01',
    kind: 'event',
    scope: 'project',
    project_path: '/Users/demo/project',
    title: 'ONNX embedder recovered after timeout',
    body: 'Probe succeeded after reloading `bge-small-en-v1.5`.',
    severity: 'warning',
    actor: 'system',
    occurred_at: '2026-10-28T08:40:00Z',
    created_at: '2026-10-28T08:40:05Z',
    tags: ['embedder'],
    token_est: 20,
    access_count: 3,
  },
  {
    ref: 'mem_cfg_01',
    kind: 'config_change',
    scope: 'project',
    project_path: '/Users/demo/project',
    title: 'Increased embed_queue_low_cap',
    target: 'embed_queue_low_cap',
    before_value: '2048',
    after_value: '4096',
    reason: 'Avoid dropping background jobs on large monorepo indexing.',
    actor: 'human',
    occurred_at: '2026-10-26T16:10:00Z',
    created_at: '2026-10-26T16:10:00Z',
    tags: ['config'],
    token_est: 26,
    access_count: 4,
  },
  {
    ref: 'mem_disc_01',
    kind: 'discovery',
    scope: 'project',
    project_path: '/Users/demo/project',
    title: 'Handoff scratchpad deduplicates sibling searches',
    body: 'Sharing live trail entries reduces repeated `search_semantic` calls by ~18%.',
    actor: 'agent',
    occurred_at: '2026-10-25T14:22:00Z',
    created_at: '2026-10-25T14:22:00Z',
    tags: ['handoff', 'performance'],
    token_est: 25,
    access_count: 7,
  },
  {
    ref: 'mem_quar_01',
    kind: 'fact',
    scope: 'project',
    project_path: '/Users/demo/project',
    subject: 'deploy_host',
    predicate: 'port',
    object: '8443',
    title: 'Quarantined host port note',
    body: 'Unverified note awaiting operator review.',
    quarantined: 1,
    actor: 'agent',
    created_at: '2026-10-28T11:00:00Z',
    tags: ['network'],
    token_est: 18,
    access_count: 0,
  },
]

const storyContextNotes: ContextNoteListResponse = {
  items: [
    {
      ref: 'ctx_5c1e0f2a9b3d',
      session_id: 'hof_a1b2c3d4e5f60718.c1',
      project_path: '/Users/demo/project',
      label: 'Auth middleware trace result',
      snippet: 'Auth runs in middleware/session.go; tokens refresh in RefreshIfStale before every handler.',
      tags: 'auth,handoff',
      kind: 'handoff_result',
      virtual_tokens: 1420,
      access_count: 4,
      revision: 2,
      created_at: '2026-10-28T09:15:00Z',
      last_accessed_at: '2026-10-28T10:31:05Z',
      handoff_tree_active: true,
    },
    {
      ref: 'ctx_9a8b7c6d5e4f',
      session_id: 'sess-demo-main',
      project_path: '/Users/demo/project',
      label: 'Migration checklist',
      snippet: 'Step 1: verify SQLite schema version. Step 2: run Vitest & Playwright story checks.',
      tags: 'plan,ui',
      kind: 'context_note',
      virtual_tokens: 680,
      access_count: 2,
      revision: 2,
      created_at: '2026-10-27T17:00:00Z',
      last_accessed_at: '2026-10-28T08:10:00Z',
      handoff_tree_active: false,
    },
  ],
  total: 2,
  page: 1,
  per_page: 50,
  total_pages: 1,
}

const storyContextNoteDetails: Record<string, ContextNoteDetailResponse> = {
  ctx_5c1e0f2a9b3d: {
    note: {
      ref: 'ctx_5c1e0f2a9b3d',
      session_id: 'hof_a1b2c3d4e5f60718.c1',
      project_path: '/Users/demo/project',
      label: 'Auth middleware trace result',
      content:
        '# Auth Middleware Trace\n\n- Entry: `middleware/session.go`\n- Refresh: `RefreshIfStale` runs before every handler.\n- Token cache TTL: 5m.',
      tags: 'auth,handoff',
      kind: 'handoff_result',
      virtual_tokens: 1420,
      access_count: 4,
      created_at: '2026-10-28T09:15:00Z',
      last_accessed_at: '2026-10-28T10:31:05Z',
      revision: 2,
    },
    revisions: [
      {
        revision: 1,
        content: '# Auth Middleware Trace\n\n- Entry: `middleware/session.go`\n- Refresh: TBD',
        token_est: 980,
        op: 'replace',
        created_at: '2026-10-28T09:30:00Z',
      },
    ],
    active_handoff_tree: true,
  },
  ctx_9a8b7c6d5e4f: {
    note: {
      ref: 'ctx_9a8b7c6d5e4f',
      session_id: 'sess-demo-main',
      project_path: '/Users/demo/project',
      label: 'Migration checklist',
      content: '## Checklist\n\n1. Verify SQLite schema version.\n2. Run Vitest & Playwright story checks.',
      tags: 'plan,ui',
      kind: 'context_note',
      virtual_tokens: 680,
      access_count: 2,
      created_at: '2026-10-27T17:00:00Z',
      last_accessed_at: '2026-10-28T08:10:00Z',
      revision: 2,
    },
    revisions: [
      {
        revision: 1,
        content: '## Checklist\n\n1. Verify SQLite schema version.',
        token_est: 420,
        op: 'append',
        created_at: '2026-10-27T17:10:00Z',
      },
    ],
    active_handoff_tree: false,
  },
}

const storyDocEntries: DocEntriesResponse = {
  source_id: 1,
  entries: [
    {
      id: 101,
      source_id: 1,
      title: 'useEffect Reference',
      path: '/reference/react/useEffect',
      content: '# `useEffect`\n\n`useEffect` is a React Hook that lets you synchronize a component with an external system.',
      updated_at: '2026-10-26T12:00:00Z',
    },
  ],
  total: 1,
  page: 1,
  per_page: 50,
  total_pages: 1,
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function handleStoryApiRequest(urlStr: string): Response | null {
  let pathname = urlStr
  let searchParams = new URLSearchParams()
  try {
    const parsed = new URL(urlStr, 'http://localhost')
    pathname = parsed.pathname
    searchParams = parsed.searchParams
  } catch {
    // keep raw
  }

  if (pathname === '/api/dashboard/kb') {
    const kindsParam = searchParams.get('kinds')
    const statusParam = searchParams.get('status')
    const qParam = (searchParams.get('q') || '').toLowerCase()
    const kinds = kindsParam ? kindsParam.split(',').filter(Boolean) : []
    const filtered = storyKBEntries.filter((e) => {
      if (statusParam === 'quarantined' && !e.quarantined) return false
      if (statusParam === 'active' && (e.valid_until || e.quarantined)) return false
      if (kinds.length > 0 && !kinds.includes(e.kind)) return false
      if (qParam) {
        const hay = [e.ref, e.title, e.body, e.subject, e.predicate, e.object, e.rule]
          .filter(Boolean)
          .join(' ')
          .toLowerCase()
        if (!hay.includes(qParam)) return false
      }
      return true
    })
    const res: KBListResponse = {
      entries: filtered,
      total: filtered.length,
      page: 1,
      per_page: 50,
      total_pages: 1,
      counts: {
        fact: 14,
        procedure: 5,
        preference: 8,
        event: 6,
        config_change: 3,
        discovery: 4,
        total: 40,
        quarantined: 1,
      },
    }
    return jsonResponse(res)
  }

  if (pathname === '/api/dashboard/kb/entry') {
    const ref = searchParams.get('ref') || ''
    const found = storyKBEntries.find((e) => e.ref === ref) ?? storyKBEntries[0]
    const detail: KBDetailResponse = {
      entry: found,
      history: [found],
      conflicts: [],
    }
    return jsonResponse(detail)
  }

  if (pathname === '/api/dashboard/context-notes') {
    return jsonResponse(storyContextNotes)
  }

  if (pathname === '/api/dashboard/context-notes/detail') {
    const ref = searchParams.get('ref') || ''
    const detail = storyContextNoteDetails[ref] ?? storyContextNoteDetails.ctx_5c1e0f2a9b3d
    return jsonResponse(detail)
  }

  if (pathname === '/api/dashboard/doc-entries') {
    const sourceId = Number(searchParams.get('source_id') || '1')
    return jsonResponse({ ...storyDocEntries, source_id: sourceId })
  }

  return null
}

let savedRealFetch: typeof window.fetch | null = null
let activeStoryMocks = 0

function StoryFetchStub({ children }: { children: ReactNode }) {
  const installedRef = useRef(false)
  if (!installedRef.current && typeof window !== 'undefined') {
    if (!savedRealFetch) {
      savedRealFetch = window.fetch.bind(window)
    }
    activeStoryMocks += 1
    installedRef.current = true
    const realFetch = savedRealFetch
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const mocked = handleStoryApiRequest(url)
      if (mocked) return mocked
      return realFetch(input, init)
    }
  }

  useEffect(() => {
    return () => {
      if (installedRef.current) {
        installedRef.current = false
        activeStoryMocks = Math.max(0, activeStoryMocks - 1)
        if (activeStoryMocks === 0 && savedRealFetch) {
          window.fetch = savedRealFetch
          savedRealFetch = null
        }
      }
    }
  }, [])

  return <>{children}</>
}

const meta: Meta = {
  title: 'Dashboard/Memory',
  decorators: [
    (Story) => (
      <StoryFetchStub>
        <StoryFrame>
          <Story />
        </StoryFrame>
      </StoryFetchStub>
    ),
  ],
}

export default meta

type Story = StoryObj

export const Healthy: Story = {
  render: () => (
    <MemoryTab data={storyMemoryData} onRefresh={() => {}} docSourcesPage={1} onDocSourcesPageChange={() => {}} />
  ),
}

export const Timeline: Story = {
  render: () => (
    <MemoryTab
      data={storyMemoryData}
      subTab="timeline"
      onRefresh={() => {}}
      docSourcesPage={1}
      onDocSourcesPageChange={() => {}}
    />
  ),
}

export const Preferences: Story = {
  render: () => (
    <MemoryTab
      data={storyMemoryData}
      subTab="preferences"
      onRefresh={() => {}}
      docSourcesPage={1}
      onDocSourcesPageChange={() => {}}
    />
  ),
}

export const VirtualContext: Story = {
  render: () => (
    <MemoryTab
      data={storyMemoryData}
      subTab="virtual-context"
      onRefresh={() => {}}
      docSourcesPage={1}
      onDocSourcesPageChange={() => {}}
    />
  ),
}

export const DocSources: Story = {
  render: () => (
    <MemoryTab
      data={storyMemoryData}
      subTab="doc-sources"
      onRefresh={() => {}}
      docSourcesPage={1}
      onDocSourcesPageChange={() => {}}
    />
  ),
}

export const EmptyDocs: Story = {
  render: () => (
    <MemoryTab
      data={fixtureMemoryEmptyDocs}
      subTab="doc-sources"
      onRefresh={() => {}}
      docSourcesPage={1}
      onDocSourcesPageChange={() => {}}
    />
  ),
}
