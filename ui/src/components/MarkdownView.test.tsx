import { ThemeProvider } from '@mui/material/styles'
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { dashboardTheme } from '../theme'
import { MarkdownView } from './MarkdownView'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let container: HTMLDivElement
let root: Root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

const renderMarkdown = (content?: string | null) => {
  act(() => {
    root.render(
      <ThemeProvider theme={dashboardTheme}>
        <MarkdownView content={content} />
      </ThemeProvider>,
    )
  })
}

describe('MarkdownView', () => {
  it.each([undefined, null, '', '   \n\t  '])('renders nothing for empty content (%j)', (value) => {
    renderMarkdown(value)
    expect(container.querySelector('[data-testid="markdown-view"]')).toBeNull()
    expect(container.textContent).toBe('')
  })

  it('renders headings, paragraphs, lists, task items, and inline/fenced code', () => {
    const md = [
      '# Title',
      '## Section',
      '### Subsection',
      '',
      'A paragraph with `inlineCode()` and ~~removed~~ text.',
      '',
      '- First item',
      '- [x] Completed task',
      '- [ ] Pending task',
      '',
      '1. Step one',
      '2. Step two',
      '',
      '```ts',
      'const answer = 42',
      '```',
    ].join('\n')

    renderMarkdown(md)

    expect(container.querySelector('h1')?.textContent).toBe('Title')
    expect(container.querySelector('h2')?.textContent).toBe('Section')
    expect(container.querySelector('h3')?.textContent).toBe('Subsection')
    expect(container.querySelector('p')?.textContent).toContain('A paragraph with inlineCode()')
    expect(container.querySelector('del')?.textContent).toBe('removed')

    const inlineCodes = [...container.querySelectorAll('p code')]
    expect(inlineCodes.map((el) => el.textContent)).toContain('inlineCode()')

    const codeBlock = container.querySelector('pre code')
    expect(codeBlock).not.toBeNull()
    expect(codeBlock?.textContent).toContain('const answer = 42')

    const ulItems = [...container.querySelectorAll('ul > li')]
    expect(ulItems).toHaveLength(3)
    const checkboxes = [...container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')]
    expect(checkboxes).toHaveLength(2)
    expect(checkboxes[0].checked).toBe(true)
    expect(checkboxes[1].checked).toBe(false)

    const olItems = [...container.querySelectorAll('ol > li')]
    expect(olItems.map((el) => el.textContent)).toEqual(['Step one', 'Step two'])
  })

  it('renders GFM tables cleanly', () => {
    const md = [
      '| Key | Status |',
      '| --- | --- |',
      '| `feature_kb` | enabled |',
      '| `max_body` | 16 KB |',
    ].join('\n')

    renderMarkdown(md)

    const table = container.querySelector('table')
    expect(table).not.toBeNull()
    const headers = [...container.querySelectorAll('th')].map((th) => th.textContent)
    expect(headers).toEqual(['Key', 'Status'])
    const cells = [...container.querySelectorAll('td')].map((td) => td.textContent)
    expect(cells).toEqual(['feature_kb', 'enabled', 'max_body', '16 KB'])
  })

  it('opens external links safely with rel="noopener noreferrer" and leaves anchors local', () => {
    renderMarkdown('[External](https://example.com/docs) and [Local](#overview)')

    const links = [...container.querySelectorAll('a')]
    expect(links).toHaveLength(2)

    const [ext, local] = links
    expect(ext.getAttribute('href')).toBe('https://example.com/docs')
    expect(ext.getAttribute('target')).toBe('_blank')
    expect(ext.getAttribute('rel')).toBe('noopener noreferrer')

    expect(local.getAttribute('href')).toBe('#overview')
    expect(local.getAttribute('target')).toBeNull()
    expect(local.getAttribute('rel')).toBeNull()
  })

  it('neutralizes raw HTML, script tags, and javascript: links', () => {
    const untrusted = [
      'Hello <script>window.__xss = true; alert("xss")</script> world',
      '',
      '<img src="x" onerror="alert(1)" />',
      '',
      '<iframe src="https://evil.example"></iframe>',
      '',
      '<div onclick="alert(1)">raw div</div>',
      '',
      '[bad link](javascript:alert(1))',
    ].join('\n')

    renderMarkdown(untrusted)

    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('iframe')).toBeNull()
    expect(container.querySelector('[onclick]')).toBeNull()
    expect(container.querySelector('[onerror]')).toBeNull()
    expect((window as { __xss?: boolean }).__xss).toBeUndefined()

    const badLink = container.querySelector('a')
    expect(badLink?.getAttribute('href') ?? '').not.toMatch(/^javascript:/i)
  })
})
