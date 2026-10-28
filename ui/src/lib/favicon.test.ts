import { afterEach, describe, expect, it } from 'vitest'

import { setFavicon } from './favicon'

afterEach(() => document.head.querySelectorAll('link[rel~="icon"]').forEach((l) => l.remove()))

describe('setFavicon', () => {
  it('creates the icon link once and updates its href in place', () => {
    setFavicon('/dashboard/favicon?v=aaa')
    const link = document.head.querySelector<HTMLLinkElement>('link[rel~="icon"]')!
    expect(link.getAttribute('href')).toBe('/dashboard/favicon?v=aaa')
    setFavicon('/dashboard/favicon?v=bbb')
    expect(document.head.querySelectorAll('link[rel~="icon"]')).toHaveLength(1)
    expect(link.getAttribute('href')).toBe('/dashboard/favicon?v=bbb')
  })
})
