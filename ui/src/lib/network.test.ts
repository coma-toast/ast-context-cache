import { describe, expect, it } from 'vitest'

import type { NetworkState } from '../api/types'

import {
  formatHostPort,
  generateToken,
  isHostname,
  isIPAddress,
  listenerChips,
  needsTokenWarning,
  networkLockReason,
  parseList,
  TOKEN_BYTES,
  validateAddrs,
  validateHosts,
} from './network'

const state = (overrides: Partial<NetworkState> = {}): NetworkState => ({
  listen_extra_addrs: [],
  trusted_hosts: [],
  token_set: false,
  locked: {},
  listeners: [],
  base_listen: '127.0.0.1',
  base_wildcard: false,
  ...overrides,
})

describe('parseList', () => {
  it.each([
    ['', []],
    ['  \n ', []],
    ['100.64.0.1', ['100.64.0.1']],
    ['100.64.0.1, 100.64.0.2\n100.64.0.1\r\n', ['100.64.0.1', '100.64.0.2']],
    ['a.ts.net b.ts.net,,c.ts.net', ['a.ts.net', 'b.ts.net', 'c.ts.net']],
  ])('%j -> %j', (text, want) => {
    expect(parseList(text)).toEqual(want)
  })
})

describe('isIPAddress', () => {
  it.each([
    ['100.105.22.11', true],
    ['255.255.255.255', true],
    ['fd7a:115c:a1e0::1', true],
    ['[fd7a:115c:a1e0::1]', true],
    ['::ffff:100.64.0.1', true],
    ['256.1.1.1', false],
    ['100.105.022.11', false],
    ['1.2.3', false],
    ['100.105.22.11:7830', false],
    ['mac.ts.net', false],
    ['', false],
  ])('%s -> %s', (s, want) => {
    expect(isIPAddress(s)).toBe(want)
  })
})

describe('isHostname', () => {
  it.each([
    ['jasons-macbook-air.halibut-velociraptor.ts.net', true],
    ['Demo-Laptop.Example.ts.net.', true],
    ['short', true],
    ['foo_bar.ts.net', false],
    ['-foo.ts.net', false],
    ['foo-.ts.net', false],
    ['foo..ts.net', false],
    ['*.ts.net', false],
    ['http://foo.ts.net', false],
    [`${'a'.repeat(64)}.ts.net`, false],
  ])('%s -> %s', (s, want) => {
    expect(isHostname(s)).toBe(want)
  })
})

describe('validateAddrs', () => {
  it('accepts tailnet IPv4 and IPv6 addresses', () => {
    expect(validateAddrs(['100.105.22.11', 'fd7a:115c:a1e0::1'])).toBe('')
  })

  it.each([
    ['mac.ts.net', 'is not an IP address'],
    ['0.0.0.0', 'is a wildcard'],
    ['::', 'is a wildcard'],
    ['127.0.0.1', 'is loopback'],
    ['127.8.9.10', 'is loopback'],
    ['::1', 'is loopback'],
    ['0:0:0:0:0:0:0:1', 'is loopback'],
  ])('rejects %s', (entry, want) => {
    expect(validateAddrs(['100.105.22.11', entry])).toContain(want)
  })
})

describe('validateHosts', () => {
  it('accepts MagicDNS names, short names and IPs', () => {
    expect(validateHosts(['mac.tailnet.ts.net.', 'mac', '100.105.22.11'])).toBe('')
  })

  it.each([
    ['bad_host.ts.net', 'is not a valid hostname'],
    ['*.ts.net', 'is not a valid hostname'],
    ['0.0.0.0', 'is a wildcard'],
  ])('rejects %s', (entry, want) => {
    expect(validateHosts([entry])).toContain(want)
  })
})

describe('formatHostPort', () => {
  it('brackets IPv6', () => {
    expect(formatHostPort('100.64.0.1', 7821)).toBe('100.64.0.1:7821')
    expect(formatHostPort('fd7a::1', 7830)).toBe('[fd7a::1]:7830')
  })
})

describe('listenerChips', () => {
  it('maps listening and failed listeners', () => {
    const chips = listenerChips([
      { server: 'mcp', addr: '100.64.0.1', port: 7821, status: 'listening' },
      { server: 'dashboard', addr: '100.64.0.2', port: 7830, status: 'error', error: 'bind: address not available' },
      { server: 'dashboard', addr: 'fd7a::1', port: 7830, status: 'error' },
    ])
    expect(chips).toEqual([
      { key: 'mcp-100.64.0.1', label: 'mcp 100.64.0.1:7821', color: 'success', tooltip: 'Listening on 100.64.0.1:7821' },
      { key: 'dashboard-100.64.0.2', label: 'dashboard 100.64.0.2:7830', color: 'error', tooltip: 'bind: address not available' },
      { key: 'dashboard-fd7a::1', label: 'dashboard [fd7a::1]:7830', color: 'error', tooltip: 'Not listening on [fd7a::1]:7830' },
    ])
  })

  it('handles a missing list', () => {
    expect(listenerChips(null)).toEqual([])
  })
})

describe('networkLockReason', () => {
  it('names the env var for locked keys only', () => {
    const s = state({ locked: { trusted_hosts: 'AST_TRUSTED_HOSTS' } })
    expect(networkLockReason(s, 'trusted_hosts')).toBe('Set by AST_TRUSTED_HOSTS in the environment')
    expect(networkLockReason(s, 'listen_extra_addrs')).toBe('')
    expect(networkLockReason(null, 'remote_access_token')).toBe('')
    expect(networkLockReason(state({ locked: null }), 'trusted_hosts')).toBe('')
  })
})

describe('needsTokenWarning', () => {
  it.each([
    { name: 'nothing exposed', s: state(), want: false },
    { name: 'extra address without token', s: state({ listen_extra_addrs: ['100.64.0.1'] }), want: true },
    { name: 'extra address with token', s: state({ listen_extra_addrs: ['100.64.0.1'], token_set: true }), want: false },
    { name: 'wildcard base without token', s: state({ base_wildcard: true }), want: true },
    { name: 'null addresses', s: state({ listen_extra_addrs: null }), want: false },
  ])('$name', ({ s, want }) => {
    expect(needsTokenWarning(s)).toBe(want)
  })

  it('is false before the state loads', () => {
    expect(needsTokenWarning(null)).toBe(false)
  })
})

describe('generateToken', () => {
  it('encodes TOKEN_BYTES as unpadded base64url', () => {
    const token = generateToken((a) => a.fill(0xfb))
    expect(token).toHaveLength(43)
    expect(token).toMatch(/^[A-Za-z0-9_-]+$/)
    expect(token).toContain('-')
    expect(token).toContain('_')
  })

  it('asks for TOKEN_BYTES of randomness', () => {
    let asked = 0
    generateToken((a) => {
      asked = a.length
      return a
    })
    expect(asked).toBe(TOKEN_BYTES)
  })

  it('uses the CSPRNG by default', () => {
    const a = generateToken()
    const b = generateToken()
    expect(a).toHaveLength(43)
    expect(a).not.toBe(b)
  })
})
