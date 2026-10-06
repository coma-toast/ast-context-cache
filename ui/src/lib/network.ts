import type { NetworkKey, NetworkListener, NetworkState } from '../api/types'

/** Env var that overrides (and locks) each network setting, mirroring `internal/netlisten`. */
export const NETWORK_ENV: Record<NetworkKey, string> = {
  listen_extra_addrs: 'AST_LISTEN_EXTRA',
  trusted_hosts: 'AST_TRUSTED_HOSTS',
  remote_access_token: 'AST_ACCESS_TOKEN',
}

/** Bytes of randomness in a generated access token (43 base64url characters). */
export const TOKEN_BYTES = 32

const IPV4 = /^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}$/
const HOST_LABEL = /^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$/
const MAX_HOSTNAME = 253

/** Splits a comma, newline or space separated list, dropping blanks and repeats. */
export const parseList = (text: string): string[] => [...new Set(text.split(/[\s,]+/).filter(Boolean))]

const stripBrackets = (s: string) => s.replace(/^\[/, '').replace(/\]$/, '')

/** Canonical IPv6 text (lowercase, zeros compressed) as the URL parser writes it, or '' if invalid. */
const canonicalIPv6 = (s: string): string => {
  if (!s.includes(':')) return ''
  try {
    return stripBrackets(new URL(`http://[${s}]/`).hostname)
  } catch {
    return ''
  }
}

export const isIPv4 = (s: string): boolean => IPV4.test(s)

export const isIPAddress = (s: string): boolean => isIPv4(stripBrackets(s)) || canonicalIPv6(stripBrackets(s)) !== ''

const isWildcardIP = (ip: string): boolean => ip === '0.0.0.0' || canonicalIPv6(ip) === '::'

const isLoopbackIP = (ip: string): boolean => (isIPv4(ip) && ip.startsWith('127.')) || canonicalIPv6(ip) === '::1'

/** RFC 1123 hostname: dot-separated labels of letters, digits and inner hyphens; a trailing dot is fine. */
export const isHostname = (s: string): boolean => {
  const h = normalizeHost(s)
  return h !== '' && h.length <= MAX_HOSTNAME && h.split('.').every((label) => HOST_LABEL.test(label))
}

export const normalizeHost = (s: string): string => s.toLowerCase().replace(/\.$/, '')

/**
 * Checks extra listen addresses the way the server does: IPs only, no wildcard (use --listen for
 * that) and no loopback (always listened on). Returns the first problem, or '' when all are valid.
 */
export const validateAddrs = (list: string[]): string => {
  for (const entry of list) {
    const ip = stripBrackets(entry)
    if (!isIPAddress(ip)) return `"${entry}" is not an IP address (put hostnames in trusted hostnames)`
    if (isWildcardIP(ip)) return `"${entry}" is a wildcard; use --listen / AST_LISTEN to bind every interface`
    if (isLoopbackIP(ip)) return `"${entry}" is loopback, which is always listened on`
  }
  return ''
}

/** Checks trusted hostnames (IPs are accepted too, wildcards are not). Returns the first problem or ''. */
export const validateHosts = (list: string[]): string => {
  for (const entry of list) {
    const ip = stripBrackets(entry)
    if (isIPAddress(ip)) {
      if (isWildcardIP(ip)) return `"${entry}" is a wildcard`
      continue
    }
    if (!isHostname(entry)) return `"${entry}" is not a valid hostname`
  }
  return ''
}

/** host:port with IPv6 brackets. */
export const formatHostPort = (addr: string, port: number): string =>
  addr.includes(':') ? `[${addr}]:${port}` : `${addr}:${port}`

export interface ListenerChip {
  key: string
  label: string
  color: 'success' | 'error'
  tooltip: string
}

/** One status chip per extra listener: which server, where, and whether the bind worked. */
export const listenerChips = (listeners: NetworkListener[] | null | undefined): ListenerChip[] =>
  (listeners ?? []).map((l) => {
    const ok = l.status === 'listening'
    const where = formatHostPort(l.addr, l.port)
    return {
      key: `${l.server}-${l.addr}`,
      label: `${l.server} ${where}`,
      color: ok ? 'success' : 'error',
      tooltip: ok ? `Listening on ${where}` : l.error || `Not listening on ${where}`,
    }
  })

/** Why a setting is read-only here, or '' when the dashboard can change it. */
export const networkLockReason = (state: NetworkState | null, key: NetworkKey): string => {
  const env = state?.locked?.[key]
  return env ? `Set by ${env} in the environment` : ''
}

/**
 * True when the server is reachable beyond loopback (extra addresses, or a wildcard base) without
 * a token, so anyone who can reach it can use it.
 */
export const needsTokenWarning = (state: NetworkState | null): boolean =>
  !!state && !state.token_set && ((state.listen_extra_addrs?.length ?? 0) > 0 || state.base_wildcard)

const toBase64Url = (bytes: Uint8Array): string =>
  btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

/** A random access token: TOKEN_BYTES from the CSPRNG, base64url without padding. */
export const generateToken = (
  fill: (a: Uint8Array<ArrayBuffer>) => Uint8Array = (a) => crypto.getRandomValues(a),
): string =>
  toBase64Url(fill(new Uint8Array(TOKEN_BYTES)))
