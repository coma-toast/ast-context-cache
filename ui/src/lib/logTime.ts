const CLOCK_RE = /^\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2}:\d{2})/

/**
 * Shortens a log timestamp ("2026-10-05T12:34:56.789-05:00", or "2026-10-05 12:34:56") to its
 * wall-clock time as written, "12:34:56", so the server's zone is kept. Anything else is
 * returned unchanged.
 */
export const shortLogTime = (ts: string): string => CLOCK_RE.exec(ts.trim())?.[1] ?? ts
