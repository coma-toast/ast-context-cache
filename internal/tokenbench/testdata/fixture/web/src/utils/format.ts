/** formatCurrency renders cents as a localized currency string. */
export function formatCurrency(cents: number, currency = "USD", locale = "en-US"): string {
  return new Intl.NumberFormat(locale, { style: "currency", currency }).format(cents / 100);
}

/** formatDate renders an ISO date for order history. */
export function formatDate(iso: string, locale = "en-US"): string {
  return new Date(iso).toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" });
}

export function pluralize(n: number, word: string): string {
  return n === 1 ? word : `${word}s`;
}
