const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

/** isValidEmail checks the shape of an email address before login. */
export function isValidEmail(email: string): boolean {
  return EMAIL.test(email);
}

/** clampQuantity keeps a cart quantity between 1 and max. */
export function clampQuantity(n: number, max = 99): number {
  return Math.max(1, Math.min(max, Math.floor(n)));
}
