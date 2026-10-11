"""Money conversions between cents and decimals."""

from decimal import ROUND_HALF_EVEN, Decimal


def to_cents(amount):
    """Convert a Decimal amount to integer cents with banker's rounding."""
    return int((Decimal(amount) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def from_cents(cents):
    """Convert integer cents to a Decimal amount."""
    return Decimal(cents) / 100
