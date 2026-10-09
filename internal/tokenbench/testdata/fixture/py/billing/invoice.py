"""Invoices and their totals."""

from dataclasses import dataclass, field
from decimal import Decimal

from .tax import calculate_tax


@dataclass
class LineItem:
    sku: str
    quantity: int
    unit_price: Decimal


@dataclass
class Invoice:
    customer_id: str
    region: str
    items: list = field(default_factory=list)
    discount_pct: Decimal = Decimal("0")

    def add_item(self, sku, quantity, unit_price):
        """Append a line item to the invoice."""
        self.items.append(LineItem(sku, quantity, Decimal(unit_price)))

    def subtotal(self):
        """Sum of line items before discount and tax."""
        return sum((i.unit_price * i.quantity for i in self.items), Decimal("0"))

    def apply_discount(self, pct):
        """Set a percentage discount, clamped to 0..100."""
        self.discount_pct = max(Decimal("0"), min(Decimal("100"), Decimal(pct)))


def compute_total(invoice):
    """Subtotal minus discount plus regional sales tax, rounded to cents."""
    discounted = invoice.subtotal() * (1 - invoice.discount_pct / 100)
    return (discounted + calculate_tax(discounted, invoice.region)).quantize(Decimal("0.01"))
