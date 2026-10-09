from decimal import Decimal

from billing.invoice import Invoice, compute_total


def test_compute_total_applies_discount_and_tax():
    inv = Invoice(customer_id="c1", region="US-NY")
    inv.add_item("sku-1", 2, "10.00")
    inv.apply_discount(50)
    assert compute_total(inv) == Decimal("10.40")


def test_apply_discount_clamps():
    inv = Invoice(customer_id="c1", region="US-NY")
    inv.apply_discount(150)
    assert inv.discount_pct == Decimal("100")
