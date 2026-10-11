from decimal import Decimal

from billing.tax import TaxTable, calculate_tax


def test_unknown_region_is_untaxed():
    assert TaxTable().lookup("XX") == Decimal("0")


def test_calculate_tax_california():
    assert calculate_tax(Decimal("100"), "US-CA") == Decimal("7.2500")
