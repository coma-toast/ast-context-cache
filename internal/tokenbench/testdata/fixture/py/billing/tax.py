"""Regional sales tax."""

from decimal import Decimal


class TaxTable:
    """Sales tax rates keyed by region code."""

    def __init__(self, rates=None):
        self.rates = rates or {"US-CA": Decimal("0.0725"), "US-NY": Decimal("0.04"), "EU-DE": Decimal("0.19")}

    def lookup(self, region):
        """Rate for region, or zero when the region is untaxed."""
        return self.rates.get(region, Decimal("0"))


DEFAULT_TABLE = TaxTable()


def calculate_tax(amount, region, table=DEFAULT_TABLE):
    """Sales tax owed on amount in region."""
    return amount * table.lookup(region)
