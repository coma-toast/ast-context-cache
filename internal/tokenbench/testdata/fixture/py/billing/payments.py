"""Payment capture and refunds through an external gateway."""


class PaymentError(Exception):
    """Raised when the gateway declines a charge."""


class PaymentGateway:
    """Thin client for the card processor."""

    def __init__(self, api_key, timeout=10):
        self.api_key = api_key
        self.timeout = timeout

    def charge(self, customer_id, amount_cents, idempotency_key):
        """Capture amount_cents from the customer's saved card."""
        if amount_cents <= 0:
            raise PaymentError("amount must be positive")
        return {"customer": customer_id, "amount": amount_cents, "key": idempotency_key}

    def refund(self, charge_id, amount_cents=None):
        """Refund all or part of a previous charge."""
        return {"charge": charge_id, "amount": amount_cents}
