"""A shopping cart."""
from decimal import Decimal

from .pricing import line_total, round_money

TAX_RATE = Decimal("0.08")


class Cart:
    def __init__(self):
        self.items = {}  # sku -> (quantity, unit_price)

    def add(self, sku: str, quantity: int, unit_price) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        current, _ = self.items.get(sku, (0, unit_price))
        self.items[sku] = (current + quantity, unit_price)

    def remove(self, sku: str, quantity: int = None) -> None:
        """Remove some units of a SKU, or the whole line if quantity is None or covers it all."""
        if sku not in self.items:
            raise KeyError(sku)
        current, price = self.items[sku]
        if quantity is None or quantity >= current:
            del self.items[sku]
        else:
            self.items[sku] = (current - quantity, price)

    def clear(self) -> None:
        self.items.clear()

    def skus(self) -> list:
        return sorted(self.items)

    def subtotal(self) -> Decimal:
        return round_money(sum((line_total(q, p) for q, p in self.items.values()), Decimal("0")))

    def total(self, discount_percent=0) -> Decimal:
        """Subtotal, minus a percentage discount, plus tax."""
        discounted = self.subtotal() * (1 - Decimal(str(discount_percent)) / 100)
        return round_money(discounted * (1 + TAX_RATE))
