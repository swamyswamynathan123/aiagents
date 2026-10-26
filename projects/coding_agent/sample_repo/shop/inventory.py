"""Stock levels with reservations, so the same unit can't be sold twice."""


class OutOfStock(Exception):
    pass


class Inventory:
    def __init__(self, stock: dict):
        self.stock = dict(stock)
        self.reserved = {}

    def available(self, sku: str) -> int:
        return self.stock.get(sku, 0) - self.reserved.get(sku, 0)

    def reserve(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.available(sku):
            raise OutOfStock(sku)
        self.reserved[sku] = self.reserved.get(sku, 0) + quantity

    def release(self, sku: str, quantity: int) -> None:
        self.reserved[sku] = max(0, self.reserved.get(sku, 0) - quantity)
