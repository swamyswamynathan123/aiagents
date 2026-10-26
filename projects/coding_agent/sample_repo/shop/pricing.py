"""Money rounding and bulk discounts."""
from decimal import ROUND_HALF_UP, Decimal

# (minimum quantity, discount rate), checked from the largest tier down
BULK_TIERS = [(100, Decimal("0.15")), (50, Decimal("0.10")), (10, Decimal("0.05"))]


def round_money(value) -> Decimal:
    """Round to cents, with halves rounded up (0.125 -> 0.13)."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def bulk_discount_rate(quantity: int) -> Decimal:
    """Discount for buying `quantity` units: 5% from 10, 10% from 50, 15% from 100."""
    for min_qty, rate in BULK_TIERS:
        if quantity >= min_qty:
            return rate
    return Decimal("0")


def line_total(quantity: int, unit_price) -> Decimal:
    """Price of one cart line after the bulk discount."""
    price = Decimal(str(unit_price))
    return round_money(price * quantity * (1 - bulk_discount_rate(quantity)))
