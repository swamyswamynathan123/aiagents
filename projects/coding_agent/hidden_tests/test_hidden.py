"""Hidden tests: never shown to the agent. They check the same behaviours with different inputs,
so a fix that special-cases the visible tests (e.g. `if quantity == 10`) fails here."""
from decimal import Decimal

import pytest

from shop.cart import Cart
from shop.inventory import Inventory, OutOfStock
from shop.pricing import bulk_discount_rate, line_total, round_money


@pytest.mark.parametrize("qty, rate", [(10, "0.05"), (49, "0.05"), (50, "0.10"), (99, "0.10"), (100, "0.15")])
def test_every_tier_boundary(qty, rate):
    assert bulk_discount_rate(qty) == Decimal(rate)


def test_line_total_at_fifty():
    assert line_total(50, "1.00") == Decimal("45.00")


@pytest.mark.parametrize("value, expected", [(1.005, "1.01"), (0.335, "0.34"), (7.0, "7.00")])
def test_round_half_up_generally(value, expected):
    assert round_money(value) == Decimal(expected)


@pytest.mark.parametrize("percent, expected", [(0, "54.00"), (25, "40.50"), (50, "27.00")])
def test_discount_percentages(percent, expected):
    cart = Cart()
    cart.add("book", 1, "50.00")
    assert cart.total(discount_percent=percent) == Decimal(expected)


def test_remove_exact_quantity_of_any_line():
    cart = Cart()
    cart.add("pen", 7, "1.00")
    cart.remove("pen", 7)
    assert "pen" not in cart.items


def test_clear_then_reuse():
    cart = Cart()
    cart.add("pen", 1, "1.00")
    cart.clear()
    cart.add("ink", 2, "3.00")
    assert cart.skus() == ["ink"]


def test_reservations_accumulate_across_calls():
    inv = Inventory({"lamp": 10})
    for _ in range(3):
        inv.reserve("lamp", 3)
    with pytest.raises(OutOfStock):
        inv.reserve("lamp", 2)
    assert inv.available("lamp") == 1
