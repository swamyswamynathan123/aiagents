from decimal import Decimal

from shop.pricing import bulk_discount_rate, line_total, round_money


def test_round_money_rounds_halves_up():
    assert round_money(0.125) == Decimal("0.13")
    assert round_money(2.5) == Decimal("2.50")


def test_no_discount_below_ten():
    assert bulk_discount_rate(9) == Decimal("0")


def test_discount_starts_at_ten_units():
    assert bulk_discount_rate(10) == Decimal("0.05")
    assert line_total(10, "2.00") == Decimal("19.00")


def test_largest_tier():
    assert bulk_discount_rate(150) == Decimal("0.15")
