from decimal import Decimal

import pytest

from shop.cart import Cart


def make_cart():
    cart = Cart()
    cart.add("mug", 2, "10.00")
    cart.add("tea", 1, "80.00")
    return cart


def test_subtotal():
    assert make_cart().subtotal() == Decimal("100.00")


def test_total_adds_tax():
    assert make_cart().total() == Decimal("108.00")


def test_total_applies_percentage_discount_before_tax():
    assert make_cart().total(discount_percent=10) == Decimal("97.20")


def test_remove_some_units():
    cart = make_cart()
    cart.remove("mug", 1)
    assert cart.items["mug"][0] == 1


def test_removing_all_units_removes_the_line():
    cart = make_cart()
    cart.remove("mug", 2)
    assert cart.skus() == ["tea"]


def test_clear_empties_the_cart():
    cart = make_cart()
    cart.clear()
    assert cart.skus() == [] and cart.subtotal() == Decimal("0.00")


def test_add_rejects_non_positive_quantity():
    with pytest.raises(ValueError):
        Cart().add("mug", 0, "1.00")
