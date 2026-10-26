import pytest

from shop.inventory import Inventory, OutOfStock


def test_reserve_reduces_availability():
    inv = Inventory({"mug": 5})
    inv.reserve("mug", 3)
    assert inv.available("mug") == 2


def test_cannot_reserve_more_than_available():
    inv = Inventory({"mug": 5})
    inv.reserve("mug", 3)
    with pytest.raises(OutOfStock):
        inv.reserve("mug", 3)


def test_release_returns_units():
    inv = Inventory({"mug": 5})
    inv.reserve("mug", 4)
    inv.release("mug", 2)
    assert inv.available("mug") == 3
