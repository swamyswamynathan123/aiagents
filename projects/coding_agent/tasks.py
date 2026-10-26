"""
Coding tasks: each is a known bug injected into the clean sample repo.

Because the correct code is known, every task is checkable: the visible tests must fail before
the agent starts, and both the visible and the hidden tests must pass after a correct fix.
"""
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE_REPO = HERE / "sample_repo"
HIDDEN_TESTS = HERE / "hidden_tests"

# task id -> request shown to the agent, and the mutations that create the bug
TASKS = {
    "tier-boundary": {
        "title": "Bulk discount starts one unit too late",
        "request": "Customers buying exactly 10 units aren't getting the bulk discount. The pricing tests are "
                   "failing; please fix the bug.",
        "mutations": [("shop/pricing.py", "if quantity >= min_qty:", "if quantity > min_qty:")],
    },
    "discount-percent": {
        "title": "Percentage discount is ten times too large",
        "request": "Cart totals with a percentage discount are wrong. Make the failing cart tests pass.",
        "mutations": [("shop/cart.py", "Decimal(str(discount_percent)) / 100", "Decimal(str(discount_percent)) / 10")],
    },
    "remove-exact": {
        "title": "Removing every unit leaves an empty line",
        "request": "If I remove all units of an item, it still shows up in the cart. Fix it.",
        "mutations": [("shop/cart.py", "if quantity is None or quantity >= current:",
                       "if quantity is None or quantity > current:")],
    },
    "rounding": {
        "title": "Money rounds halves to even",
        "request": "round_money(0.125) returns 0.12 but should return 0.13. Fix the rounding.",
        "mutations": [("shop/pricing.py", 'quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)', 'quantize(Decimal("0.01"))')],
    },
    "oversell": {
        "title": "Inventory ignores existing reservations",
        "request": "We're overselling: reserving stock doesn't account for units already reserved. "
                   "The inventory tests show the problem.",
        "mutations": [("shop/inventory.py", "if quantity > self.available(sku):",
                       "if quantity > self.stock.get(sku, 0):")],
    },
    "missing-clear": {
        "title": "Cart.clear() is missing",
        "request": "test_clear_empties_the_cart fails because Cart has no clear() method. Implement it.",
        "mutations": [("shop/cart.py", "    def clear(self) -> None:\n        self.items.clear()\n\n", "")],
    },
    "two-bugs": {
        "title": "Two bugs in different modules",
        "request": "Several tests are failing after a refactor. Find and fix everything that's broken.",
        "mutations": [("shop/pricing.py", "if quantity >= min_qty:", "if quantity > min_qty:"),
                      ("shop/inventory.py", "if quantity > self.available(sku):", "if quantity > self.stock.get(sku, 0):")],
    },
}


def make_workspace(task_id: str) -> Path:
    """A fresh copy of the sample repo with the task's bug applied (hidden tests are not copied)."""
    workspace = Path(tempfile.mkdtemp(prefix=f"coding-{task_id}-"))
    shutil.copytree(SAMPLE_REPO, workspace, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, correct, buggy in TASKS[task_id]["mutations"]:
        path = workspace / rel
        text = path.read_text(encoding="utf-8")
        assert text.count(correct) == 1, f"mutation anchor not unique in {rel}: {correct!r}"
        path.write_text(text.replace(correct, buggy), encoding="utf-8", newline="\n")
    return workspace


def with_hidden_tests(workspace: Path) -> Path:
    """A copy of the workspace plus the hidden tests, for evaluation only."""
    graded = Path(tempfile.mkdtemp(prefix="coding-graded-"))
    shutil.copytree(workspace, graded, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(HIDDEN_TESTS / "test_hidden.py", graded / "tests" / "test_hidden.py")
    return graded
