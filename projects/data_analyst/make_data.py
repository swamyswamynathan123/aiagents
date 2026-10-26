"""
Generate the sample dataset (data/sales_2025.csv): one year of retail orders.
Deterministic (fixed seed), so evals can compute exact ground-truth answers.
Run from the repo root:  python -m projects.data_analyst.make_data
"""
import random
from datetime import date, timedelta
from pathlib import Path

PRODUCTS = {
    "Electronics": {"Laptop": 1200.0, "Headphones": 150.0, "Monitor": 300.0},
    "Home": {"Blender": 80.0, "Vacuum": 250.0, "Desk Lamp": 40.0},
    "Outdoor": {"Tent": 220.0, "Backpack": 90.0, "Bike": 650.0},
}
REGIONS = {"North": 1.0, "South": 1.25, "East": 0.9, "West": 1.1}  # demand multipliers
SEASON = {"Electronics": {11: 1.8, 12: 2.2}, "Outdoor": {5: 1.6, 6: 1.9, 7: 1.8}, "Home": {1: 1.3, 12: 1.2}}

OUT = Path(__file__).parent / "data" / "sales_2025.csv"


def generate(n_orders: int = 1500, seed: int = 7) -> list:
    rng = random.Random(seed)
    rows, start = [], date(2025, 1, 1)
    flat = [(c, p, price) for c, items in PRODUCTS.items() for p, price in items.items()]
    while len(rows) < n_orders:
        day = start + timedelta(days=rng.randrange(365))
        category, product, price = rng.choice(flat)
        region = rng.choice(list(REGIONS))
        weight = REGIONS[region] * SEASON.get(category, {}).get(day.month, 1.0)
        if rng.random() > weight / 2.2:  # rejection sampling shapes regional and seasonal demand
            continue
        channel = "Online" if rng.random() < 0.6 else "Store"
        discount = rng.choice([0.0, 0.0, 0.0, 0.1, 0.2] if channel == "Online" else [0.0, 0.0, 0.0, 0.0, 0.1])
        units = rng.choice([1, 1, 1, 2, 2, 3, 4]) if price < 500 else rng.choice([1, 1, 1, 2])
        revenue = round(units * price * (1 - discount), 2)
        rows.append((day.isoformat(), region, category, product, channel, units, price, discount, revenue))
    rows.sort()
    return [(f"O-{i:05d}", *r) for i, r in enumerate(rows, 1)]


def main():
    OUT.parent.mkdir(exist_ok=True)
    header = "order_id,date,region,category,product,channel,units,unit_price,discount,revenue"
    lines = [header] + [",".join(str(v) for v in r) for r in generate()]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines) - 1} orders to {OUT}")


if __name__ == "__main__":
    main()
