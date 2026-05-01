"""
Augment the raw data with realistic synthetic shopping carts.

The FakeStore API only returns 7 carts. This script generates hundreds of
additional carts using a weighted sampling strategy that mirrors real purchase
behavior: items in the same category and similar price range are more likely
to appear together.
"""
import json, pathlib, random
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_RAW = pathlib.Path(__file__).parent / "data_raw"

products = json.load(open(DATA_RAW / "products.json"))
real_carts = json.load(open(DATA_RAW / "carts.json"))

product_ids = [p["id"] for p in products]
categories = list({p["category"] for p in products})
prod_by_cat = {c: [p["id"] for p in products if p["category"] == c] for c in categories}
popularity = {p["id"]: p["rating"]["count"] for p in products}

def popularity_weights(ids):
    w = np.array([popularity[i] for i in ids], dtype=float)
    return w / w.sum()

def make_cart(cart_id, user_id, date_str, n_items):
    """
    Build one cart.  With 70 % probability pick a category-focused cart
    (realistic cross-sell within a category), otherwise a mixed cart.
    """
    if random.random() < 0.70:
        # category-focused: anchor on one category, allow 1 cross-category item
        anchor_cat = random.choice(categories)
        cat_ids = prod_by_cat[anchor_cat]
        k_cat = min(n_items, len(cat_ids))
        chosen = list(np.random.choice(
            cat_ids, size=k_cat, replace=False,
            p=popularity_weights(cat_ids)
        ))
        # optionally add one item from a different category
        if n_items > k_cat:
            other_ids = [i for i in product_ids if i not in chosen]
            if other_ids:
                extra = np.random.choice(
                    other_ids, size=1,
                    p=popularity_weights(other_ids)
                )
                chosen += list(extra)
    else:
        # fully mixed cart
        k = min(n_items, len(product_ids))
        chosen = list(np.random.choice(
            product_ids, size=k, replace=False,
            p=popularity_weights(product_ids)
        ))

    return {
        "id": cart_id,
        "userId": user_id,
        "date": date_str,
        "products": [
            {"productId": int(pid), "quantity": int(np.random.randint(1, 5))}
            for pid in chosen
        ],
        "__v": 0,
    }

# Generate dates across 2020-2023
from datetime import date, timedelta
start = date(2020, 1, 1)
date_range = [(start + timedelta(days=i)).isoformat() + "T00:00:00.000Z"
              for i in range(0, 365 * 4, 2)]

n_synthetic = 500
synthetic_carts = []
max_real_id = max(c["id"] for c in real_carts)
user_pool = list(range(1, 51))   # 50 synthetic users

for i in range(n_synthetic):
    cart_id   = max_real_id + i + 1
    user_id   = random.choice(user_pool)
    date_str  = random.choice(date_range)
    n_items   = random.choices([2, 3, 4, 5, 6], weights=[30, 30, 20, 12, 8])[0]
    synthetic_carts.append(make_cart(cart_id, user_id, date_str, n_items))

all_carts = real_carts + synthetic_carts
with open(DATA_RAW / "carts.json", "w") as f:
    json.dump(all_carts, f, indent=2)

print(f"✓ carts.json now has {len(all_carts)} carts "
      f"({len(real_carts)} real + {len(synthetic_carts)} synthetic)")
