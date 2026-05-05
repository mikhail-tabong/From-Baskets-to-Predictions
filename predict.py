"""
Inference: given a product A, recommend the top-N products most likely
to be co-purchased with it.

Usage
-----
  python predict.py --product_id 9
  python predict.py --product_id 9 --top_n 5
  python predict.py --list_products
"""
import argparse, json, pathlib
import pandas as pd
import numpy as np
import xgboost as xgb
from pandas import Timestamp
from collections import defaultdict

ROOT     = pathlib.Path(__file__).parent
DATA_RAW = ROOT / "data_raw"
MODELS   = ROOT / "models"

def load_artifacts():
    products = pd.json_normalize(json.load(open(DATA_RAW / "products.json"))).set_index("id")
    carts    = json.load(open(DATA_RAW / "carts.json"))

    co_occur = defaultdict(int)
    freq     = defaultdict(int)
    for cart in carts:
        ids = [p["productId"] for p in cart["products"]]
        for pid in ids:
            freq[pid] += 1
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = min(ids[i], ids[j]), max(ids[i], ids[j])
                co_occur[(a, b)] += 1

    model = xgb.XGBClassifier()
    model.load_model(MODELS / "co_purchase_model.json")
    return products, co_occur, freq, model


def build_features(A, B, products, co_occur, freq):
    today = Timestamp("today")
    pa, pb = products.loc[A], products.loc[B]
    price_a, price_b = pa["price"], pb["price"]
    pop_a,   pop_b   = pa["rating.count"], pb["rating.count"]
    age_a = (today - pd.to_datetime(pa["release_date"])).days
    age_b = (today - pd.to_datetime(pb["release_date"])).days
    key   = (min(A, B), max(A, B))
    return {
        "price_diff"      : abs(price_a - price_b),
        "price_ratio"     : min(price_a, price_b) / max(price_a, price_b) if max(price_a, price_b) > 0 else 1.0,
        "same_category"   : int(pa["category"] == pb["category"]),
        "rating_diff"     : abs(pa["rating.rate"] - pb["rating.rate"]),
        "popularity_A"    : pop_a,
        "popularity_B"    : pop_b,
        "popularity_ratio": min(pop_a, pop_b) / max(pop_a, pop_b) if max(pop_a, pop_b) > 0 else 1.0,
        "age_diff"        : abs(age_a - age_b),
        "co_occur_count"  : co_occur.get(key, 0),
        "freq_A"          : freq.get(A, 0),
        "freq_B"          : freq.get(B, 0),
    }


def recommend(product_id, top_n, products, co_occur, freq, model):
    candidates = [pid for pid in products.index if pid != product_id]
    rows = [build_features(product_id, B, products, co_occur, freq) for B in candidates]
    X_cand = pd.DataFrame(rows)
    probs  = model.predict_proba(X_cand)[:, 1]

    results = pd.DataFrame({
        "product_id"  : candidates,
        "title"       : [products.loc[b, "title"][:60] for b in candidates],
        "category"    : [products.loc[b, "category"] for b in candidates],
        "price"       : [products.loc[b, "price"] for b in candidates],
        "co_purchase_prob": probs,
    }).sort_values("co_purchase_prob", ascending=False).head(top_n).reset_index(drop=True)
    results.index += 1
    return results


def main():
    parser = argparse.ArgumentParser(description="Co-purchase recommendation engine")
    parser.add_argument("--product_id", type=int, help="Source product ID")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--list_products", action="store_true", help="List available products")
    args = parser.parse_args()

    products, co_occur, freq, model = load_artifacts()

    if args.list_products:
        print("\nAvailable products:")
        for pid, row in products.iterrows():
            print(f"  [{pid:2d}] ({row['category']}) {row['title'][:70]}")
        return

    if args.product_id is None:
        parser.print_help()
        return

    if args.product_id not in products.index:
        print(f"Product ID {args.product_id} not found. Use --list_products to see options.")
        return

    anchor = products.loc[args.product_id]
    print(f"\nAnchor product [{args.product_id}]: {anchor['title']}")
    print(f"Category: {anchor['category']} | Price: ${anchor['price']:.2f}")
    print(f"\nTop-{args.top_n} co-purchase recommendations:")
    recs = recommend(args.product_id, args.top_n, products, co_occur, freq, model)
    print(recs.to_string(index=True))


if __name__ == "__main__":
    main()
