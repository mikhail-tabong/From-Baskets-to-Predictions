"""
Feature engineering: build the feature matrix X from pairs.csv + raw data.

Features per (A, B) product pair
---------------------------------
price_diff          Absolute price difference
price_ratio         min/max price (0-1; 1 = same price)
same_category       1 if both products share the same category
rating_diff         Absolute difference in average ratings
popularity_A        Rating count of product A (proxy for demand)
popularity_B        Rating count of product B
popularity_ratio    min/max popularity (0-1)
age_diff            Absolute difference in days since release
co_occur_count      How many carts contain *both* A and B (from full cart log)
freq_A              How often product A appears across all carts
freq_B              How often product B appears across all carts
"""
import json, pathlib
import pandas as pd
import numpy as np
from pandas import Timestamp
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent
DATA_RAW = ROOT / "data_raw"

# Load data
pairs    = pd.read_csv(ROOT / "pairs.csv")
products = pd.json_normalize(json.load(open(DATA_RAW / "products.json"))).set_index("id")
carts    = json.load(open(DATA_RAW / "carts.json"))

# Build co-occurrence and frequency maps from the full cart log
co_occur  = defaultdict(int)
freq      = defaultdict(int)

for cart in carts:
    ids = [p["productId"] for p in cart["products"]]
    for pid in ids:
        freq[pid] += 1
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = min(ids[i], ids[j]), max(ids[i], ids[j])
            co_occur[(a, b)] += 1

today = Timestamp("today")

def create_features(row):
    A, B = int(row["A"]), int(row["B"])
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

X = pairs.apply(create_features, axis=1, result_type="expand")
y = pairs["label"]

X.to_parquet(ROOT / "X.parquet")
y.to_csv(ROOT / "y.csv", index=False)

print(f"✓ Features saved — {X.shape[0]} rows × {X.shape[1]} features")
print(X.describe().round(3).to_string())
