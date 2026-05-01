import json, pathlib
import pandas as pd
from pandas import Timestamp

ROOT = pathlib.Path(__file__).parent
DATA_RAW = ROOT / "data_raw"

# Load data
pairs = pd.read_csv(ROOT / "pairs.csv")
products = pd.json_normalize(json.load(open(DATA_RAW / "products.json"))).set_index("id")

def create_features(row):
    A, B = row["A"], row["B"]
    pa, pb = products.loc[A], products.loc[B]
    return {
        "price_diff"   : abs(pa["price"] - pb["price"]),
        "same_category": int(pa["category"] == pb["category"]),
        "rating_B"     : pb["rating.rate"],
        "popularity_B" : pb["rating.count"],
        "age_B"        : (Timestamp("today") - pd.to_datetime(pb["release_date"])).days
    }

# Apply features to each row
X = pairs.apply(create_features, axis=1, result_type="expand")
y = pairs["label"]

X.to_parquet(ROOT / "X.parquet")
y.to_csv(ROOT / "y.csv", index=False)
print("✓ Features saved to X.parquet and y.csv")
