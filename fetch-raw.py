# fetch_raw.py
import json, requests, pathlib
from datetime import date, timedelta

BASE = "https://fakestoreapi.com"
cache = pathlib.Path("data_raw")
cache.mkdir(exist_ok=True)

def fetch(route):
    r = requests.get(f"{BASE}/{route}", timeout=10)
    r.raise_for_status()
    return r.json()

with open(cache / "carts.json", "w") as f:
    json.dump(fetch("carts"), f, indent=2)

products = fetch("products")
for p in products:
    p["release_date"] = str(date(2023, 1, 1) + timedelta(days=p["id"] * 17))
with open(cache / "products.json", "w") as f:
    json.dump(products, f, indent=2)