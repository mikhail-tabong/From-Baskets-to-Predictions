# fetch_raw.py — fetch product and cart data from the FakeStore API
import json, logging, requests, pathlib
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE  = "https://fakestoreapi.com"
cache = pathlib.Path(__file__).parent / "data_raw"
cache.mkdir(exist_ok=True)

def fetch(route):
    url = f"{BASE}/{route}"
    log.info("GET %s", url)
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

carts = fetch("carts")
with open(cache / "carts.json", "w") as f:
    json.dump(carts, f, indent=2)
log.info("Saved %d carts → data_raw/carts.json", len(carts))

products = fetch("products")
for p in products:
    p["release_date"] = str(date(2023, 1, 1) + timedelta(days=p["id"] * 17))
with open(cache / "products.json", "w") as f:
    json.dump(products, f, indent=2)
log.info("Saved %d products → data_raw/products.json", len(products))
