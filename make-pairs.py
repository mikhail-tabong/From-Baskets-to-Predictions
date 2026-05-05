# make_pairs.py — extract positive/negative co-purchase pairs from cart data
import itertools, random, json, logging, pathlib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

DATA_RAW = pathlib.Path(__file__).parent / "data_raw"
OUT      = pathlib.Path(__file__).parent

def ordered_pairs(items):
    return list(itertools.permutations(items, 2))

carts    = json.load(open(DATA_RAW / "carts.json"))
products = json.load(open(DATA_RAW / "products.json"))
log.info("Loaded %d carts, %d products", len(carts), len(products))

all_product_ids = [p["id"] for p in products]
positive_pairs  = []
negative_pairs  = []

for cart in carts:
    item_ids = [p["productId"] for p in cart["products"]]
    if len(item_ids) < 2:
        continue
    positive_pairs.extend(ordered_pairs(item_ids))
    for A in item_ids:
        candidates    = list(set(all_product_ids) - set(item_ids))
        num_negatives = min(3, len(candidates))
        for B in random.sample(candidates, num_negatives):
            negative_pairs.append((A, B))

log.info("Raw pairs — positives: %d, negatives: %d", len(positive_pairs), len(negative_pairs))

num_samples = min(len(positive_pairs), len(negative_pairs))
df_pos = pd.DataFrame(positive_pairs, columns=["A", "B"]).sample(n=num_samples, random_state=42).assign(label=1)
df_neg = pd.DataFrame(random.sample(negative_pairs, num_samples), columns=["A", "B"]).assign(label=0)

df_all = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
df_all.to_csv(OUT / "pairs.csv", index=False)

log.info("✓ pairs.csv — %d rows (%d positive, %d negative)", len(df_all), num_samples, num_samples)
