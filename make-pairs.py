import itertools, random, json, pandas as pd

def ordered_pairs(items):
    return list(itertools.permutations(items, 2))

# Load data
carts = json.load(open("/Users/kyle/Desktop/Current Working Directory/Module6Assignment/data_raw/carts.json"))
products = json.load(open("/Users/kyle/Desktop/Current Working Directory/Module6Assignment/data_raw/products.json"))
all_product_ids = [p["id"] for p in products]

positive_pairs = []
negative_pairs = []

for cart in carts:
    item_ids = [p["productId"] for p in cart["products"]]
    if len(item_ids) < 2:
        continue

    # Positive (A, B) pairs within the same cart
    positive_pairs.extend(ordered_pairs(item_ids))

    # For each A, generate up to 3 negatives
    for A in item_ids:
        candidates = list(set(all_product_ids) - set(item_ids))
        num_negatives = min(3, len(candidates))  # up to 3 negatives per A
        sampled_Bs = random.sample(candidates, num_negatives)
        for B in sampled_Bs:
            negative_pairs.append((A, B))

# Balance the dataset: match number of negatives to positives
num_samples = min(len(positive_pairs), len(negative_pairs))
df_pos = pd.DataFrame(positive_pairs, columns=["A", "B"]).sample(n=num_samples, random_state=42).assign(label=1)
df_neg = pd.DataFrame(random.sample(negative_pairs, num_samples), columns=["A", "B"]).assign(label=0)

# Combine and shuffle
df_all = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
df_all.to_csv("pairs.csv", index=False)

print(f"✓ Created {len(df_all)} labeled pairs ({num_samples} positives, {num_samples} negatives)")