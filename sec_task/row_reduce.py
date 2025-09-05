import math
from collections import Counter, defaultdict
import pandas as pd
from pathlib import Path

# ==== CONFIG ====
CSV_PATH = "../../../data.csv.zip"         
OUT_PATH = "stratified_data.parquet"
CHUNKSIZE = 250_000
STRATIFY_COLS = ["Label", "Traffic Type"]  
TARGET_TOTAL = 100_000                     


def row_key(df, cols):
    # returns a pandas Series of tuples, e.g. ("Malicious", "DoS")
    if len(cols) == 1:
        return df[cols[0]].astype("category")
    return pd.MultiIndex.from_frame(df[cols].astype("category")).to_series(index=df.index)

# ---- PASS 1: COUNT PER STRATUM ----
counts = Counter()
total_rows = 0

for chunk in pd.read_csv(CSV_PATH, usecols=STRATIFY_COLS, chunksize=CHUNKSIZE, low_memory=False, compression='zip'):
    k = row_key(chunk, STRATIFY_COLS)
    # k is a Series of keys; count efficiently
    vc = k.value_counts()
    counts.update(vc.to_dict())
    total_rows += len(chunk)

# ---- COMPUTE QUOTAS ----
# Raw quota (floors) + track remainders for fair rounding
quotas = {}
remainders = []

for k, c in counts.items():
    exact = TARGET_TOTAL * (c / total_rows)
    q = math.floor(exact)
    quotas[k] = q
    remainders.append((exact - q, k))

allocated = sum(quotas.values())
leftover = TARGET_TOTAL - allocated

# Distribute leftover to the largest remainders
remainders.sort(reverse=True)  # descending by fractional part
for frac, k in remainders[:leftover]:
    quotas[k] += 1

# Guard: if some strata have fewer available rows than quota (rare), clamp them
for k, c in counts.items():
    if quotas[k] > c:
        quotas[k] = c

# ---- PASS 2: COLLECT ----
taken = defaultdict(int)
parts = []

for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE, low_memory=False, compression='zip'):
   
   # shuffle rows in this chunk
    chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
    k = row_key(chunk, STRATIFY_COLS)

    mask = []
    for key in k:
        if taken[key] < quotas.get(key, 0):
            mask.append(True)
            taken[key] += 1
        else:
            mask.append(False)
    kept = chunk.loc[mask]
    if not kept.empty:
        parts.append(kept)

    # Early exit if all quotas are met
    if all(taken[k] >= quotas[k] for k in quotas):
        break

sampled = pd.concat(parts, ignore_index=True)
# Save to Parquet (fast, compressed). Use columns=... to drop heavy, unneeded columns if you wish.
sampled_cols = None  # or a list of columns you want to keep
sampled.to_parquet(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Final size:", len(sampled))

# ---- QUICK CHECK: compare distributions ----
def dist(series):
    vc = series.value_counts(normalize=True).sort_index()
    return vc

print("\nDistribution in sample:")
print(dist(sampled["Label"]))

if "Traffic Type" in sampled.columns:
    print("\nTraffic Type in sample (within Label):")
    print(sampled.groupby("Label")["Traffic Type"].value_counts(normalize=True))
