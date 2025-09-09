from pathlib import Path
from collections import Counter, defaultdict
import math, random
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import pyarrow.parquet as pq

# ---- CONFIG ----
PARQUET_PATH = "../init_compr/tii_ssrc23_parquet"   # file.parquet OR a folder with parquet parts
STRATIFY_COLS = ["Label", "Traffic Type"]   # adjust if your names differ
TARGET_TOTAL  = 100_000               # your target rows
BATCH_SIZE    = 500_000               # how many rows per streamed batch (tune to RAM)

# make per-row stratum keys 
def row_key(df: pd.DataFrame, cols):
    if len(cols) == 1:
        return df[cols[0]].astype("category")
    return pd.MultiIndex.from_frame(df[cols].astype("category")).to_series(index=df.index)

# Build a streaming dataset (works for single file or partitioned folder)
dataset = ds.dataset(PARQUET_PATH, format="parquet", partitioning="hive")


# ---- PASS 1: COUNT PER STRATUM ----
counts = Counter()
total_rows = 0
batch_count = 0

# Scanner that only materializes the columns we need
scanner = ds.Scanner.from_dataset(
    dataset,
    columns=STRATIFY_COLS,   # only Label / Traffic Type
    batch_size=BATCH_SIZE
)

for batch_count, batch in enumerate(scanner.to_batches(), start=1):
    ch = batch.to_pandas(types_mapper=pd.ArrowDtype)
    k  = row_key(ch, STRATIFY_COLS)
    counts.update(k.value_counts().to_dict())
    total_rows += len(ch)
    if batch_count % 5 == 0:
        print(f"[P1 batch {batch_count}] cumulative rows={total_rows:,}")

print(f"PASS 1 finished. Batches={batch_count}, total_rows={total_rows:,}")
print("Arrow dataset rows:", dataset.count_rows())

# Compute proportional quotas + largest-remainder rounding
quotas = {}
remainders = []
for k_, c in counts.items():
    exact = TARGET_TOTAL * (c / total_rows)
    q = math.floor(exact)
    quotas[k_] = q
    remainders.append((exact - q, k_))

allocated = sum(quotas.values())
leftover = TARGET_TOTAL - allocated
remainders.sort(reverse=True)
for _, k_ in remainders[:leftover]:
    quotas[k_] += 1

# Defensive clamp (rare)
for k_, c in counts.items():
    if quotas[k_] > c:
        quotas[k_] = c


#---- PASS 2: COLLECT ----

parts_dir = Path("tii_ssrc23_stratified_parts")
parts_dir.mkdir(exist_ok=True)

taken  = defaultdict(int)
part_i = 0
rows_seen = 0
rows_kept = 0

# Full columns this time (we want to write real rows out)
scanner2 = ds.Scanner.from_dataset(
    dataset,
    batch_size=BATCH_SIZE
)

for batch_idx, batch in enumerate(scanner2.to_batches(), start=1):
    df = batch.to_pandas(types_mapper=pd.ArrowDtype)
    rows_seen += len(df)

    # Shuffle inside batch
    df = df.sample(frac=1, random_state=94).reset_index(drop=True)

    # Stratum keys after shuffle
    k_series = row_key(df, STRATIFY_COLS)

    mask = []
    get_quota = quotas.get
    for key in k_series:
        if taken[key] < get_quota(key, 0):
            mask.append(True)
            taken[key] += 1
        else:
            mask.append(False)

    kept = df.loc[mask]
    if not kept.empty:
        rows_kept += len(kept)
        kept.to_parquet(parts_dir / f"part_{part_i:05d}.parquet", index=False)
        part_i += 1

    # ---- LOG MESSAGE ----
    print(f"[Batch {batch_idx}] rows_seen={rows_seen:,} | rows_kept={rows_kept:,} | parts={part_i}")

    # Early exit if quotas are satisfied
    if all(taken[k_] >= quotas[k_] for k_ in quotas):
        print("All quotas satisfied, stopping early.")
        break

print(f"Finished Pass 2. Total rows_seen={rows_seen:,}, rows_kept={rows_kept:,}, parts written={part_i}")


# Concatenate parts into a single parquet (optional; you can also keep the folder)
parts = sorted(Path("tii_ssrc23_stratified_parts").glob("part_*.parquet"))
sampled = pd.concat(
    (pd.read_parquet(p, engine="pyarrow", dtype_backend="pyarrow") for p in parts),
    ignore_index=True
)
sampled.to_parquet("stratified.parquet", index=False)

print("Saved: stratified.parquet")
print("Rows:", len(sampled))

# ---- Sanity checks (distributions) ----
def dist(series: pd.Series):
    return series.value_counts(normalize=True).sort_index()

print("\nDistribution in sample (Label):")
print(dist(sampled["Label"]))

if "Traffic Type" in sampled.columns:
    print("\nTraffic Type in sample (within Label):")
    print(sampled.groupby("Label")["Traffic Type"].value_counts(normalize=True))
