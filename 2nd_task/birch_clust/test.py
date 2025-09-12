# test_birch_results.py
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# --- paths (edit if yours differ) ---
OUT_REDUCED = Path("merged_reduced_birch_half.parquet")
OUT_STATS   = Path("merged_reduced_birch_stats.csv")

# --- basic existence checks ---
if not OUT_REDUCED.exists():
    raise FileNotFoundError(f"Reduced parquet not found: {OUT_REDUCED.resolve()}")
print("Found reduced file:", OUT_REDUCED)

# --- count rows & list columns without loading all data ---
dataset = ds.dataset(str(OUT_REDUCED), format="parquet")
n_rows = dataset.count_rows()
schema = dataset.schema
cols = [f.name for f in schema]
print(f"Reduced rows: {n_rows:,}, columns: {len(cols)}")

# --- cluster counts in reduced file (streamed) ---
cluster_col = "birch_cluster"
if cluster_col not in cols:
    raise ValueError(f"Column '{cluster_col}' not found in reduced file.")

counts = {}
scanner = dataset.scanner(columns=[cluster_col], batch_size=100_000, use_threads=True)
for rb in scanner.to_reader():
    s = rb.column(0).to_numpy()
    uniq, cnt = np.unique(s, return_counts=True)
    for u, c in zip(uniq, cnt):
        counts[int(u)] = counts.get(int(u), 0) + int(c)

reduced_counts = pd.Series(counts, name="kept_count").sort_index()
print("Clusters in reduced:", len(reduced_counts))
print(reduced_counts.head())

# --- compare with full-cluster counts if stats CSV exists ---
if OUT_STATS.exists():
    stats = pd.read_csv(OUT_STATS)
    if "birch_cluster" in stats.columns and "count" in stats.columns:
        stats = stats.set_index("birch_cluster").sort_index()
        joined = stats.join(reduced_counts, how="left").fillna(0)
        joined["kept_ratio"] = joined["kept_count"] / joined["count"].clip(lower=1)
        print("\n~50% keep check (min/median/max ratio):",
              f"{joined['kept_ratio'].min():.3f} / {joined['kept_ratio'].median():.3f} / {joined['kept_ratio'].max():.3f}")
        print(joined.head())
    else:
        print("Stats CSV found but missing columns 'birch_cluster'/'count'.")
else:
    print("Stats CSV not found (skip 50% check).")

# --- pick two numeric columns for a quick scatter (small sampled read) ---
numeric_cols = []
for f in schema:
    if pa.types.is_integer(f.type) or pa.types.is_floating(f.type):
        if f.name != cluster_col:
            numeric_cols.append(f.name)

if len(numeric_cols) >= 2:
    xcol, ycol = numeric_cols[0], numeric_cols[1]
    want = [xcol, ycol, cluster_col]
    xs, ys, cs = [], [], []

    # read up to ~100k points for plotting
    limit = 100_000
    taken = 0
    scanner = dataset.scanner(columns=want, batch_size=50_000, use_threads=True)
    for rb in scanner.to_reader():
        df = rb.to_pandas()
        need = max(0, limit - taken)
        if need <= 0:
            break
        if len(df) > need:
            df = df.sample(n=need, random_state=42)
        xs.append(df[xcol].astype("float32"))
        ys.append(df[ycol].astype("float32"))
        cs.append(df[cluster_col].astype("int32"))
        taken += len(df)

    if taken > 0:
        import matplotlib
        plt.figure(figsize=(9, 6))
        plt.scatter(pd.concat(xs), pd.concat(ys), c=pd.concat(cs), s=2, alpha=0.6)
        plt.xlabel(xcol); plt.ylabel(ycol)
        plt.title(f"Birch reduced (n={taken:,})")
        plt.tight_layout()
    else:
        print("Could not collect enough rows for plotting.")
else:
    print("Not enough numeric columns for scatter.")
