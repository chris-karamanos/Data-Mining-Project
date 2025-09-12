# birch_centroids.py
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

# ----------------- config -----------------
IN_PATH = "../../1st_task/data/reduced/merged_reduced.parquet"
OUT_CENTROIDS = "data/merged_reduced_birch_centroids.parquet"
OUT_STATS     = "data/merged_reduced_birch_stats.csv"

CHUNK_ROWS = 20_000                  
BIRCH_THRESHOLD = 3.0           
BRANCHING = 100
RANDOM_STATE = 42

# ------------------------------------------
Path(OUT_CENTROIDS).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_STATS).parent.mkdir(parents=True, exist_ok=True)

def iter_batches(dataset, columns, batch_rows):
    scanner = dataset.scanner(columns=columns, batch_size=batch_rows, use_threads=True)
    for rb in scanner.to_reader():
        yield rb.to_pandas()  # keep it simple; pandas handles Arrow types

# open dataset lazily & infer columns
dataset = ds.dataset(IN_PATH, format="parquet")

# small sample to detect numeric cols and optional Label
sample_rb = next(dataset.scanner(batch_size=1).to_reader())
sample_df = sample_rb.to_pandas()
num_cols = sample_df.select_dtypes(include="number").columns.tolist()
label_col = "Label" if "Label" in dataset.schema.names else None
if label_col and label_col in num_cols:
    num_cols.remove(label_col)

if len(num_cols) < 2:
    raise ValueError("Need at least 2 numeric columns for clustering.")
print(f"Using {len(num_cols)} numeric columns.", flush=True)

# ----------------- PASS A: fit scaler -----------------
scaler = StandardScaler()
for batch in tqdm(iter_batches(dataset, num_cols, CHUNK_ROWS), desc="Pass A: scaler"):
    X = batch[num_cols].astype("float32")
    X = X.fillna(X.median())
    scaler.partial_fit(X)

# ----------------- PASS B: build BIRCH tree -----------------
birch = Birch(
    threshold=BIRCH_THRESHOLD,
    branching_factor=BRANCHING,
    n_clusters=None,
    compute_labels=False
)

for batch in tqdm(iter_batches(dataset, num_cols, CHUNK_ROWS), desc="Pass B: birch"):
    X = batch[num_cols].astype("float32")
    X = X.fillna(X.median())
    Xs = scaler.transform(X)
    birch.partial_fit(Xs)

# ----------------- PASS C: stream, assign labels, accumulate sums -----------------
# For centroids we need SUM(feature) and COUNT per cluster (in original scale).
sums = {}
counts = defaultdict(int)
label_counts = defaultdict(Counter) if label_col else None

for batch in tqdm(iter_batches(dataset, num_cols + ([label_col] if label_col else []), CHUNK_ROWS),
                  desc="Pass C: centroids"):
    X_raw = batch[num_cols].astype("float32")
    X_raw = X_raw.fillna(X_raw.median())

    Xs = scaler.transform(X_raw)                 # scale to predict cluster
    labels = birch.predict(Xs)

    # group indices by cluster for this batch
    unique_labels = np.unique(labels)
    for cl in unique_labels:
        idx = np.where(labels == cl)[0]
        if len(idx) == 0:
            continue
        # accumulate sums/counts in ORIGINAL feature space
        sub_sum = X_raw.iloc[idx].sum(axis=0).to_numpy(dtype=np.float64)
        if cl not in sums:
            sums[cl] = np.zeros(len(num_cols), dtype=np.float64)
        sums[cl] += sub_sum
        counts[cl] += len(idx)

        # label mode (optional)
        if label_col:
            label_vals = batch.iloc[idx][label_col].to_numpy()
            label_counts[cl].update(label_vals)

# ----------------- build centroids dataframe -----------------
clusters = sorted(counts.keys())
centroid_mat = np.vstack([sums[cl] / counts[cl] for cl in clusters]).astype("float32")
centroids_df = pd.DataFrame(centroid_mat, columns=num_cols)
centroids_df.insert(0, "birch_cluster", clusters)
centroids_df["cluster_size"] = [counts[cl] for cl in clusters]

if label_col:
    majority = [label_counts[cl].most_common(1)[0][0] if label_counts[cl] else None for cl in clusters]
    centroids_df[label_col] = majority

# save centroids and stats
centroids_df.to_parquet(OUT_CENTROIDS, index=False)
stats_df = pd.DataFrame(
    {"birch_cluster": clusters, "count": [counts[c] for c in clusters]}
).sort_values("birch_cluster")
stats_df.to_csv(OUT_STATS, index=False)

print(f"Saved centroids -> {OUT_CENTROIDS}")
print(f"Saved stats     -> {OUT_STATS}")
print(f"Clusters: {len(clusters)}  |  Total rows accounted: {sum(counts.values()):,}")
print(centroids_df.head())