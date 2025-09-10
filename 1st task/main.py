
    # split_csv_v2.py
import os, json
from collections import Counter
import pandas as pd
import numpy as np
import sys

IN = "../csv/data.csv"
OUT_DIR = "data/chunks"
CHUNK = 1000000

SUMMARY_PATH = os.path.join(OUT_DIR, "_summary.json")

# If we've already split before, just print the summary and exit
if os.path.isfile(SUMMARY_PATH):
    print('Parquet files existed')
    with open(SUMMARY_PATH, "r") as f:
        print(f.read())
    sys.exit(0)

os.makedirs(OUT_DIR, exist_ok=True)


# Counters for quick sanity
counts_label = Counter()
counts_type = Counter()
counts_subtype = Counter()

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    # mutate in-place to save RAM
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    # handle pandas nullable ints (capital I)
    for c in df.select_dtypes(include=["Int64"]).columns:
        try:
            df[c] = df[c].astype("Int32")  # keeps NA support, smaller footprint
        except Exception:
            pass
    for c in df.select_dtypes(include=["float64","Float64"]).columns:
        df[c] = df[c].astype("float32")
        # Convert object columns to pandas Category (dictionary-encoded in Parquet)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category")
    return df

wrote_rows = 0
num_parts = 0

for i, chunk in enumerate(pd.read_csv(
        IN, chunksize=CHUNK, low_memory=False)):

    # Update global counts once per chunk
    if "Label" in chunk: counts_label.update(chunk["Label"].dropna())
    if "Traffic Type" in chunk: counts_type.update(chunk["Traffic Type"].dropna())
    if "Traffic Subtype" in chunk: counts_subtype.update(chunk["Traffic Subtype"].dropna())

    chunk = downcast(chunk)
    path = f"{OUT_DIR}/part_{i:04d}.parquet"
    chunk.to_parquet(path, index=False, compression="snappy")
    wrote_rows += len(chunk)
    num_parts += 1
    print("wrote", path, len(chunk), "rows")
 

# Save tiny summary
summary = {
    "rows_written": wrote_rows,
    "format": "parquet",
    "num_parts": num_parts,
    "label_counts": dict(counts_label),
    "traffic_type_counts": dict(counts_type),
    "traffic_subtype_counts": dict(counts_subtype),
}
with open(f"{OUT_DIR}/_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("Summary saved to", f"{OUT_DIR}/_summary.json")

