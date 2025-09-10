import glob
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

IN_PATTERN = "data/reduced/*reduced.parquet"   # <-- adjust if your files live elsewhere
OUT_FILE   = Path("data/reduced/merged_reduced.parquet")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

writer = None
total = 0
cols_order = None

for i, path in enumerate(sorted(glob.glob(IN_PATTERN)), 1):
    # Load this part
    df = pd.read_parquet(path, engine="pyarrow")
    # On first file, lock column order/schema
    if writer is None:
        cols_order = df.columns.tolist()
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        writer = pq.ParquetWriter(OUT_FILE.as_posix(), tbl.schema, compression="zstd")
    else:
        # Ensure same column order (in case files differ)
        df = df[cols_order]
        tbl = pa.Table.from_pandas(df, preserve_index=False)

    writer.write_table(tbl)
    total += len(df)
    if i % 3 == 0 or i == 1:
        print(f"[{i}] appended {len(df):,} rows → total {total:,}")

if writer:
    writer.close()
print(f"✅ Done → {OUT_FILE}  (rows: {total:,}, cols: {len(cols_order or [])})")
