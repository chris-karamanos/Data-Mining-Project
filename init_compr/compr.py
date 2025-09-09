import os, time, math
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path

CSV_PATH = "../../../data.csv"        
OUT_DIR = Path("tii_ssrc23_parquet")
OUT_DIR.mkdir(exist_ok=True)

size_gb = os.path.getsize(CSV_PATH) / (1024**3)
print(f"[info] Reading {CSV_PATH} ({size_gb:.2f} GB)…")


# 1) Stream-read CSV with Arrow (no pandas overhead)
table_reader = pv.open_csv(
    CSV_PATH,
    read_options=pv.ReadOptions(block_size=64*1024*1024),
    parse_options=pv.ParseOptions(delimiter=","),
    convert_options=pv.ConvertOptions(strings_can_be_null=True)
)

# 2) Helpers to tighten types on the fly (Downcast numerics)
def tighten_numeric(col):
    # Try int64 -> int32 -> int16 if safe; float64 -> float32 if safe.
    t = col.type
    if pa.types.is_integer(t):
        for target in [pa.int32(), pa.int16()]:
            try:
                c2 = pc.cast(col, target, safe=True)
                return c2
            except pa.ArrowInvalid:
                pass
    elif pa.types.is_floating(t):
        try:
            return pc.cast(col, pa.float32(), safe=True)
        except pa.ArrowInvalid:
            return col
    return col

def dict_encode_if_string(col):
    if pa.types.is_string(col.type):
        # Dictionary encode (low-cardinality strings benefit a lot)
        return pc.dictionary_encode(col)
    return col

# 3) Decide partition columns that (likely) exist in this dataset:
PARTITION_COLS = ["Type", "Subtype", "Label"]  
t0 = time.time()
rows_total = 0


# 4) Process & write chunks
for i, chunk in enumerate(table_reader, 1):
    names = chunk.column_names
    cols  = []
    for name, col in zip(names, chunk.itercolumns()):
        c = tighten_numeric(col)
        c = dict_encode_if_string(c)
        cols.append(c)
    chunk = pa.table(cols, names=names)

    existing_parts = [c for c in PARTITION_COLS if c in names]
    if not existing_parts:
        existing_parts = []

    pq.write_to_dataset(
        chunk,
        root_path=str(OUT_DIR),
        partition_cols=existing_parts,
        compression="zstd",
        use_dictionary=True,
        data_page_size=1_048_576,
        write_statistics=True,
        existing_data_behavior="overwrite_or_ignore"
    )

    rows_total += chunk.num_rows
    if i == 1:
        print(f"[progress] First chunk: {chunk.num_rows} rows, {chunk.num_columns} cols")
    if i % 5 == 0:
        elapsed = time.time() - t0
        rate = rows_total / max(elapsed,1)
        print(f"[progress] chunks={i}, rows={rows_total:,}, rate≈{int(rate):,} rows/s, elapsed={elapsed:.1f}s", flush=True)

print(f"[done] Wrote dataset to {OUT_DIR}. Total rows={rows_total:,}. Elapsed={time.time()-t0:.1f}s")
