from pathlib import Path
import polars as pl

ROOT = Path(r"C:\Users\chrka\OneDrive\Documents\GitHub\Data-Mining-Project\init_compr\tii_ssrc23_parquet")
print("[check] root exists:", ROOT.exists(), "->", ROOT)

# Try Polars' hive partitioning first; fall back to PyArrow dataset if needed
try:
    lf = pl.scan_parquet(str(ROOT), hive_partitioning=True)  # Polars ≥ 0.20
except TypeError:
    import pyarrow.dataset as ds
    ds_hive = ds.dataset(str(ROOT), format="parquet", partitioning="hive")
    lf = pl.scan_pyarrow_dataset(ds_hive)


schema_names = lf.collect_schema().names()
rename_map = {}
if "Traffic Type" in schema_names:    rename_map["Traffic Type"] = "Type"
if "Traffic Subtype" in schema_names: rename_map["Traffic Subtype"] = "Subtype"
if rename_map:
    lf = lf.rename(rename_map)

print("[schema]", lf.collect_schema().names())

# 1) Totals & distinct counts
summary = (
    lf.select([
        pl.len().alias("rows"),
        pl.col("Type").n_unique().alias("n_types"),
        pl.col("Subtype").n_unique().alias("n_subtypes"),
        pl.col("Label").n_unique().alias("n_labels"),
    ])
).collect()
print(summary)

# 2) Label distribution
dist = (
    lf.group_by("Label")
      .len()
      .sort("len", descending=True)
).collect()
print(dist)

# example filtered aggregation (uses real column name)
dos = (
    lf.filter( (pl.col("Type") == "DoS") & (pl.col("Label") == "Malicious") )
      .group_by("Subtype")
      .agg(pl.col("Fwd Packet Length Mean").mean().alias("avg_fwd_len"))
      .sort("avg_fwd_len", descending=True)
      .limit(10)
).collect()
print(dos)
