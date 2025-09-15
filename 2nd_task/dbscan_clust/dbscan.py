#!/usr/bin/env python3
from __future__ import annotations
import os, json, math, time
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Limit threads to avoid RAM spikes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import pyarrow.parquet as pq, pyarrow as pa
    HAVE_PYARROW = True
except Exception:
    HAVE_PYARROW = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Tunables (env overrides)
MIN_SAMPLES    = int(os.getenv("DBSCAN_MIN_SAMPLES", "64"))
MAX_ROWS       = int(os.getenv("DBSCAN_MAX_ROWS", "100000"))   # ↓ default from 300k
EPS_Q          = float(os.getenv("DBSCAN_EPS_Q", "0.95"))
EPS_SAMPLE     = int(os.getenv("DBSCAN_EPS_SAMPLE", "20000"))  # ↓ default from 50k
FEATURE_CAP    = int(os.getenv("DBSCAN_FEATURE_CAP", "40"))    # keep top-variance K features
USE_PCA_K      = int(os.getenv("DBSCAN_PCA", "0"))             # 0 = off; else K dims

META_COLS_CANDIDATES = ["Label", "Traffic Type", "Traffic Subtype"]

def default_input_path() -> str:
    env = os.getenv("DBSCAN_IN_PATH")
    if env and os.path.exists(env): return env
    candidates = [
        os.path.join(SCRIPT_DIR, "..", "row_sampl", "stratified.parquet"),
        os.path.join(SCRIPT_DIR, "..", "..", "1st_task", "data", "reduced", "merged_reduced.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return env or candidates[-1]

def _numeric_columns_from_schema(pf: "pq.ParquetFile") -> Tuple[List[str], List[str]]:
    schema = pf.schema_arrow
    all_cols = [schema.names[i] for i in range(len(schema))]
    meta_cols = [c for c in META_COLS_CANDIDATES if c in all_cols]
    num_cols = []
    for i in range(len(schema)):
        name, typ = schema.names[i], schema.types[i]
        if name in meta_cols: continue
        try:
            if pa.types.is_integer(typ) or pa.types.is_floating(typ):
                num_cols.append(name)
        except Exception:
            pass
    return num_cols, meta_cols

def load_frame_parquet(path: str, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if HAVE_PYARROW:
        pf = pq.ParquetFile(path)
        total_rows, n_groups = pf.metadata.num_rows, pf.num_row_groups
        num_cols, meta_cols = _numeric_columns_from_schema(pf)
        read_cols = num_cols + meta_cols if num_cols else None
        if max_rows >= total_rows:
            table = pf.read(columns=read_cols)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
        else:
            sample_rate = max_rows / total_rows
            chunks, remaining = [], max_rows
            for rg in range(n_groups):
                rg_rows = pf.metadata.row_group(rg).num_rows
                take = min(rg_rows, int(math.ceil(rg_rows * sample_rate)))
                if take <= 0: continue
                df_rg = pf.read_row_group(rg, columns=read_cols).to_pandas(types_mapper=pd.ArrowDtype)
                if take < len(df_rg):
                    idx = rng.choice(len(df_rg), size=take, replace=False)
                    df_rg = df_rg.iloc[idx]
                chunks.append(df_rg)
                remaining -= len(df_rg)
                if remaining <= 0: break
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        meta_keep = [c for c in META_COLS_CANDIDATES if c in df.columns]
        X = df.select_dtypes(include=["number"]).astype("float32", copy=False)
        if meta_keep:
            for c in meta_keep: X[c] = df[c].astype("string")
        return X
    df = pd.read_parquet(path, engine="pyarrow")
    meta_keep = [c for c in META_COLS_CANDIDATES if c in df.columns]
    X = df.select_dtypes(include=["number"])
    if len(df) > max_rows:
        X = X.sample(max_rows, random_state=random_state)
        if meta_keep:
            meta = df.loc[X.index, meta_keep].astype("string")
            for c in meta_keep: X[c] = meta[c]
    else:
        if meta_keep:
            for c in meta_keep: X[c] = df[c].astype("string")
    return X.astype("float32", copy=False)

def estimate_eps(Xs: np.ndarray, min_samples: int, q: float = 0.95, sample_rows: int = 20000, random_state: int = 42) -> float:
    n = Xs.shape[0]
    if n <= min_samples + 1: return 0.5
    if n > sample_rows:
        idx = np.random.default_rng(random_state).choice(n, size=sample_rows, replace=False)
        Xk = Xs[idx]
    else:
        Xk = Xs
    k = max(2, min(min_samples, len(Xk) - 1))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1)
    nbrs.fit(Xk)
    dists, _ = nbrs.kneighbors(Xk, n_neighbors=k)
    kth = dists[:, -1]
    eps = float(np.quantile(kth, q))
    if eps <= 0: eps = float(np.median(kth[kth > 0])) if np.any(kth > 0) else 0.5
    return eps

def compute_cluster_centroids(df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str], meta_cols: List[str]) -> pd.DataFrame:
    ser = pd.Series(labels, index=df.index, name="dbscan_cluster")
    valid = ser >= 0
    if not valid.any():
        return pd.DataFrame(columns=["dbscan_cluster"] + feature_cols + ["cluster_size"] + meta_cols)
    grouped = df.loc[valid, feature_cols].groupby(ser[valid])
    centroid = grouped.mean()
    sizes = grouped.size().rename("cluster_size")
    out = centroid.join(sizes)
    for mc in meta_cols:
        maj = df.loc[valid, [mc]].join(ser).groupby("dbscan_cluster")[mc] \
              .agg(lambda s: s.value_counts(dropna=True).idxmax() if len(s) else pd.NA).rename(mc)
        out = out.join(maj)
    return out.reset_index()

def main():
    t0 = time.time()
    in_path = default_input_path()
    print(f"[DBSCAN] Input: {in_path}")
    print(f"[DBSCAN] min_samples={MIN_SAMPLES}  MAX_ROWS={MAX_ROWS}  EPS_Q={EPS_Q}  EPS_SAMPLE={EPS_SAMPLE}  FEATURE_CAP={FEATURE_CAP}  PCA={USE_PCA_K}")

    if not os.path.exists(in_path):
        raise SystemExit(f"Input file not found: {in_path}")

    df = load_frame_parquet(in_path, max_rows=MAX_ROWS, random_state=42)
    if df.empty: raise SystemExit("Loaded empty frame; check input/columns.")

    meta_cols = [c for c in META_COLS_CANDIDATES if c in df.columns]
    feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in meta_cols]

    if len(feature_cols) < 2:
        raise SystemExit(f"Need ≥2 numeric features; got: {feature_cols}")

    # Cap features by variance (helps a lot with memory & runtime)
    if FEATURE_CAP and len(feature_cols) > FEATURE_CAP:
        var = df[feature_cols].var(axis=0, skipna=True)
        top = var.sort_values(ascending=False).head(FEATURE_CAP).index.tolist()
        feature_cols = top
        print(f"[DBSCAN] Using top-{FEATURE_CAP} variance features.")

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

    # Optional PCA for further dimensionality reduction
    if USE_PCA_K and USE_PCA_K < X.shape[1]:
        from sklearn.decomposition import PCA
        X = PCA(n_components=USE_PCA_K, svd_solver="randomized", random_state=42).fit_transform(X).astype(np.float32, copy=False)
        print(f"[DBSCAN] PCA -> shape {X.shape}")

    # Standardize (keep float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X).astype(np.float32, copy=False)

    eps = estimate_eps(Xs, min_samples=MIN_SAMPLES, q=EPS_Q, sample_rows=EPS_SAMPLE, random_state=42)
    print(f"[DBSCAN] Estimated eps ≈ {eps:.4f}")

    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="euclidean", n_jobs=1)  # 1 job to avoid RAM spikes
    labels = db.fit_predict(Xs)

    unique, counts = np.unique(labels, return_counts=True)
    stats_df = pd.DataFrame({"dbscan_cluster": unique, "count": counts}).sort_values("dbscan_cluster").reset_index(drop=True)
    n_noise = int(stats_df.loc[stats_df["dbscan_cluster"] == -1, "count"].sum()) if (-1 in unique) else 0
    print(f"[DBSCAN] clusters (excl. noise): {int((unique >= 0).sum())}    noise: {n_noise}/{len(labels)}")

    centroids_df = compute_cluster_centroids(df, labels, feature_cols, meta_cols)

    sil = None
    try:
        if ((unique >= 0).sum() >= 2) and (len(df) <= 10000):
            sil = float(silhouette_score(Xs[labels >= 0], labels[labels >= 0]))
            print(f"[DBSCAN] silhouette (no-noise): {sil:.4f}")
    except Exception as e:
        print(f"[DBSCAN] silhouette failed: {e!r}")

    cent_path = os.path.join(OUT_DIR, "merged_reduced_dbscan_centroids.parquet")
    stats_path = os.path.join(OUT_DIR, "merged_reduced_dbscan_stats.csv")
    meta_path  = os.path.join(OUT_DIR, "merged_reduced_dbscan_meta.json")

    centroids_df.to_parquet(cent_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    meta = {
        "in_path": in_path, "rows_loaded": int(len(df)),
        "feature_cols": feature_cols, "meta_cols": meta_cols,
        "standardization": True,
        "dbscan_params": {"eps": float(eps), "min_samples": int(MIN_SAMPLES), "metric": "euclidean"},
        "n_clusters_excl_noise": int((unique >= 0).sum()),
        "noise_count": int(n_noise),
        "silhouette_no_noise": sil,
        "created_at": pd.Timestamp.utcnow().isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)

    print(f"[DBSCAN] Wrote:\n  - {cent_path}\n  - {stats_path}\n  - {meta_path}\nDone in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
