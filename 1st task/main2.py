import pandas as pd, glob, os, math

paths = sorted(glob.glob("outputs/eda_chunks/*_describe.csv"))
os.makedirs("outputs/eda_chunks", exist_ok=True)

gmin, gmax = {}, {}
sum_w, cnt_w = {}, {}          # for weighted mean: sum(n_i * mean_i), sum(n_i)
sum_m2_within, sum_nm2 = {}, {}  # for pooled variance

for f in paths:
    d = pd.read_csv(f, index_col=0)
    # normalize row labels just in case
    if isinstance(d.index, pd.Index) and d.index.dtype == "object":
        d.index = d.index.str.strip()

    # --- mins ---
    if "min" in d.index:
        mins = pd.to_numeric(d.loc["min"], errors="coerce")
        for c, v in mins.items():
            if pd.notna(v):
                gmin[c] = v if c not in gmin else min(gmin[c], v)

    # --- maxs ---
    if "max" in d.index:
        maxs = pd.to_numeric(d.loc["max"], errors="coerce")
        for c, v in maxs.items():
            if pd.notna(v):
                gmax[c] = v if c not in gmax else max(gmax[c], v)

    # --- means + counts (+ stds for pooled variance) ---
    have = {"mean","count"} <= set(d.index)
    have_std = have and ("std" in d.index)
    if have:
        means  = pd.to_numeric(d.loc["mean"],  errors="coerce").rename("m")
        counts = pd.to_numeric(d.loc["count"], errors="coerce").rename("n")
        parts = [means, counts]

        if have_std:
            stds = pd.to_numeric(d.loc["std"], errors="coerce").rename("s")
            parts.append(stds)

        aligned = pd.concat(parts, axis=1).dropna()

        for c, row in aligned.iterrows():
            m, n = float(row["m"]), float(row["n"])
            sum_w[c] = sum_w.get(c, 0.0) + m * n
            cnt_w[c] = cnt_w.get(c, 0.0) + n

            if have_std and not pd.isna(row["s"]):
                s = float(row["s"])
                # within-chunk sum of squares: (n_i - 1) * s_i^2
                sum_m2_within[c] = sum_m2_within.get(c, 0.0) + (n - 1.0) * (s ** 2)
                # for between-chunk term: n_i * mean_i^2
                sum_nm2[c] = sum_nm2.get(c, 0.0) + n * (m ** 2)

# finalize weighted mean
global_mean = {c: (sum_w[c] / cnt_w[c]) for c in sum_w if cnt_w[c] > 0}

# pooled std
global_std = {}
for c in set(sum_m2_within) & set(sum_nm2) & set(cnt_w) & set(global_mean):
    N = cnt_w[c]
    if N and N > 1:
        mu = global_mean[c]
        # between-chunk term = Σ n_i m_i^2 − N * mu^2
        between = sum_nm2[c] - N * (mu ** 2)
        var = (sum_m2_within[c] + between) / (N - 1.0)
        global_std[c] = math.sqrt(var) if var >= 0 else float("nan")

all_cols = sorted(set(gmin) | set(gmax) | set(global_mean) | set(global_std))
out = pd.DataFrame({
    "global_min":  [gmin.get(c)        for c in all_cols],
    "global_max":  [gmax.get(c)        for c in all_cols],
    "global_mean": [global_mean.get(c) for c in all_cols],
    "global_std":  [global_std.get(c)  for c in all_cols],
    "total_count": [cnt_w.get(c)       for c in all_cols],
}, index=all_cols)
out.index.name = "column"
out.to_csv("outputs/eda_chunks/global_minmax_mean_from_describe.csv")
print("wrote outputs/eda_chunks/global_minmax_mean_from_describe.csv")
