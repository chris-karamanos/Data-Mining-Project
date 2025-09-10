import pandas as pd, glob, os, os.path

os.makedirs("outputs/eda_chunks", exist_ok=True)

for p in sorted(glob.glob("data/chunks/*.parquet")):
    df = pd.read_parquet(p)

    desc = df.describe(include="all")

    for row in ["unique", "top", "freq"]:
        if row not in desc.index:
            desc.loc[row] = pd.Series(dtype=object)

    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        vc = s.value_counts()
        desc.loc["unique", c] = int(s.nunique())
        desc.loc["top", c]    = vc.index[0]
        desc.loc["freq", c]   = int(vc.iloc[0])

    name = os.path.splitext(os.path.basename(p))[0]
    desc.to_csv(f"outputs/eda_chunks/{name}_describe.csv")

    