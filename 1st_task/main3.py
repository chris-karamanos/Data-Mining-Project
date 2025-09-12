import glob, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

KEEP = [
    "Flow Duration","Total Fwd Packet","Total Bwd packets",
    "Total Length of Fwd Packet","Total Length of Bwd Packet",
    "Flow Bytes/s","Flow Packets/s",
    "Fwd Packet Length Max","Fwd Packet Length Min",
    "Bwd Packet Length Max","Bwd Packet Length Min",
    "Packet Length Mean","Packet Length Std",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Mean","Bwd IAT Mean",
    "Protocol","SYN Flag Count","ACK Flag Count","RST Flag Count",
    "PSH Flag Count","FIN Flag Count","Fwd PSH Flags","Bwd PSH Flags",
    "Label","Traffic Type","Traffic Subtype",
]
Path("data/reduced").mkdir(parents=True, exist_ok=True)

for p in sorted(glob.glob("data/chunks/part_*.parquet")):
    df = pd.read_parquet(p, engine="pyarrow")
    df[KEEP].to_parquet(
        f"data/reduced/{Path(p).stem}_reduced.parquet",
        engine="pyarrow", compression="zstd", index=False
    )