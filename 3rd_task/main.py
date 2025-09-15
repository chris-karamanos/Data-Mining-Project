#!/usr/bin/env python3
"""
Ερώτημα 3 — Ταξινόμηση με SVM & MLP (Neural Networks)

Datasets:
  1) ../2nd_task/row_sampl/stratified.parquet
  2) ../2nd_task/birch_clust/birch_representatives.parquet
"""

from __future__ import annotations
import os, json, time, inspect
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

# Quiet super-noisy convergence warnings from LinearSVC (optional)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---- pyarrow (for safe parquet→pandas with dictionary columns) ----
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PA = True
except Exception:
    HAVE_PA = False

# ---------------- Configuration ----------------
MAX_ROWS      = int(os.getenv("MAX_ROWS", "200000"))
SVM_MAX_TRAIN = int(os.getenv("SVM_MAX_TRAIN", "60000"))
RANDOM_STATE  = 42

DATASETS = {
    "stratified": os.getenv("DATASET1", "../2nd_task/row_sampl/stratified.parquet"),
    "birch_reps": os.getenv("DATASET2", "../2nd_task/birch_clust/birch_representatives.parquet"),
}
TARGETS = ["Label", "Traffic Type"]

OUT_DIR = Path("3rd_task/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- small util ----------
def accepts_fit_param(estimator, param: str) -> bool:
    """True if estimator.fit accepts a given keyword argument."""
    try:
        return param in inspect.signature(estimator.fit).parameters
    except Exception:
        return False

# -------------- Helpers ----------------
def load_parquet_sample(path: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Read parquet robustly: cast dictionary columns -> string before pandas."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if HAVE_PA:
        table = pq.read_table(p)

        # Cast any dictionary<values=string,...> columns to plain string
        if any(pa.types.is_dictionary(f.type) for f in table.schema):
            new_fields = []
            for f in table.schema:
                if pa.types.is_dictionary(f.type):
                    new_fields.append(pa.field(f.name, pa.string()))
                else:
                    new_fields.append(f)
            table = table.cast(pa.schema(new_fields))

        # Uniform sample BEFORE converting to pandas (saves RAM)
        if max_rows and table.num_rows > max_rows:
            idx = np.random.default_rng(RANDOM_STATE).choice(table.num_rows, size=max_rows, replace=False)
            table = table.take(pa.array(idx))

        # Convert to pandas (avoid Arrow extension dtypes)
        df = table.to_pandas(types_mapper=None)
    else:
        # Fallback readers
        try:
            df = pd.read_parquet(p, engine="fastparquet")
        except Exception:
            df = pd.read_parquet(p)  # default engine

        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=RANDOM_STATE)

    return df

def prepare_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    to_drop = [c for c in ["Label", "Traffic Type", "Traffic Subtype",
                           "birch_cluster", "cluster_size"] if c in df.columns and c != target]
    X = df.drop(columns=to_drop, errors="ignore").select_dtypes(include=["number"]).copy()
    y = df[target].astype("string")

    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    if X.shape[1] < 2:
        raise SystemExit(f"Need ≥2 numeric features for target '{target}'. Got: {list(X.columns)}")
    return X, y

def sample_weights_inverse_freq(y: pd.Series) -> np.ndarray:
    vc = y.value_counts()
    w = y.map(lambda c: 1.0 / vc[c])
    return (w / w.mean()).to_numpy(float)

def save_confusion_matrix(cm: np.ndarray, labels: List[str], path_csv: Path) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path_csv, index=True)

def safe_auc(y_true_enc: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float | None:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true_enc, y_proba[:, 1]))
        return float(roc_auc_score(y_true_enc, y_proba, multi_class="ovo", average="macro"))
    except Exception:
        return None

def collect_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    auc = None
    if y_proba is not None:
        le = LabelEncoder().fit(y_true)
        y_true_enc = le.transform(y_true)
        if y_proba.ndim == 1:
            y_proba = np.vstack([1 - y_proba, y_proba]).T
        auc = safe_auc(y_true_enc, y_proba, n_classes=len(le.classes_))

    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec,
            "f1_macro": f1, "roc_auc": auc, "n_classes": int(pd.Series(y_true).nunique())}

# -------------- Models ----------------
def train_eval_svm(X_train, y_train, X_test, y_test, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_linear = (len(X_train) > 100_000) or (y_train.nunique() > 20)

    if use_linear:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))
        ])
        # bump max_iter to avoid convergence spam
        param_grid = {"clf__C": [0.1, 1, 5], "clf__max_iter": [2000, 5000]}
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE))
        ])
        param_grid = {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", "auto"]}
        if len(X_train) > SVM_MAX_TRAIN:
            idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=SVM_MAX_TRAIN, replace=False)
            X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True, verbose=0)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    y_proba = None
    if hasattr(best.named_steps["clf"], "predict_proba"):
        try:
            y_proba = best.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = collect_metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = gs.best_params_

    (out_dir / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(classification_report(y_test, y_pred), encoding="utf-8")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    save_confusion_matrix(cm, labels=sorted(y_test.unique()), path_csv=out_dir / "confusion_matrix.csv")
    return metrics

def train_eval_mlp(X_train, y_train, X_test, y_test, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_STATE
        ))
    ])

    param_grid = {
        "clf__hidden_layer_sizes": [(128,), (256,), (128, 64)],
        "clf__alpha": [1e-4, 1e-3]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid=param_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, refit=True, verbose=0
    )

    # Only pass sample_weight if this scikit-learn supports it for MLPClassifier
    fit_kwargs = {}
    if accepts_fit_param(pipe.named_steps["clf"], "sample_weight"):
        sw = sample_weights_inverse_freq(y_train)
        fit_kwargs["clf__sample_weight"] = sw

    gs.fit(X_train, y_train, **fit_kwargs)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    # Probabilities (if available)
    y_proba = None
    try:
        y_proba = best.predict_proba(X_test)
    except Exception:
        pass

    metrics = collect_metrics(y_test, y_pred, y_proba)
    metrics["best_params"] = gs.best_params_

    (out_dir / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(classification_report(y_test, y_pred), encoding="utf-8")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    save_confusion_matrix(cm, labels=sorted(y_test.unique()), path_csv=out_dir / "confusion_matrix.csv")
    return metrics

# -------------- Orchestration ----------------
def run_for_dataset(ds_key: str, path: str) -> None:
    print(f"\n=== Dataset: {ds_key} -> {path} ===")
    df = load_parquet_sample(path, MAX_ROWS)

    results: Dict[str, Any] = {}
    for target in TARGETS:
        if target not in df.columns:
            print(f"[skip] '{target}' column not found in {ds_key}")
            continue

        X, y = prepare_xy(df, target)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        task_dir = OUT_DIR / ds_key / target.replace(" ", "_")
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training SVM for target={target} (n={len(X_tr)})")
        m_svm = train_eval_svm(X_tr, y_tr, X_te, y_te, task_dir / "svm")

        print(f"Training MLP for target={target} (n={len(X_tr)})")
        m_mlp = train_eval_mlp(X_tr, y_tr, X_te, y_te, task_dir / "mlp")

        best = "svm" if m_svm["f1_macro"] >= m_mlp["f1_macro"] else "mlp"
        summary = {
            "svm": m_svm,
            "mlp": m_mlp,
            "best_model": best,
            "best_f1_macro": max(m_svm["f1_macro"], m_mlp["f1_macro"]),
        }
        (task_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results[target] = summary
        print(f"[{ds_key} | {target}] best={best} F1_macro={summary['best_f1_macro']:.4f}")

    (OUT_DIR / ds_key / "dataset_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

def main() -> None:
    t0 = time.time()
    for key, path in DATASETS.items():
        try:
            run_for_dataset(key, path)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
    print(f"\nAll done in {time.time()-t0:.1f}s. See: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
