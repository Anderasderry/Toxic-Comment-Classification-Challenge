"""
Grid search hyperparameters for the TF-IDF + logistic regression baseline.

Uses the same data paths, label columns, and validation split convention as
baseline_tfidf_lr.py (default val_size=0.1, random_state=42).

Example:
  python grid_search_tfidf_lr.py
  python grid_search_tfidf_lr.py --data-dir ./datasets --out ./grid_search_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split

from baseline_tfidf_lr import (
    LABEL_COLS,
    _default_data_dir,
    _resolve_table,
    build_pipeline,
    stacked_positive_proba,
)
from run_paths import new_run_timestamp, path_with_timestamp


def search_best_thresholds(y_true: pd.DataFrame, y_proba: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 19)
    best = np.full(len(LABEL_COLS), 0.5, dtype=np.float64)
    y_mat = y_true.to_numpy()
    for i in range(len(LABEL_COLS)):
        yt = y_mat[:, i]
        yp = y_proba[:, i]
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            pred = (yp >= t).astype(np.int32)
            s = f1_score(yt, pred, zero_division=0)
            if s > best_f1:
                best_f1 = s
                best_t = float(t)
        best[i] = best_t
    return best


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate(y_val: pd.DataFrame, val_proba: np.ndarray) -> dict[str, float]:
    y_true = y_val.to_numpy()
    thresholds = search_best_thresholds(y_val, val_proba)
    y_pred = (val_proba >= thresholds[None, :]).astype(np.int32)

    out: dict[str, float] = {}
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    f1s: list[float] = []

    for i, name in enumerate(LABEL_COLS):
        ra = _safe_roc_auc(y_true[:, i], val_proba[:, i])
        pa = float(average_precision_score(y_true[:, i], val_proba[:, i]))
        f1 = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        if ra is not None:
            roc_aucs.append(ra)
            out[f"roc_auc_{name}"] = ra
        out[f"pr_auc_{name}"] = pa
        out[f"f1_{name}"] = f1
        out[f"thr_{name}"] = float(thresholds[i])
        pr_aucs.append(pa)
        f1s.append(f1)

    out["roc_auc_mean"] = float(np.mean(roc_aucs)) if roc_aucs else float("nan")
    out["pr_auc_mean"] = float(np.mean(pr_aucs))
    out["f1_mean"] = float(np.mean(f1s))
    out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Folder containing train.csv or train.csv.zip.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size (same convention as baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "grid_search_results.csv",
        help="CSV path for full grid results (timestamp appended by default).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not append a run timestamp to --out filename.",
    )
    args = parser.parse_args()

    run_ts = new_run_timestamp()
    out_path = args.out.expanduser().resolve()
    if not args.no_timestamp:
        out_path = path_with_timestamp(out_path, run_ts)

    data_dir = args.data_dir.expanduser().resolve()
    train_path = _resolve_table(data_dir, "train")
    train_df = pd.read_csv(train_path)

    for col in LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(f"train.csv missing label column: {col}")
    if "comment_text" not in train_df.columns:
        raise ValueError("Expected comment_text column in train.csv.")

    X = train_df["comment_text"].astype(str)
    y = train_df[LABEL_COLS].astype(np.int32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=args.seed,
    )

    param_grid = {
        "max_features": [50_000, 100_000, 150_000],
        "ngram_max": [1, 2],
        "min_df": [3, 5],
        "max_df": [0.9, 0.95],
        "C": [2.0, 4.0],
    }

    rows: list[dict[str, float | int | str]] = []
    combos = list(ParameterGrid(param_grid))
    for idx, params in enumerate(combos, start=1):
        print(f"[{idx:02d}/{len(combos)}] params={params}")
        pipe = build_pipeline(
            max_features=params["max_features"],
            ngram_max=params["ngram_max"],
            min_df=params["min_df"],
            max_df=params["max_df"],
            C=params["C"],
        )
        pipe.fit(X_tr, y_tr)
        val_proba = stacked_positive_proba(pipe, X_val)
        metrics = evaluate(y_val, val_proba)

        row: dict[str, float | int | str] = {
            "run_id": run_ts,
            "max_features": int(params["max_features"]),
            "ngram_max": int(params["ngram_max"]),
            "min_df": int(params["min_df"]),
            "max_df": float(params["max_df"]),
            "C": float(params["C"]),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"    roc_auc_mean={metrics['roc_auc_mean']:.4f} "
            f"pr_auc_mean={metrics['pr_auc_mean']:.4f} "
            f"micro_f1={metrics['micro_f1']:.4f} macro_f1={metrics['macro_f1']:.4f}",
        )

    result = pd.DataFrame(rows).sort_values(
        ["roc_auc_mean", "pr_auc_mean", "macro_f1"],
        ascending=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    best = result.iloc[0]
    print("\nBest config by roc_auc_mean:")
    print(
        f"  max_features={int(best['max_features'])}, ngram_max={int(best['ngram_max'])}, "
        f"min_df={int(best['min_df'])}, max_df={float(best['max_df'])}, C={float(best['C'])}",
    )
    print(
        f"  roc_auc_mean={float(best['roc_auc_mean']):.4f}, "
        f"pr_auc_mean={float(best['pr_auc_mean']):.4f}, "
        f"micro_f1={float(best['micro_f1']):.4f}, "
        f"macro_f1={float(best['macro_f1']):.4f}",
    )
    print(f"\nSaved results: {out_path.resolve()}")
    print(
        "\nTo train baseline with these settings, update build_pipeline() defaults in "
        "baseline_tfidf_lr.py or pass the same arguments if you add CLI flags there.",
    )


if __name__ == "__main__":
    main()
