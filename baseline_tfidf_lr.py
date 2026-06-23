"""
Baseline: TF-IDF + six binary logistic regressions (multi-label toxic comments).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from run_paths import fig_run_dir, new_run_timestamp, path_with_timestamp
from viz import plot_baseline_final_eval, plot_tfidf_top_terms

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def _resolve_table(data_dir: Path, stem: str) -> Path:
    """Prefer plain CSV; fall back to stem.csv.zip (Kaggle bundle layout)."""
    plain = data_dir / f"{stem}.csv"
    if plain.is_file():
        return plain
    zipped = data_dir / f"{stem}.csv.zip"
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(
        f"Missing {stem}.csv or {stem}.csv.zip under {data_dir}",
    )


def build_pipeline(
    *,
    max_features: int = 100_000,
    ngram_max: int = 2,
    min_df: int = 3,
    max_df: float = 0.9,
    C: float = 4.0,
    max_iter: int = 2000,
) -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    base_lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="saga",
    )
    clf = MultiOutputClassifier(base_lr, n_jobs=-1)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def stacked_positive_proba(pipe: Pipeline, X) -> np.ndarray:
    parts = pipe.predict_proba(X)
    return np.column_stack([p[:, 1] for p in parts])


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_validation_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_cols: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Validation metrics aligned with train_transformer.py (threshold=0.5 for F1)."""
    metrics: dict[str, float] = {}
    aucs: list[float] = []
    prs: list[float] = []

    for i, name in enumerate(label_cols):
        y = y_true[:, i]
        p = y_proba[:, i]
        ra = _safe_roc_auc(y, p)
        if ra is not None:
            metrics[f"roc_auc_{name}"] = ra
            aucs.append(ra)
        metrics[f"pr_auc_{name}"] = float(average_precision_score(y, p))
        prs.append(metrics[f"pr_auc_{name}"])

    if aucs:
        metrics["roc_auc_mean"] = float(np.mean(aucs))
    metrics["pr_auc_mean"] = float(np.mean(prs))

    preds = (y_proba >= threshold).astype(np.int32)
    metrics["f1_micro"] = float(f1_score(y_true, preds, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, preds, average="macro", zero_division=0))
    return metrics


def print_validation_metrics(metrics: dict[str, float], label_cols: list[str]) -> None:
    print("Validation ROC-AUC (per label):")
    for name in label_cols:
        key = f"roc_auc_{name}"
        if key in metrics:
            print(f"  {name:16s} {metrics[key]:.4f}")
    if "roc_auc_mean" in metrics:
        print(f"  {'mean':16s} {metrics['roc_auc_mean']:.4f}")
    if "pr_auc_mean" in metrics:
        print(f"  {'pr_auc_mean':16s} {metrics['pr_auc_mean']:.4f}")
    if "f1_micro" in metrics:
        print(f"  {'f1_micro':16s} {metrics['f1_micro']:.4f}")
    if "f1_macro" in metrics:
        print(f"  {'f1_macro':16s} {metrics['f1_macro']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Folder containing train/test/sample CSV or .csv.zip files.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Holdout fraction for local ROC-AUC (0 skips validation).",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(__file__).resolve().parent / "submission_baseline.csv",
        help="Output submission CSV path (a run timestamp is appended to the filename).",
    )
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figs",
        help="Root directory for run figures (figs/baseline_<timestamp>/).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip saving evaluation plots.",
    )
    args = parser.parse_args()

    run_ts = new_run_timestamp()
    args.submission = path_with_timestamp(args.submission.expanduser().resolve(), run_ts)
    save_figures = not args.no_figures
    fig_dir = fig_run_dir(args.figs_dir, "baseline", run_ts) if save_figures else None

    data_dir = args.data_dir.expanduser().resolve()

    train_path = _resolve_table(data_dir, "train")
    test_path = _resolve_table(data_dir, "test")
    sample_path = _resolve_table(data_dir, "sample_submission")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    for col in LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(f"train.csv missing label column: {col}")
    if "comment_text" not in train_df.columns or "comment_text" not in test_df.columns:
        raise ValueError("Expected comment_text column in train and test.")

    X = train_df["comment_text"].astype(str)
    y = train_df[LABEL_COLS].astype(np.int32)

    pipe = build_pipeline()

    if save_figures and fig_dir is not None:
        print(f"Figures directory: {fig_dir.resolve()}")

    if args.val_size > 0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=args.val_size,
            random_state=42,
        )
        print("Fitting on train split for validation...")
        pipe.fit(X_tr, y_tr)
        val_proba = stacked_positive_proba(pipe, X_val)
        metrics = compute_validation_metrics(
            y_val.values.astype(np.float64),
            val_proba,
            LABEL_COLS,
        )
        print_validation_metrics(metrics, LABEL_COLS)

        if save_figures and fig_dir is not None:
            plot_baseline_final_eval(
                y_val.values.astype(np.float64),
                val_proba,
                LABEL_COLS,
                fig_dir,
            )
            plot_tfidf_top_terms(pipe, LABEL_COLS, fig_dir / "tfidf_top_terms.png")

        print("Refitting on full training data...")
        pipe.fit(X, y)
    else:
        print("Fitting on full training data...")
        pipe.fit(X, y)
        if save_figures:
            print(
                "Warning: --val-size 0: baseline figures need a validation split; "
                "skipped confusion matrix / ROC.",
                flush=True,
            )

    print("Predicting test set...")
    test_proba = stacked_positive_proba(pipe, test_df["comment_text"].astype(str))

    submission = pd.DataFrame({"id": test_df["id"].values})
    submission[LABEL_COLS] = test_proba
    label_order = [c for c in sample.columns if c != "id"]
    if label_order:
        submission = submission[["id"] + label_order]

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission, index=False)
    print(f"Wrote submission: {args.submission.resolve()}")


if __name__ == "__main__":
    main()
