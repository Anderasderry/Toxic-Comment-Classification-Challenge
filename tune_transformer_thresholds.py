"""
Per-label threshold tuning for fine-tuned transformer checkpoints on the
validation split (same convention as baseline_tfidf_lr.py: val_size=0.1, seed=42).

Loads the validation-stage checkpoint, predicts probabilities on the holdout set,
reports metrics at tau=0.5 and after per-label threshold search.

Examples:
  python tune_transformer_thresholds.py --model distilbert
  python tune_transformer_thresholds.py --model hatebert
  python tune_transformer_thresholds.py --model distilbert --checkpoint ./checkpoints/distilbert/val/checkpoint-17952
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from baseline_tfidf_lr import (
    LABEL_COLS,
    _default_data_dir,
    _resolve_table,
    compute_validation_metrics,
    print_validation_metrics,
)
from grid_search_tfidf_lr import evaluate, search_best_thresholds
from run_paths import new_run_timestamp, path_with_timestamp
from train_transformer import (
    DISTILBERT_DEFAULTS,
    HATEBERT_DEFAULTS,
    ToxicCommentDataset,
    _default_hatebert_model,
    get_trainer_tokenizer,
    predict_proba,
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PRESETS = {
    "distilbert": DISTILBERT_DEFAULTS,
    "hatebert": HATEBERT_DEFAULTS,
}


def find_latest_checkpoint(val_dir: Path) -> Path:
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing validation checkpoint directory: {val_dir}")
    candidates = sorted(val_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* folders under {val_dir}")
    return candidates[-1]


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def build_predict_trainer(checkpoint_dir: Path, *, batch_size: int) -> Trainer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    args = TrainingArguments(
        output_dir=str(checkpoint_dir / "_predict"),
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )
    try:
        return Trainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
    except TypeError:
        return Trainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )


def validation_split(
    train_df: pd.DataFrame,
    *,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx_tr, idx_val = train_test_split(
        train_df.index,
        test_size=val_size,
        random_state=seed,
    )
    return train_df.loc[idx_tr], train_df.loc[idx_val]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS),
        required=True,
        help="Transformer preset: distilbert or hatebert.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Folder containing train.csv or train.csv.zip.",
    )
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Validation checkpoint directory (default: latest under checkpoints/<model>/val/).",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Eval batch size (default: preset training batch size).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional CSV to cache validation probabilities.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "transformer_threshold_results.csv",
        help="Summary metrics CSV (timestamp appended).",
    )
    parser.add_argument(
        "--proba-only",
        action="store_true",
        help="Load probabilities from --cache instead of running the checkpoint.",
    )
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    batch_size = args.batch_size or preset.batch_size
    run_ts = new_run_timestamp()
    out_path = path_with_timestamp(args.out.expanduser().resolve(), run_ts)

    default_cache = (
        PROJECT_ROOT
        / "transformer_cache"
        / args.model
        / "validation_proba.csv"
    )
    cache_path = (args.cache or default_cache).expanduser().resolve()

    data_dir = args.data_dir.expanduser().resolve()
    train_df = pd.read_csv(_resolve_table(data_dir, "train"))
    for col in LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(f"train.csv missing label column: {col}")

    _, val_df = validation_split(train_df, val_size=args.val_size, seed=args.seed)
    y_val = val_df[LABEL_COLS].astype(np.float64)

    if args.proba_only:
        if not cache_path.is_file():
            raise SystemExit(f"Missing probability cache: {cache_path}")
        val_proba = pd.read_csv(cache_path)[LABEL_COLS].to_numpy(dtype=np.float64)
        checkpoint_dir = args.checkpoint
    else:
        checkpoint_dir = args.checkpoint
        if checkpoint_dir is None:
            checkpoint_dir = find_latest_checkpoint(preset.output_dir / "val")
        else:
            checkpoint_dir = checkpoint_dir.expanduser().resolve()

        print(f"Loading checkpoint: {checkpoint_dir.resolve()}")
        trainer = build_predict_trainer(checkpoint_dir, batch_size=batch_size)
        tokenizer = get_trainer_tokenizer(trainer)
        val_ds = ToxicCommentDataset(
            val_df["comment_text"],
            val_df[LABEL_COLS],
            tokenizer,
            args.max_length,
        )
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"Predicting validation split ({len(val_df)} comments) on {device}...")
        val_proba = predict_proba(trainer, val_ds)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame(val_proba, columns=LABEL_COLS)
        cache_df.insert(0, "val_index", val_df.index.astype(int).tolist())
        cache_df.to_csv(cache_path, index=False)
        print(f"Cached validation probabilities: {cache_path.resolve()}")

    metrics_default = compute_validation_metrics(
        y_val.to_numpy(),
        val_proba,
        LABEL_COLS,
        threshold=0.5,
    )
    metrics_tuned = evaluate(y_val, val_proba)
    thresholds = search_best_thresholds(y_val, val_proba)

    print(f"\n{args.model} — validation metrics (threshold=0.5):")
    print_validation_metrics(metrics_default, LABEL_COLS)

    print("\nPer-label F1 with tuned thresholds:")
    for i, name in enumerate(LABEL_COLS):
        print(f"  {name:16s} f1={metrics_tuned[f'f1_{name}']:.4f}  thr={thresholds[i]:.2f}")
    print(f"  {'macro_f1':16s} {metrics_tuned['macro_f1']:.4f}")
    print(f"  {'micro_f1':16s} {metrics_tuned['micro_f1']:.4f}")

    summary: dict[str, float | int | str] = {
        "run_id": run_ts,
        "model": args.model,
        "checkpoint": str(checkpoint_dir) if checkpoint_dir is not None else "",
        "checkpoint_step": _checkpoint_step(checkpoint_dir) if checkpoint_dir is not None else -1,
        "n_samples": len(val_df),
        "val_size": args.val_size,
        "seed": args.seed,
        "cache_path": str(cache_path),
        "default_roc_auc_mean": metrics_default.get("roc_auc_mean", float("nan")),
        "default_pr_auc_mean": metrics_default.get("pr_auc_mean", float("nan")),
        "default_micro_f1": metrics_default.get("f1_micro", float("nan")),
        "default_macro_f1": metrics_default.get("f1_macro", float("nan")),
        "tuned_macro_f1": metrics_tuned["macro_f1"],
        "tuned_micro_f1": metrics_tuned["micro_f1"],
        "tuned_pr_auc_mean": metrics_tuned["pr_auc_mean"],
        "tuned_roc_auc_mean": metrics_tuned["roc_auc_mean"],
    }
    for i, name in enumerate(LABEL_COLS):
        key_roc = f"roc_auc_{name}"
        key_pr = f"pr_auc_{name}"
        if key_roc in metrics_default:
            summary[f"default_{key_roc}"] = metrics_default[key_roc]
        summary[f"default_{key_pr}"] = metrics_default.get(key_pr, float("nan"))
        summary[f"tuned_thr_{name}"] = float(thresholds[i])
        summary[f"tuned_f1_{name}"] = metrics_tuned[f"f1_{name}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path.resolve()}")

    meta_path = cache_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "run_id": run_ts,
                "model": args.model,
                "checkpoint": str(checkpoint_dir) if checkpoint_dir is not None else None,
                "n_samples": len(val_df),
                "thresholds": {name: float(thresholds[i]) for i, name in enumerate(LABEL_COLS)},
                "default_macro_f1": float(metrics_default.get("f1_macro", float("nan"))),
                "tuned_macro_f1": float(metrics_tuned["macro_f1"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
