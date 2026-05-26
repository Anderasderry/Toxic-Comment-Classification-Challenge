"""
Fine-tune a Hugging Face transformer for multi-label toxic comment classification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

from run_paths import fig_run_dir, new_run_timestamp, path_with_timestamp
from viz import plot_label_statistics, plot_transformer_eval, plot_transformer_training

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_data_dir() -> Path:
    return PROJECT_ROOT / "datasets"


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


class ToxicCommentDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        labels: pd.DataFrame | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts.astype(str).tolist()
        self.labels = None if labels is None else labels.astype(np.float32).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics_builder(label_cols: list[str]):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        probs = sigmoid(np.asarray(logits, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.float64)

        metrics: dict[str, float] = {}
        aucs: list[float] = []
        prs: list[float] = []

        for i, name in enumerate(label_cols):
            y = labels[:, i]
            p = probs[:, i]
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, p)
            pr = average_precision_score(y, p)
            metrics[f"roc_auc_{name}"] = auc
            metrics[f"pr_auc_{name}"] = pr
            aucs.append(auc)
            prs.append(pr)

        if aucs:
            metrics["roc_auc_mean"] = float(np.mean(aucs))
            metrics["pr_auc_mean"] = float(np.mean(prs))

        preds = (probs >= 0.5).astype(np.int32)
        metrics["f1_micro"] = float(f1_score(labels, preds, average="micro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
        return metrics

    return compute_metrics


def print_validation_metrics(metrics: dict[str, float], label_cols: list[str]) -> None:
    print("Validation ROC-AUC (per label):")
    for name in label_cols:
        key = f"eval_roc_auc_{name}"
        if key in metrics:
            print(f"  {name:16s} {metrics[key]:.4f}")
    if "eval_roc_auc_mean" in metrics:
        print(f"  {'mean':16s} {metrics['eval_roc_auc_mean']:.4f}")
    if "eval_pr_auc_mean" in metrics:
        print(f"  {'pr_auc_mean':16s} {metrics['eval_pr_auc_mean']:.4f}")
    if "eval_f1_micro" in metrics:
        print(f"  {'f1_micro':16s} {metrics['eval_f1_micro']:.4f}")
    if "eval_f1_macro" in metrics:
        print(f"  {'f1_macro':16s} {metrics['eval_f1_macro']:.4f}")


def _create_trainer(tokenizer, **kwargs) -> Trainer:
    """Support both tokenizer= (legacy) and processing_class= (new transformers)."""
    try:
        return Trainer(tokenizer=tokenizer, **kwargs)
    except TypeError as exc:
        if "tokenizer" in str(exc):
            return Trainer(processing_class=tokenizer, **kwargs)
        raise


def get_trainer_tokenizer(trainer: Trainer):
    return getattr(trainer, "processing_class", None) or trainer.tokenizer


def _training_arguments(**kwargs) -> TrainingArguments:
    """Support both eval_strategy (new) and evaluation_strategy (legacy)."""
    try:
        return TrainingArguments(**kwargs)
    except TypeError as exc:
        if "evaluation_strategy" in kwargs and "evaluation_strategy" in str(exc):
            legacy = dict(kwargs)
            legacy["eval_strategy"] = legacy.pop("evaluation_strategy")
            return TrainingArguments(**legacy)
        if "eval_strategy" in kwargs and "eval_strategy" in str(exc):
            legacy = dict(kwargs)
            legacy["evaluation_strategy"] = legacy.pop("eval_strategy")
            return TrainingArguments(**legacy)
        raise


def build_trainer(
    *,
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    output_dir: Path,
    label_cols: list[str],
    epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    max_steps: int = -1,
) -> Trainer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
    )

    use_eval = eval_dataset is not None
    training_args = _training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch" if use_eval else "no",
        save_strategy="epoch" if use_eval else "no",
        load_best_model_at_end=use_eval,
        metric_for_best_model="roc_auc_mean" if use_eval else None,
        greater_is_better=True,
        logging_steps=200,
        save_total_limit=1,
        seed=seed,
        fp16=fp16,
        report_to="none",
    )

    return _create_trainer(
        tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_builder(label_cols) if use_eval else None,
    )


def predict_proba(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    try:
        pred = trainer.predict(dataset)
    except TypeError:
        pred = trainer.predict(
            dataset,
            batch_size=trainer.args.per_device_eval_batch_size,
        )
    return sigmoid(pred.predictions)


@dataclass(frozen=True)
class TrainDefaults:
    model_slug: str
    model_name: str
    submission: Path
    output_dir: Path
    batch_size: int = 16
    description: str = "Fine-tune a transformer for multi-label toxic comments."


DISTILBERT_DEFAULTS = TrainDefaults(
    model_slug="distilbert",
    model_name="distilbert-base-uncased",
    submission=PROJECT_ROOT / "submission_distilbert.csv",
    output_dir=PROJECT_ROOT / "checkpoints" / "distilbert",
    batch_size=16,
    description="Fine-tune DistilBERT for multi-label toxic comment classification.",
)

def _default_hatebert_model() -> str:
    local = PROJECT_ROOT / "models" / "hateBERT"
    weight = local / "model.safetensors"
    if weight.is_file() and weight.stat().st_size >= 400_000_000:
        return str(local)
    return "GroNLP/hateBERT"


HATEBERT_DEFAULTS = TrainDefaults(
    model_slug="hatebert",
    model_name=_default_hatebert_model(),
    submission=PROJECT_ROOT / "submission_hatebert.csv",
    output_dir=PROJECT_ROOT / "checkpoints" / "hatebert",
    batch_size=8,
    description="Fine-tune HateBERT for multi-label toxic comment classification.",
)


def main(defaults: TrainDefaults = DISTILBERT_DEFAULTS) -> None:
    parser = argparse.ArgumentParser(description=defaults.description)
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
        help="Holdout fraction for local metrics (0 skips validation).",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=defaults.submission,
        help="Output submission CSV path (a run timestamp is appended to the filename).",
    )
    parser.add_argument(
        "--model-name",
        default=defaults.model_name,
        help=f"Hugging Face model id (default: {defaults.model_name}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="Directory for Trainer checkpoints.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If >0, subsample training rows (debug).",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed-precision training even on GPU.",
    )
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=PROJECT_ROOT / "figs",
        help="Root directory for run figures (figs/<model>_<timestamp>/).",
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
    fig_dir = (
        fig_run_dir(args.figs_dir, defaults.model_slug, run_ts) if save_figures else None
    )

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

    if args.max_train_samples > 0:
        train_df = train_df.sample(
            n=min(args.max_train_samples, len(train_df)),
            random_state=args.seed,
        )

    texts = train_df["comment_text"]
    labels = train_df[LABEL_COLS]
    fp16 = torch.cuda.is_available() and not args.no_fp16

    print(f"Model: {args.model_name}")
    if save_figures:
        if args.val_size <= 0:
            print(
                "Warning: --val-size 0 skips epoch curves and validation plots; "
                "only label statistics will be saved.",
                flush=True,
            )
        plot_label_statistics(labels.values, LABEL_COLS, fig_dir)
        print(f"Figures directory: {fig_dir.resolve()}")

    if args.val_size > 0:
        idx_tr, idx_val = train_test_split(
            train_df.index,
            test_size=args.val_size,
            random_state=args.seed,
        )
        tr_df = train_df.loc[idx_tr]
        val_df = train_df.loc[idx_val]

        print("Training on train split for validation...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_ds = ToxicCommentDataset(
            tr_df["comment_text"],
            tr_df[LABEL_COLS],
            tokenizer,
            args.max_length,
        )
        val_ds = ToxicCommentDataset(
            val_df["comment_text"],
            val_df[LABEL_COLS],
            tokenizer,
            args.max_length,
        )
        val_trainer = build_trainer(
            model_name=args.model_name,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            output_dir=args.output_dir / "val",
            label_cols=LABEL_COLS,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            fp16=fp16,
        )
        val_trainer.train()
        val_metrics = val_trainer.evaluate()
        print_validation_metrics(val_metrics, LABEL_COLS)

        if save_figures and fig_dir is not None:
            plot_transformer_training(val_trainer.state.log_history, fig_dir)
            val_proba = predict_proba(val_trainer, val_ds)
            plot_transformer_eval(
                val_df[LABEL_COLS].values.astype(np.float64),
                val_proba,
                LABEL_COLS,
                fig_dir,
            )

        print("Refitting on full training data...")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        full_trainer = build_trainer(
            model_name=args.model_name,
            train_dataset=ToxicCommentDataset(
                texts,
                labels,
                AutoTokenizer.from_pretrained(args.model_name),
                args.max_length,
            ),
            eval_dataset=None,
            output_dir=args.output_dir / "final",
            label_cols=LABEL_COLS,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            fp16=fp16,
        )
        full_trainer.train()
        predict_trainer = full_trainer
    else:
        print("Training on full training data...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        predict_trainer = build_trainer(
            model_name=args.model_name,
            train_dataset=ToxicCommentDataset(
                texts,
                labels,
                tokenizer,
                args.max_length,
            ),
            eval_dataset=None,
            output_dir=args.output_dir,
            label_cols=LABEL_COLS,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            fp16=fp16,
        )
        predict_trainer.train()

    print("Predicting test set...")
    test_ds = ToxicCommentDataset(
        test_df["comment_text"],
        None,
        get_trainer_tokenizer(predict_trainer),
        args.max_length,
    )
    test_proba = predict_proba(predict_trainer, test_ds)

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
