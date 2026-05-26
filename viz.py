"""
Plots for toxic comment classification (saved under figs/<model>_<timestamp>/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Baseline (final validation only: confusion matrix + ROC)
# ---------------------------------------------------------------------------


def plot_baseline_final_eval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: list[str],
    fig_dir: Path,
    *,
    threshold: float = 0.5,
) -> None:
    """Confusion matrices (grid) + per-label ROC curves on validation set."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    plot_confusion_matrices_grid(y_true, y_pred, label_cols, fig_dir / "confusion_matrices.png")
    plot_roc_curves(y_true, y_prob, label_cols, fig_dir / "roc_curves.png")


# ---------------------------------------------------------------------------
# Transformer: training curves + full validation diagnostics
# ---------------------------------------------------------------------------


def plot_transformer_training(log_history: list[dict], fig_dir: Path) -> None:
    plot_loss_curves(log_history, fig_dir / "loss_curves.png")
    plot_eval_metrics_per_epoch(log_history, fig_dir / "metrics_per_epoch.png")
    plot_learning_rate(log_history, fig_dir / "learning_rate.png")


def plot_transformer_eval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: list[str],
    fig_dir: Path,
    *,
    threshold: float = 0.5,
) -> None:
    y_pred = (y_prob >= threshold).astype(np.int32)
    plot_confusion_matrices_grid(y_true, y_pred, label_cols, fig_dir / "confusion_matrices.png")
    plot_roc_curves(y_true, y_prob, label_cols, fig_dir / "roc_curves.png")
    plot_pr_curves(y_true, y_prob, label_cols, fig_dir / "pr_curves.png")
    plot_per_label_metric_bars(y_true, y_prob, y_pred, label_cols, fig_dir / "per_label_metrics.png")
    plot_score_histograms(y_true, y_prob, label_cols, fig_dir)
    plot_calibration_curves(y_true, y_prob, label_cols, fig_dir / "calibration_curves.png")


def plot_label_statistics(y: np.ndarray, label_cols: list[str], fig_dir: Path) -> None:
    plot_label_frequency(y, label_cols, fig_dir / "label_frequency.png")
    plot_label_cooccurrence(y, label_cols, fig_dir / "label_cooccurrence.png")


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def plot_loss_curves(log_history: list[dict], out_path: Path) -> None:
    train_steps, train_loss = [], []
    eval_epochs, eval_loss = [], []

    for row in log_history:
        if "loss" in row and "eval_loss" not in row:
            train_steps.append(row.get("step", len(train_steps)))
            train_loss.append(row["loss"])
        if "eval_loss" in row:
            eval_epochs.append(row.get("epoch", len(eval_epochs)))
            eval_loss.append(row["eval_loss"])

    if not train_loss and not eval_loss:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_loss:
        ax.plot(train_steps, train_loss, label="train loss", alpha=0.8)
    if eval_loss:
        ax.plot(eval_epochs, eval_loss, "o-", label="eval loss", linewidth=2)
    ax.set_xlabel("step (train) / epoch (eval)")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    _save(fig, out_path)


def plot_eval_metrics_per_epoch(log_history: list[dict], out_path: Path) -> None:
    epochs: list[float] = []
    series: dict[str, list[float]] = {}

    metric_keys = (
        "eval_roc_auc_mean",
        "eval_pr_auc_mean",
        "eval_f1_macro",
        "eval_f1_micro",
    )
    for row in log_history:
        if not any(k in row for k in metric_keys):
            continue
        epochs.append(row.get("epoch", len(epochs)))
        for k in metric_keys:
            if k in row:
                series.setdefault(k, []).append(row[k])

    if not series:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {
        "eval_roc_auc_mean": "ROC-AUC (mean)",
        "eval_pr_auc_mean": "PR-AUC (mean)",
        "eval_f1_macro": "F1 (macro)",
        "eval_f1_micro": "F1 (micro)",
    }
    for k, vals in series.items():
        ax.plot(epochs[: len(vals)], vals, "o-", label=labels.get(k, k), linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Validation metrics per epoch")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, out_path)


def plot_learning_rate(log_history: list[dict], out_path: Path) -> None:
    steps, lrs = [], []
    for row in log_history:
        if "learning_rate" in row:
            steps.append(row.get("step", len(steps)))
            lrs.append(row["learning_rate"])
    if not lrs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs, color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    _save(fig, out_path)


def plot_confusion_matrices_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: list[str],
    out_path: Path,
) -> None:
    n = len(label_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(label_cols):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=False,
            xticklabels=["pred 0", "pred 1"],
            yticklabels=["true 0", "true 1"],
        )
        axes[i].set_title(name)

    for j in range(len(label_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion matrices (threshold=0.5)", y=1.02)
    fig.tight_layout()
    _save(fig, out_path)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(label_cols):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curves (validation)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_path)


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(label_cols):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            continue
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(y, y_prob[:, i])
        ap = average_precision_score(y, y_prob[:, i])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–recall curves (validation)")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, out_path)


def plot_per_label_metric_bars(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_cols: list[str],
    out_path: Path,
) -> None:
    roc_scores, pr_scores, f1_scores = [], [], []
    names = []
    for i, name in enumerate(label_cols):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            continue
        names.append(name)
        roc_scores.append(roc_auc_score(y, y_prob[:, i]))
        pr_scores.append(average_precision_score(y, y_prob[:, i]))
        f1_scores.append(f1_score(y, y_pred[:, i], zero_division=0))

    if not names:
        return

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, roc_scores, w, label="ROC-AUC")
    ax.bar(x, pr_scores, w, label="PR-AUC")
    ax.bar(x + w, f1_scores, w, label="F1 @0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-label validation metrics")
    ax.legend()
    _save(fig, out_path)


def plot_score_histograms(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: list[str],
    fig_dir: Path,
) -> None:
    n = len(label_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(label_cols):
        ax = axes[i]
        scores = y_prob[:, i]
        neg, pos = scores[y_true[:, i] == 0], scores[y_true[:, i] == 1]
        if len(neg):
            ax.hist(neg, bins=30, alpha=0.6, label="negative", density=True)
        if len(pos):
            ax.hist(pos, bins=30, alpha=0.6, label="positive", density=True)
        ax.set_title(name)
        ax.set_xlabel("predicted probability")
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(label_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Score distributions by true label", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "score_histograms.png")


def plot_calibration_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: list[str],
    out_path: Path,
) -> None:
    n = len(label_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(label_cols):
        ax = axes[i]
        y = y_true[:, i]
        if len(np.unique(y)) < 2 or y.sum() < 5:
            ax.set_title(f"{name} (skipped)")
            ax.axis("off")
            continue
        prob_true, prob_pred = calibration_curve(y, y_prob[:, i], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", label=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(name)
        ax.set_xlabel("mean predicted prob")
        ax.set_ylabel("fraction positive")

    for j in range(len(label_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Calibration curves (validation)", y=1.02)
    fig.tight_layout()
    _save(fig, out_path)


def plot_label_frequency(y: np.ndarray, label_cols: list[str], out_path: Path) -> None:
    rates = y.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(label_cols, rates, color="steelblue")
    ax.set_ylabel("positive rate")
    ax.set_title("Label frequency in training set")
    ax.set_ylim(0, max(rates.max() * 1.15, 0.05))
    plt.xticks(rotation=30, ha="right")
    _save(fig, out_path)


def plot_label_cooccurrence(y: np.ndarray, label_cols: list[str], out_path: Path) -> None:
    co = np.zeros((len(label_cols), len(label_cols)), dtype=np.float64)
    for i in range(len(label_cols)):
        for j in range(len(label_cols)):
            co[i, j] = np.mean((y[:, i] == 1) & (y[:, j] == 1))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        co,
        annot=True,
        fmt=".3f",
        xticklabels=label_cols,
        yticklabels=label_cols,
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
    )
    ax.set_title("Label co-occurrence (training set)")
    _save(fig, out_path)


def plot_tfidf_top_terms(
    pipeline,
    label_cols: list[str],
    out_path: Path,
    *,
    top_k: int = 12,
) -> None:
    """Top positive/negative TF-IDF features per label (baseline interpretability)."""
    tfidf = pipeline.named_steps["tfidf"]
    clfs = pipeline.named_steps["clf"].estimators_
    names = np.array(tfidf.get_feature_names_out())

    n = len(label_cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (label, clf) in enumerate(zip(label_cols, clfs)):
        coef = clf.coef_.ravel()
        order = np.argsort(coef)
        neg_terms = names[order[:top_k]]
        pos_terms = names[order[-top_k:][::-1]]
        neg_vals = coef[order[:top_k]]
        pos_vals = coef[order[-top_k:][::-1]]

        ax_neg, ax_pos = axes[i, 0], axes[i, 1]
        ax_neg.barh(range(top_k), neg_vals, color="tab:red", alpha=0.8)
        ax_neg.set_yticks(range(top_k))
        ax_neg.set_yticklabels(neg_terms, fontsize=7)
        ax_neg.set_title(f"{label}: anti-toxic")
        ax_neg.invert_yaxis()

        ax_pos.barh(range(top_k), pos_vals, color="tab:blue", alpha=0.8)
        ax_pos.set_yticks(range(top_k))
        ax_pos.set_yticklabels(pos_terms, fontsize=7)
        ax_pos.set_title(f"{label}: toxic indicators")
        ax_pos.invert_yaxis()

    fig.suptitle("Top TF-IDF coefficients per label", y=1.01)
    fig.tight_layout()
    _save(fig, out_path)
