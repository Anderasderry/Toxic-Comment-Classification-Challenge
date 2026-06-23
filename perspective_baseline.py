"""
Score validation comments with the Jigsaw Perspective API and compare metrics
against locally trained models.

Uses the same data paths, label columns, and validation split as
baseline_tfidf_lr.py (default val_size=0.1, random_state=42).

API key: set PERSPECTIVE_API_KEY in the environment, or pass --api-key.
Scores are cached under perspective_cache/ so reruns do not repeat API calls.

Example:
  export PERSPECTIVE_API_KEY="your_key"
  python perspective_baseline.py
  python perspective_baseline.py --max-samples 500 --sleep 0.3
  python perspective_baseline.py --max-samples 0   # entire validation split
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.model_selection import train_test_split

from baseline_tfidf_lr import (
    LABEL_COLS,
    _default_data_dir,
    _resolve_table,
    compute_validation_metrics,
    print_validation_metrics,
)
from grid_search_tfidf_lr import evaluate, search_best_thresholds
from run_paths import fig_run_dir, new_run_timestamp, path_with_timestamp
from viz import plot_transformer_eval

PROJECT_ROOT = Path(__file__).resolve().parent

PERSPECTIVE_API_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
)

# Perspective attribute names aligned with our LABEL_COLS order.
PERSPECTIVE_ATTR_MAP: dict[str, str] = {
    "toxic": "TOXICITY",
    "severe_toxic": "SEVERE_TOXICITY",
    "obscene": "OBSCENE",
    "threat": "THREAT",
    "insult": "INSULT",
    "identity_hate": "IDENTITY_ATTACK",
}

# Perspective documents a 3000-character limit for comment text.
MAX_COMMENT_CHARS = 3000


def build_http_session(retries: int) -> requests.Session:
    """Session tuned for long runs through proxies (SSL drops are retried)."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class PerspectiveAPIError(RuntimeError):
    pass


def _default_cache_path() -> Path:
    return PROJECT_ROOT / "perspective_cache" / "validation_scores.csv"


def _resolve_api_key(cli_key: str | None) -> str:
    key = (cli_key or os.environ.get("PERSPECTIVE_API_KEY", "")).strip()
    if not key:
        raise SystemExit(
            "Perspective API key missing. Set PERSPECTIVE_API_KEY or pass --api-key.",
        )
    return key


def score_comment(
    text: str,
    api_key: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    retries: int = 5,
    retry_backoff: float = 2.0,
) -> dict[str, float]:
    """Return six label probabilities from Perspective API."""
    payload = {
        "comment": {"text": text[:MAX_COMMENT_CHARS]},
        "languages": ["en"],
        "requestedAttributes": {attr: {} for attr in PERSPECTIVE_ATTR_MAP.values()},
    }
    client = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.post(
                PERSPECTIVE_API_URL,
                params={"key": api_key},
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 429:
                wait = retry_backoff ** attempt
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            attr_scores = data.get("attributeScores", {})
            return {
                label: float(attr_scores[attr]["summaryScore"]["value"])
                for label, attr in PERSPECTIVE_ATTR_MAP.items()
            }
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_backoff ** attempt)

    raise PerspectiveAPIError(f"Perspective API failed after {retries} attempts: {last_error}")


def load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.is_file():
        return pd.DataFrame(columns=["val_index", *LABEL_COLS])
    df = pd.read_csv(cache_path)
    if "val_index" not in df.columns:
        raise ValueError(f"Cache file missing val_index column: {cache_path}")
    return df


def save_cache(cache_path: Path, cache_df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_df.empty:
        return
    cache_df = cache_df.drop_duplicates(subset=["val_index"], keep="last")
    cache_df = cache_df.sort_values("val_index").reset_index(drop=True)
    cache_df.to_csv(cache_path, index=False)


def fetch_scores(
    val_df: pd.DataFrame,
    api_key: str,
    cache_path: Path,
    *,
    sleep_s: float,
    retries: int,
    comment_retries: int,
) -> pd.DataFrame:
    cache_df = load_cache(cache_path)
    cached_idx = set(cache_df["val_index"].astype(int).tolist()) if len(cache_df) else set()

    rows: list[dict[str, float | int]] = cache_df.to_dict("records") if len(cache_df) else []
    pending = [idx for idx in val_df.index if int(idx) not in cached_idx]
    total = len(pending)

    if total == 0:
        print(f"All {len(val_df)} validation rows already cached at {cache_path.resolve()}")
        return _scores_for_indices(cache_df, val_df.index)

    print(f"Fetching {total} / {len(val_df)} comments from Perspective API...")
    session = build_http_session(retries)

    for n, idx in enumerate(pending, start=1):
        text = str(val_df.at[idx, "comment_text"])
        scores: dict[str, float] | None = None
        last_exc: PerspectiveAPIError | None = None

        for comment_attempt in range(1, comment_retries + 1):
            try:
                scores = score_comment(
                    text,
                    api_key,
                    session=session,
                    retries=retries,
                )
                break
            except PerspectiveAPIError as exc:
                last_exc = exc
                if comment_attempt < comment_retries:
                    wait = min(30.0, 2.0 ** comment_attempt)
                    print(
                        f"  retry val_index={idx} ({comment_attempt}/{comment_retries}) "
                        f"after transient error; waiting {wait:.0f}s",
                        flush=True,
                    )
                    session.close()
                    session = build_http_session(retries)
                    time.sleep(wait)

        if scores is None:
            save_cache(cache_path, pd.DataFrame(rows))
            print(
                f"\nStopped at val_index={idx} ({n}/{total}): {last_exc}\n"
                f"Progress saved to {cache_path.resolve()}. "
                "Re-run the same command to resume from cache.",
            )
            raise SystemExit(1) from last_exc

        row: dict[str, float | int] = {"val_index": int(idx)}
        row.update(scores)
        rows.append(row)

        if n % 25 == 0 or n == total:
            save_cache(cache_path, pd.DataFrame(rows))
            print(f"  cached {n}/{total} new requests")

        if sleep_s > 0 and n < total:
            time.sleep(sleep_s)

        if n % 500 == 0:
            session.close()
            session = build_http_session(retries)

    save_cache(cache_path, pd.DataFrame(rows))
    return _scores_for_indices(pd.DataFrame(rows), val_df.index)


def _scores_for_indices(cache_df: pd.DataFrame, indices: pd.Index) -> np.ndarray:
    if cache_df.empty:
        raise ValueError("No cached Perspective scores available for the requested indices.")

    lookup = cache_df.set_index("val_index")
    missing = [int(i) for i in indices if int(i) not in lookup.index]
    if missing:
        raise ValueError(
            f"Cache is missing {len(missing)} validation indices, e.g. {missing[:5]}. "
            "Re-run without --scores-only or delete the cache file.",
        )

    return lookup.loc[[int(i) for i in indices], LABEL_COLS].to_numpy(dtype=np.float64)


def select_validation_rows(
    train_df: pd.DataFrame,
    *,
    val_size: float,
    seed: int,
    max_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_size <= 0:
        raise ValueError("perspective_baseline.py requires --val-size > 0.")

    _, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
    )

    if max_samples > 0 and max_samples < len(val_df):
        val_df = val_df.sample(n=max_samples, random_state=seed)
        print(
            f"Using {len(val_df)} / {int(len(train_df) * val_size)} validation comments "
            f"(--max-samples {max_samples})",
        )
    else:
        print(f"Using full validation split: {len(val_df)} comments")

    return train_df, val_df.sort_index()


def metrics_to_row(metrics: dict[str, float], *, prefix: str = "") -> dict[str, float]:
    rename = {
        "roc_auc_mean": "roc_auc_mean",
        "pr_auc_mean": "pr_auc_mean",
        "f1_micro": "micro_f1",
        "f1_macro": "macro_f1",
    }
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key in rename:
            out[f"{prefix}{rename[key]}"] = value
        elif key.startswith("roc_auc_") or key.startswith("pr_auc_"):
            out[f"{prefix}{key}"] = value
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
        help="Random seed for train/val split and subsampling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max validation comments to score (0 = use entire validation split).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Perspective API key (default: PERSPECTIVE_API_KEY env var).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=_default_cache_path(),
        help="CSV cache for per-comment Perspective scores.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between API requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries per HTTP request on transient API errors.",
    )
    parser.add_argument(
        "--comment-retries",
        type=int,
        default=5,
        help="Retries per comment after SSL/proxy drops (refreshes session).",
    )
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Evaluate from cache only; do not call the API.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "perspective_results.csv",
        help="CSV path for summary metrics (timestamp appended).",
    )
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=PROJECT_ROOT / "figs",
        help="Root directory for evaluation figures.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip saving ROC/PR/confusion-matrix plots.",
    )
    args = parser.parse_args()

    run_ts = new_run_timestamp()
    out_path = path_with_timestamp(args.out.expanduser().resolve(), run_ts)
    cache_path = args.cache.expanduser().resolve()
    save_figures = not args.no_figures
    fig_dir = fig_run_dir(args.figs_dir, "perspective", run_ts) if save_figures else None

    data_dir = args.data_dir.expanduser().resolve()
    train_path = _resolve_table(data_dir, "train")
    train_df = pd.read_csv(train_path)

    for col in LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(f"train.csv missing label column: {col}")
    if "comment_text" not in train_df.columns:
        raise ValueError("Expected comment_text column in train.csv.")

    _, val_df = select_validation_rows(
        train_df,
        val_size=args.val_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    y_val = val_df[LABEL_COLS].astype(np.float64)

    if args.scores_only:
        val_proba = _scores_for_indices(load_cache(cache_path), val_df.index)
    else:
        api_key = _resolve_api_key(args.api_key)
        val_proba = fetch_scores(
            val_df,
            api_key,
            cache_path,
            sleep_s=args.sleep,
            retries=args.retries,
            comment_retries=args.comment_retries,
        )

    metrics_default = compute_validation_metrics(
        y_val.to_numpy(),
        val_proba,
        LABEL_COLS,
        threshold=0.5,
    )
    metrics_tuned = evaluate(y_val, val_proba)
    thresholds = search_best_thresholds(y_val, val_proba)

    print("\nPerspective API — validation metrics (threshold=0.5):")
    print_validation_metrics(metrics_default, LABEL_COLS)

    print("\nPer-label F1 with tuned thresholds:")
    for i, name in enumerate(LABEL_COLS):
        print(f"  {name:16s} f1={metrics_tuned[f'f1_{name}']:.4f}  thr={thresholds[i]:.2f}")
    print(f"  {'macro_f1':16s} {metrics_tuned['macro_f1']:.4f}")
    print(f"  {'micro_f1':16s} {metrics_tuned['micro_f1']:.4f}")

    summary: dict[str, float | int | str] = {
        "run_id": run_ts,
        "n_samples": len(val_df),
        "val_size": args.val_size,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "cache_path": str(cache_path),
    }
    summary.update(metrics_to_row(metrics_default, prefix="default_"))
    summary.update(
        {
            "tuned_macro_f1": metrics_tuned["macro_f1"],
            "tuned_micro_f1": metrics_tuned["micro_f1"],
            "tuned_pr_auc_mean": metrics_tuned["pr_auc_mean"],
            "tuned_roc_auc_mean": metrics_tuned["roc_auc_mean"],
        },
    )
    for i, name in enumerate(LABEL_COLS):
        summary[f"tuned_thr_{name}"] = float(thresholds[i])
        summary[f"tuned_f1_{name}"] = metrics_tuned[f"f1_{name}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path.resolve()}")

    meta_path = cache_path.with_suffix(".json")
    meta = {
        "run_id": run_ts,
        "n_samples": len(val_df),
        "val_indices": [int(i) for i in val_df.index.tolist()],
        "thresholds": {name: float(thresholds[i]) for i, name in enumerate(LABEL_COLS)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if save_figures and fig_dir is not None:
        print(f"Figures directory: {fig_dir.resolve()}")
        plot_transformer_eval(
            y_val.to_numpy(),
            val_proba,
            LABEL_COLS,
            fig_dir,
            threshold=0.5,
        )


if __name__ == "__main__":
    main()
