"""
Score validation comments with LLM APIs (DeepSeek, Qwen/DashScope) for multi-label
toxicity classification and compare metrics against locally trained models.

Uses the same validation split as baseline_tfidf_lr.py (val_size=0.1, seed=42).

API keys (environment variables):
  DEEPSEEK_API_KEY   — DeepSeek OpenAI-compatible API
  DASHSCOPE_API_KEY  — Alibaba Cloud DashScope (Qwen)

Examples:
  export DEEPSEEK_API_KEY="sk-..."
  python llm_api_baseline.py --provider deepseek --max-samples 200

  export DASHSCOPE_API_KEY="sk-..."
  python llm_api_baseline.py --provider qwen --model qwen-plus --max-samples 200

  python llm_api_baseline.py --provider qwen --scores-only

Comments rejected by provider content filters (e.g. DashScope data_inspection_failed)
are skipped, cached with content_blocked=1, and excluded from metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.model_selection import train_test_split
from urllib3.util.retry import Retry

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
MAX_COMMENT_CHARS = 1500

PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "display_name": "DeepSeek",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "display_name": "Qwen (DashScope)",
    },
}

SYSTEM_PROMPT = (
    "You are an expert English content moderation classifier. "
    "Given a user comment, estimate the probability from 0.0 to 1.0 that the "
    "comment belongs to EACH of six independent toxicity categories. "
    "Categories may co-occur. Output ONLY valid JSON with exactly these keys: "
    "toxic, severe_toxic, obscene, threat, insult, identity_hate. "
    "Use float values in [0, 1]."
)


class LLMAPIError(RuntimeError):
    pass


class ContentBlockedError(LLMAPIError):
    """Provider rejected the input (e.g. DashScope data_inspection_failed)."""


def _parse_api_error(response: requests.Response) -> tuple[str, str | None]:
    try:
        body = response.json()
        err = body.get("error", {})
        if isinstance(err, dict):
            return str(err.get("message", response.text)), err.get("code")
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    return response.text[:500], None


def _blocked_placeholder_row(val_index: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {"val_index": val_index, "content_blocked": 1}
    row.update({label: float("nan") for label in LABEL_COLS})
    return row


def build_http_session(retries: int) -> requests.Session:
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


def _default_cache_path(provider: str, model: str) -> Path:
    safe_model = re.sub(r"[^\w.-]+", "_", model)
    return PROJECT_ROOT / "llm_api_cache" / provider / safe_model / "validation_scores.csv"


def _resolve_api_key(provider: str, cli_key: str | None) -> str:
    env_key = PROVIDER_CONFIG[provider]["env_key"]
    key = (cli_key or os.environ.get(env_key, "")).strip()
    if not key:
        raise SystemExit(
            f"{PROVIDER_CONFIG[provider]['display_name']} API key missing. "
            f"Set {env_key} or pass --api-key.",
        )
    return key


def _build_user_prompt(text: str) -> str:
    snippet = text[:MAX_COMMENT_CHARS].replace("\n", " ").strip()
    keys = ", ".join(LABEL_COLS)
    return (
        f"Comment:\n\"\"\"{snippet}\"\"\"\n\n"
        f"Return JSON with keys: {keys}. "
        "Each value is the estimated probability that the comment has that label."
    )


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    return data


def _normalize_scores(data: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in LABEL_COLS:
        if label not in data:
            raise KeyError(f"Missing label {label!r} in LLM JSON")
        value = float(data[label])
        out[label] = float(min(1.0, max(0.0, value)))
    return out


def score_comment(
    text: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    session: requests.Session,
    timeout: float = 60.0,
) -> dict[str, float]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(text)},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = session.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code == 429:
        raise LLMAPIError(f"Rate limited (429): {response.text[:300]}")
    if response.status_code >= 400:
        message, code = _parse_api_error(response)
        blocked_codes = {"data_inspection_failed"}
        if code in blocked_codes or "inappropriate content" in message.lower():
            raise ContentBlockedError(message)
        raise LLMAPIError(f"HTTP {response.status_code}: {message}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return _normalize_scores(_extract_json_object(content))


def load_cache(cache_path: Path) -> pd.DataFrame:
    columns = ["val_index", "content_blocked", *LABEL_COLS]
    if not cache_path.is_file():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(cache_path)
    if "val_index" not in df.columns:
        raise ValueError(f"Cache file missing val_index column: {cache_path}")
    if "content_blocked" not in df.columns:
        df["content_blocked"] = 0
    return df


def save_cache(cache_path: Path, cache_df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_df.empty:
        return
    if "content_blocked" not in cache_df.columns:
        cache_df = cache_df.copy()
        cache_df["content_blocked"] = 0
    cache_df = cache_df.drop_duplicates(subset=["val_index"], keep="last")
    cache_df = cache_df.sort_values("val_index").reset_index(drop=True)
    cache_df.to_csv(cache_path, index=False)


def _scores_for_indices(cache_df: pd.DataFrame, indices: pd.Index) -> np.ndarray:
    if cache_df.empty:
        raise ValueError("No cached LLM scores available for the requested indices.")
    lookup = cache_df.set_index("val_index")
    missing = [int(i) for i in indices if int(i) not in lookup.index]
    if missing:
        raise ValueError(
            f"Cache is missing {len(missing)} validation indices, e.g. {missing[:5]}. "
            "Re-run without --scores-only or delete the cache file.",
        )
    return lookup.loc[[int(i) for i in indices], LABEL_COLS].to_numpy(dtype=np.float64)


def fetch_scores(
    val_df: pd.DataFrame,
    *,
    provider: str,
    model: str,
    api_key: str,
    cache_path: Path,
    sleep_s: float,
    retries: int,
    comment_retries: int,
) -> np.ndarray:
    cfg = PROVIDER_CONFIG[provider]
    cache_df = load_cache(cache_path)
    cached_idx = set(cache_df["val_index"].astype(int).tolist()) if len(cache_df) else set()
    rows: list[dict[str, float | int]] = cache_df.to_dict("records") if len(cache_df) else []
    pending = [idx for idx in val_df.index if int(idx) not in cached_idx]
    total = len(pending)

    if total == 0:
        print(f"All {len(val_df)} validation rows already cached at {cache_path.resolve()}")
        return _scores_for_indices(cache_df, val_df.index)

    print(f"Fetching {total} / {len(val_df)} comments from {cfg['display_name']} ({model})...")
    session = build_http_session(retries)
    blocked_count = 0

    for n, idx in enumerate(pending, start=1):
        text = str(val_df.at[idx, "comment_text"])
        scores: dict[str, float] | None = None
        last_exc: Exception | None = None
        content_blocked = False

        for comment_attempt in range(1, comment_retries + 1):
            try:
                scores = score_comment(
                    text,
                    base_url=cfg["base_url"],
                    model=model,
                    api_key=api_key,
                    session=session,
                )
                break
            except ContentBlockedError as exc:
                content_blocked = True
                blocked_count += 1
                print(
                    f"  skipped val_index={idx}: content blocked by provider "
                    f"({exc})",
                    flush=True,
                )
                rows.append(_blocked_placeholder_row(int(idx)))
                break
            except (LLMAPIError, requests.RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_exc = exc
                if comment_attempt < comment_retries:
                    wait = min(30.0, 2.0 ** comment_attempt)
                    print(
                        f"  retry val_index={idx} ({comment_attempt}/{comment_retries}): "
                        f"{exc}; waiting {wait:.0f}s",
                        flush=True,
                    )
                    session.close()
                    session = build_http_session(retries)
                    time.sleep(wait)

        if content_blocked:
            if n % 10 == 0 or n == total:
                save_cache(cache_path, pd.DataFrame(rows))
                print(f"  cached {n}/{total} new requests ({blocked_count} blocked so far)")
            if sleep_s > 0 and n < total:
                time.sleep(sleep_s)
            continue

        if scores is None:
            save_cache(cache_path, pd.DataFrame(rows))
            print(
                f"\nStopped at val_index={idx} ({n}/{total}): {last_exc}\n"
                f"Progress saved to {cache_path.resolve()}. "
                "Re-run the same command to resume from cache.",
            )
            raise SystemExit(1) from last_exc

        row: dict[str, float | int] = {"val_index": int(idx), "content_blocked": 0}
        row.update(scores)
        rows.append(row)

        if n % 10 == 0 or n == total:
            save_cache(cache_path, pd.DataFrame(rows))
            msg = f"  cached {n}/{total} new requests"
            if blocked_count:
                msg += f" ({blocked_count} blocked so far)"
            print(msg)

        if sleep_s > 0 and n < total:
            time.sleep(sleep_s)

        if n % 200 == 0:
            session.close()
            session = build_http_session(retries)

    save_cache(cache_path, pd.DataFrame(rows))
    if blocked_count:
        print(f"Finished with {blocked_count} comment(s) blocked by provider content filter.")
    return _scores_for_indices(pd.DataFrame(rows), val_df.index)


def select_validation_rows(
    train_df: pd.DataFrame,
    *,
    val_size: float,
    seed: int,
    max_samples: int,
) -> pd.DataFrame:
    if val_size <= 0:
        raise ValueError("llm_api_baseline.py requires --val-size > 0.")
    _, val_df = train_test_split(train_df, test_size=val_size, random_state=seed)
    if max_samples > 0 and max_samples < len(val_df):
        val_df = val_df.sample(n=max_samples, random_state=seed)
        print(
            f"Using {len(val_df)} / {int(len(train_df) * val_size)} validation comments "
            f"(--max-samples {max_samples})",
        )
    else:
        print(f"Using full validation split: {len(val_df)} comments")
    return val_df.sort_index()


def blocked_mask_for_indices(cache_df: pd.DataFrame, indices: pd.Index) -> np.ndarray:
    lookup = cache_df.set_index("val_index")
    if "content_blocked" not in lookup.columns:
        return np.zeros(len(indices), dtype=bool)
    blocked = lookup.loc[[int(i) for i in indices], "content_blocked"].fillna(0).astype(int)
    return blocked.to_numpy(dtype=bool)


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
        "--provider",
        choices=sorted(PROVIDER_CONFIG),
        required=True,
        help="LLM API provider: deepseek or qwen (DashScope).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: deepseek-chat or qwen-plus).",
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
        "--max-samples",
        type=int,
        default=200,
        help="Max validation comments (default 200; LLM calls cost money).",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--comment-retries", type=int, default=3)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "llm_api_results.csv",
    )
    parser.add_argument("--figs-dir", type=Path, default=PROJECT_ROOT / "figs")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = PROVIDER_CONFIG[args.provider]
    model = args.model or cfg["default_model"]
    run_ts = new_run_timestamp()
    slug = f"{args.provider}_{re.sub(r'[^\w.-]+', '_', model)}"
    out_path = path_with_timestamp(args.out.expanduser().resolve(), run_ts)
    cache_path = (
        args.cache.expanduser().resolve()
        if args.cache is not None
        else _default_cache_path(args.provider, model)
    )
    save_figures = not args.no_figures
    fig_dir = fig_run_dir(args.figs_dir, slug, run_ts) if save_figures else None

    data_dir = args.data_dir.expanduser().resolve()
    train_df = pd.read_csv(_resolve_table(data_dir, "train"))
    for col in LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(f"train.csv missing label column: {col}")
    if "comment_text" not in train_df.columns:
        raise ValueError("Expected comment_text column in train.csv.")

    val_df = select_validation_rows(
        train_df,
        val_size=args.val_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    y_val = val_df[LABEL_COLS].astype(np.float64)

    cache_df = load_cache(cache_path)

    if args.scores_only:
        val_proba = _scores_for_indices(cache_df, val_df.index)
    else:
        api_key = _resolve_api_key(args.provider, args.api_key)
        val_proba = fetch_scores(
            val_df,
            provider=args.provider,
            model=model,
            api_key=api_key,
            cache_path=cache_path,
            sleep_s=args.sleep,
            retries=args.retries,
            comment_retries=args.comment_retries,
        )
        cache_df = load_cache(cache_path)

    blocked = blocked_mask_for_indices(cache_df, val_df.index)
    n_blocked = int(blocked.sum())
    eval_mask = ~blocked
    n_eval = int(eval_mask.sum())

    if n_blocked:
        print(
            f"\n{n_blocked} comment(s) blocked by provider content filter; "
            f"metrics computed on {n_eval} scored comments.",
        )
    if n_eval == 0:
        raise SystemExit("No scored comments available for evaluation.")

    y_eval = y_val.to_numpy()[eval_mask]
    p_eval = val_proba[eval_mask]
    y_val_eval = y_val.iloc[eval_mask]

    metrics_default = compute_validation_metrics(
        y_eval, p_eval, LABEL_COLS, threshold=0.5,
    )
    metrics_tuned = evaluate(y_val_eval, p_eval)
    thresholds = search_best_thresholds(y_val_eval, p_eval)

    display = cfg["display_name"]
    print(f"\n{display} ({model}) — validation metrics (threshold=0.5):")
    print_validation_metrics(metrics_default, LABEL_COLS)

    print("\nPer-label F1 with tuned thresholds:")
    for i, name in enumerate(LABEL_COLS):
        print(f"  {name:16s} f1={metrics_tuned[f'f1_{name}']:.4f}  thr={thresholds[i]:.2f}")
    print(f"  {'macro_f1':16s} {metrics_tuned['macro_f1']:.4f}")
    print(f"  {'micro_f1':16s} {metrics_tuned['micro_f1']:.4f}")

    summary: dict[str, float | int | str] = {
        "run_id": run_ts,
        "provider": args.provider,
        "model": model,
        "n_samples": len(val_df),
        "n_scored": n_eval,
        "n_blocked": n_blocked,
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
    meta_path.write_text(
        json.dumps(
            {
                "run_id": run_ts,
                "provider": args.provider,
                "model": model,
                "n_samples": len(val_df),
                "n_scored": n_eval,
                "n_blocked": n_blocked,
                "val_indices": [int(i) for i in val_df.index.tolist()],
                "thresholds": {name: float(thresholds[i]) for i, name in enumerate(LABEL_COLS)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if save_figures and fig_dir is not None:
        print(f"Figures directory: {fig_dir.resolve()}")
        plot_transformer_eval(y_eval, p_eval, LABEL_COLS, fig_dir, threshold=0.5)


if __name__ == "__main__":
    main()
