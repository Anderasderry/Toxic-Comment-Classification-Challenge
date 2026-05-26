"""
Download a Hugging Face model into ./models/<name>/ for offline training.

Only fetches model.safetensors + tokenizer/config (~440MB for HateBERT),
not the duplicate pytorch_model.bin.

Example:
  export HF_ENDPOINT=https://hf-mirror.com
  python download_model.py GroNLP/hateBERT
  python train_hatebert.py --model-name ./models/hateBERT
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Training only needs one weight format; skip pytorch_model.bin (saves ~440MB).
ALLOW_PATTERNS = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "merges.txt",
]

WEIGHT_FILE = "model.safetensors"
MIN_WEIGHT_BYTES = 400_000_000  # ~440MB for BERT-base


def _local_dir(repo_id: str, models_root: Path) -> Path:
    return models_root / repo_id.split("/")[-1]


def _has_weights(model_dir: Path) -> bool:
    w = model_dir / WEIGHT_FILE
    return w.is_file() and w.stat().st_size >= MIN_WEIGHT_BYTES


# GroNLP/hateBERT pytorch_model.bin blob id (safetensors uses cb948732...).
_PYTORCH_BIN_BLOB = "8ad98e273f238555cf59f6056cb42a169d8c9648f660982d94011cdedf160721"


def _clear_stale_incomplete_cache(model_dir: Path) -> None:
    """Remove partial pytorch_model.bin only; keep model.safetensors resume progress."""
    cache_dl = model_dir / ".cache" / "huggingface" / "download"
    if not cache_dl.is_dir():
        return
    for path in list(cache_dl.glob("*.incomplete")) + list(cache_dl.glob("*.lock")):
        if _PYTORCH_BIN_BLOB in path.name or "pytorch_model" in path.name:
            path.unlink(missing_ok=True)
    for meta in cache_dl.glob(f"*{_PYTORCH_BIN_BLOB}*.metadata"):
        meta.unlink(missing_ok=True)
    print("Dropped stale pytorch_model.bin partials (progress bar should show ~440M, not 881M)")


def _ensure_hub_env() -> None:
    if not os.environ.get("HF_HUB_DISABLE_XET"):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("Set HF_HUB_DISABLE_XET=1 (avoids slow cas-bridge.xethub downloads)")
    if not os.environ.get("HF_ENDPOINT"):
        print(
            "Tip: export HF_ENDPOINT=https://hf-mirror.com  if downloads are slow",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        help="Hugging Face model id, e.g. GroNLP/hateBERT",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing folder and re-download.",
    )
    args = parser.parse_args()

    _ensure_hub_env()
    out = _local_dir(args.repo_id, args.models_dir.expanduser().resolve())

    if _has_weights(out) and not args.force:
        w = out / WEIGHT_FILE
        print(f"Already complete: {w} ({w.stat().st_size / 1e6:.0f} MB)")
        print(f"Train with: python train_hatebert.py --model-name {out}")
        return

    if args.force and out.is_dir():
        print(f"Removing {out} ...")
        shutil.rmtree(out)
    elif out.is_dir() and not _has_weights(out):
        print(f"Incomplete download at {out}; resuming (safetensors only, ~440MB)...")
        _clear_stale_incomplete_cache(out)

    print(f"Downloading {args.repo_id} -> {out}")
    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(out),
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "\nRetry (same command resumes partial downloads):\n"
            "  export HF_ENDPOINT=https://hf-mirror.com\n"
            f"  python download_model.py {args.repo_id}\n"
            "Or clean start:\n"
            f"  python download_model.py {args.repo_id} --force",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _has_weights(Path(path)):
        w = Path(path) / WEIGHT_FILE
        size = w.stat().st_size if w.is_file() else 0
        print(
            f"ERROR: {WEIGHT_FILE} missing or too small ({size} bytes). Re-run to resume.",
            file=sys.stderr,
        )
        sys.exit(1)

    w = Path(path) / WEIGHT_FILE
    print(f"Saved weights: {w} ({w.stat().st_size / 1e6:.0f} MB)")
    print(f"Train with: python train_hatebert.py --model-name {out}")


if __name__ == "__main__":
    main()
