from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import kagglehub


COMPETITION_SLUG = "jigsaw-toxic-comment-classification-challenge"


def _default_datasets_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def download_competition(
    *,
    force: bool = False,
    output_dir: Path,
) -> str:
    """Download and extract competition files; returns local directory path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return kagglehub.competition_download(
        COMPETITION_SLUG,
        force_download=force,
        output_dir=str(output_dir.resolve()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Jigsaw Toxic Comment Classification competition data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a cached copy exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_datasets_dir(),
        metavar="DIR",
        help="Directory to extract files into (default: ./datasets next to this script).",
    )
    args = parser.parse_args()

    if not os.environ.get("KAGGLE_API_TOKEN", "").strip():
        print(
            "KAGGLE_API_TOKEN is not set. Configure it in your shell or ~/.bashrc, e.g.\n"
            '  export KAGGLE_API_TOKEN="KGAT_..."',
            file=sys.stderr,
        )
        sys.exit(1)

    out = args.output_dir.expanduser().resolve()

    try:
        path = download_competition(force=args.force, output_dir=out)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Data directory:", path)


if __name__ == "__main__":
    main()
