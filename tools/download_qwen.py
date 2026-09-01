#!/usr/bin/env python3
"""Download the Qwen backbone required by MiniOneRec-Reproduction.

The model weights are hosted by Hugging Face rather than GitHub.  This script
uses huggingface_hub's resumable snapshot download and stores a complete local
Transformers model under ``models/Qwen2.5-1.5B`` by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Qwen/Qwen2.5-1.5B"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-1.5B for MiniOneRec-Reproduction."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repository (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models" / "Qwen2.5-1.5B",
        help="Directory in which to place the complete model snapshot.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face-compatible endpoint, such as a mirror.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token; prefer setting HF_TOKEN in the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    print(f"Downloading {args.repo_id} -> {output_dir}")
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=output_dir,
        token=args.token,
    )
    print(f"Model is ready at: {snapshot_path}")


if __name__ == "__main__":
    main()
