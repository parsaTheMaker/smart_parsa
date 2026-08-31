#!/usr/bin/env python3
"""Resumably download a full or seeded subset Hugging Face dataset snapshot."""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent Hub file transfers. Keep this modest on shared DNS/network infrastructure.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=12,
        help="Total snapshot attempts after transient Hub/DNS failures; completed files are reused.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=15.0,
        help="Initial retry delay; subsequent delays use capped exponential backoff.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of sample_* directories to download; 0 downloads the complete repository.",
    )
    parser.add_argument("--selection-mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_sample_directories(repo_id: str, revision: str, token: str) -> list[str]:
    """List all top-level sample directories, following Hub API pagination."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}"
    headers = {"Authorization": f"Bearer {token}"}
    params: dict[str, object] | None = {
        "recursive": "false",
        "expand": "false",
        "limit": 1000,
    }
    names: set[str] = set()
    while url:
        response = requests.get(url, params=params, headers=headers, timeout=120)
        if response.status_code in {401, 403}:
            raise RuntimeError(
                "Hugging Face denied the dataset listing. Confirm that this account accepted the Pump dataset conditions."
            )
        response.raise_for_status()
        for row in response.json():
            path = str(row.get("path", ""))
            if row.get("type") == "directory" and path.startswith("sample_") and "/" not in path:
                names.add(path)
        next_link = response.links.get("next", {}).get("url")
        url = str(next_link) if next_link else ""
        params = None
    if not names:
        raise RuntimeError(f"No top-level sample_* directories found in {repo_id}@{revision}.")
    return sorted(names)


def select_samples(args: argparse.Namespace, output_dir: Path, token: str) -> list[str]:
    """Create or reuse an auditable deterministic subset selection."""
    manifest_path = output_dir / "selected_samples.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = [str(name) for name in manifest.get("samples", [])]
        expected = {
            "repo_id": str(args.repo_id),
            "revision": str(args.revision),
            "sample_count": int(args.sample_count),
            "selection_mode": str(args.selection_mode),
            "seed": int(args.seed),
        }
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Existing subset manifest {manifest_path} has a different selection definition: "
                f"expected={expected}, actual={actual}. Use a new --output-dir for a different subset."
            )
        if selected:
            return selected

    available = list_sample_directories(args.repo_id, args.revision, token)
    if int(args.sample_count) <= 0:
        selected = available
    elif int(args.sample_count) > len(available):
        raise ValueError(f"Requested {args.sample_count} samples, but the repository has only {len(available)}.")
    elif args.selection_mode == "sequential":
        selected = available[: int(args.sample_count)]
    else:
        selected = sorted(
            np.random.default_rng(int(args.seed)).choice(
                np.asarray(available, dtype=object), size=int(args.sample_count), replace=False
            ).tolist()
        )
    manifest_path.write_text(
        json.dumps(
            {
                "repo_id": str(args.repo_id),
                "revision": str(args.revision),
                "sample_count": int(args.sample_count),
                "selection_mode": str(args.selection_mode),
                "seed": int(args.seed),
                "available_sample_count": len(available),
                "samples": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected


def download_snapshot_with_retries(args: argparse.Namespace, token: str, allow_patterns: list[str] | None) -> None:
    """Retry the whole snapshot operation without discarding completed files.

    ``snapshot_download`` resumes its local files and cache, but a transient DNS
    failure in any worker otherwise aborts the enclosing thread pool. Retrying
    the complete call is therefore safe and resumes only unfinished files.
    """
    last_error: Exception | None = None
    for attempt in range(1, int(args.retry_attempts) + 1):
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=str(args.output_dir),
                token=token,
                max_workers=int(args.workers),
                allow_patterns=allow_patterns,
            )
            return
        except Exception as error:
            last_error = error
            if attempt >= int(args.retry_attempts):
                break
            delay = min(float(args.retry_backoff_seconds) * (2 ** (attempt - 1)), 300.0)
            print(
                f"[retry {attempt}/{args.retry_attempts}] Snapshot interrupted: {type(error).__name__}: {error}. "
                f"Resuming completed files in {delay:.0f}s.",
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(
        f"Snapshot failed after {args.retry_attempts} resumable attempts. "
        "The partial local directory was preserved for a later retry."
    ) from last_error


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.sample_count < 0 or args.retry_attempts <= 0 or args.retry_backoff_seconds < 0:
        raise ValueError("--workers and --retry-attempts must be positive; --retry-backoff-seconds must be nonnegative.")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN to an account token that has accepted this dataset's access conditions.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    available = os.statvfs(output_dir.parent)
    available_gib = available.f_bavail * available.f_frsize / 2**30
    print(
        f"Snapshot target: {output_dir}\n"
        f"Repository: {args.repo_id}@{args.revision}\n"
        f"Available space at target: {available_gib:,.1f} GiB\n"
        f"Workers: {args.workers}\n"
        "Existing files are reused; interrupted downloads resume in place.",
        flush=True,
    )
    if args.dry_run:
        return 0

    # The subset manifest is written before any large transfer so a retried
    # invocation selects exactly the same cases. Create the target itself
    # before writing that manifest.
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_samples(args, output_dir, token)
    print(f"Downloading {len(selected):,} selected sample directories.", flush=True)
    allow_patterns = ["README.md", "LICENSE", "metadata.json", ".gitattributes"]
    if int(args.sample_count) > 0:
        allow_patterns.extend(f"{sample}/*" for sample in selected)

    # Keep Hub metadata beside the snapshot instead of filling the default
    # home-directory cache on the system disk.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    download_snapshot_with_retries(args, token, allow_patterns if int(args.sample_count) > 0 else None)
    print(f"Snapshot complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
