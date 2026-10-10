"""Anonymous download and CPU verification of the existing published evidence.

From an installed checkout, using Python 3.12:
  python -m pip install -e . -r requirements/cpu.txt huggingface_hub==1.19.0
  python docs/huggingface/cpu_reproduce.py --output runs/hf-verify-v1

The default restores saved models and verifies 39 statistics; it does not fit.
Add --refit only for the unchanged historical CPU refitting procedure. No model
weights, raw text, original identities, test data, GPU or cloud resources are used.
Partial downloads and failures are preserved. An existing output is never reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ID = "mangesh-ux/thinking-completion-probes"
DEFAULT_REVISION = "v1.0.0"
HUB_VERSION = "1.19.0"  # Delivery dependency, not part of the frozen numerical stack.
EXPECTED_MANIFEST_SHA256 = "fed6a224ef7b6576de47f27cc2d5ef42e171999df57ffc1aa66dfb2e3f431c7f"
PUBLIC_FILES = ("manifest.json", "train.npz", "validation.npz",
                "fitted-models.json", "heldout-predictions.npz")
ENDPOINT = "https://huggingface.co"


def _revision(value: str) -> str:
    if value != DEFAULT_REVISION and re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("Revision must be v1.0.0 or a full lowercase commit SHA")
    return value


def _write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run(output: Path, *, revision: str = DEFAULT_REVISION, refit: bool = False) -> dict:
    """Create a fresh output; anonymously retrieve only the five numeric assets."""
    revision = _revision(revision)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists; choose a fresh directory")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "repo_id": REPO_ID, "requested_revision": revision,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "refit_requested": refit, "anonymous_access": True,
        "gpu_calls": 0, "original_test_used": False,
    }
    _write_once(output / "intent.json", record)
    try:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Use Python 3.12 and requirements/cpu.txt")
        if importlib.metadata.version("huggingface_hub") != HUB_VERSION:
            raise RuntimeError("Use huggingface_hub==1.19.0 for this delivery example")
        from thinking_completion.cli import check_cpu_runtime, reproduce, verify_saved
        from thinking_completion.data import sha256_file, verify_bundle
        from huggingface_hub import HfApi, snapshot_download
        check_cpu_runtime()
        info = HfApi(endpoint=ENDPOINT, token=False).dataset_info(
            REPO_ID, revision=revision, token=False, timeout=30,
        )
        if not isinstance(info.sha, str) or re.fullmatch(r"[0-9a-f]{40}", info.sha) is None:
            raise RuntimeError("Dataset did not resolve to an immutable full commit SHA")
        if revision != DEFAULT_REVISION and info.sha != revision:
            raise RuntimeError("Resolved dataset commit differs from requested full SHA")
        record["resolved_revision"] = info.sha
        # HF local_dir creates .cache metadata. Keep that outside the exact
        # five-member verified bundle, preserving it rather than deleting it.
        staging = output / "_download"
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", revision=info.sha,
            allow_patterns=list(PUBLIC_FILES), local_dir=staging,
            endpoint=ENDPOINT, token=False, max_workers=2, etag_timeout=30,
        )
        bundle = output / "bundle"
        bundle.mkdir(exist_ok=False)
        for name in PUBLIC_FILES:
            source = staging / name
            if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(staging.resolve()):
                raise RuntimeError("Download did not produce a regular allowlisted public file")
            with source.open("rb") as src, (bundle / name).open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            if name == "manifest.json" and sha256_file(bundle / name) != EXPECTED_MANIFEST_SHA256:
                raise RuntimeError("Independent manifest SHA256 mismatch; preserve failed download")
        integrity = verify_bundle(bundle)  # Every full file SHA/size, no pickle, exact schema/pairing.
        verified = verify_saved(bundle)  # Restores frozen models; no fit or new scientific outcome.
        if verified["statistic_comparison"]["checks"] != 39:
            raise RuntimeError("Expected all 39 historical numerical statistics")
        result = {
            **record, "status": "PASS", "python": sys.version.split()[0],
            "huggingface_hub": HUB_VERSION, "integrity": integrity,
            "saved_verification": verified,
        }
        if refit:
            result["refit"] = reproduce(bundle, output / "refit")
        _write_once(output / "result.json", result)
        return result
    except Exception as error:
        _write_once(output / "failure.json", {
            **record, "status": "FAILED_PRESERVED", "error_type": type(error).__name__,
            "automatic_retry": False,
        })
        raise  # Do not hide a failing runtime, checksum, download or reproduction.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; never overwritten")
    parser.add_argument("--revision", type=_revision, default=DEFAULT_REVISION)
    parser.add_argument("--refit", action="store_true", help="Explicitly refit the unchanged TRAIN-only CPU procedure")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output, revision=args.revision, refit=args.refit),
                     sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
