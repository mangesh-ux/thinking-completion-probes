"""Small explicit interfaces for verification, reproduction and future regeneration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_saved(bundle: Path) -> dict:
    check_cpu_runtime()
    import numpy as np
    from .data import verify_bundle, load_dataset
    from .metrics import evaluate_predictions, MODELS
    from .modeling import predict_frozen
    from .verification import compare_results
    integrity = verify_bundle(bundle)
    validation = load_dataset(bundle / "validation.npz")
    models = json.loads((bundle / "fitted-models.json").read_text(encoding="utf-8"))
    with np.load(bundle / "heldout-predictions.npz", allow_pickle=False) as archive:
        predictions = {name: archive[name] for name in MODELS}
    errors = {name: float(np.max(np.abs(predict_frozen(validation, models["models"][name]) - predictions[name]))) for name in MODELS}
    if max(errors.values()) > 1e-10:
        raise ValueError("Restored model and released predictions disagree")
    statistics = evaluate_predictions(validation, predictions, models["train_problem_prevalence"])
    comparison = compare_results(statistics)
    if comparison["status"] != "PASS":
        raise ValueError("Released statistics disagree with historical reference")
    return {"status": "PASS", "scope": "Bundle integrity, frozen-model prediction restoration and saved-prediction statistics; no refitting", "manifest_sha256": integrity["manifest_sha256"], "support": integrity["support"], "statistic_comparison": comparison, "prediction_max_absolute_errors": errors, "gpu_calls": 0, "test_loaded": False}


def reproduce(bundle: Path, output: Path) -> dict:
    check_cpu_runtime()
    from .data import verify_bundle, load_dataset
    from .modeling import select, evaluate
    from .verification import compare_results
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Reproduction outputs cannot modify the release bundle")
    if output.exists():
        raise FileExistsError("Reproduction output must be fresh")
    verified = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_once(output / "reproduction-intent.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": verified["manifest_sha256"], "gpu_calls": 0, "test_loaded": False})
    try:
        # Only TRAIN enters the selector. Validation is loaded afterwards.
        train = load_dataset(bundle / "train.npz")
        print("Selecting 128 TRAIN-only configurations (640 CV fits)...", file=sys.stderr, flush=True)
        selection = select(train, output / "selection")
        validation = load_dataset(bundle / "validation.npz")
        result = evaluate(train, validation, output / "selection", output / "evaluation")
        comparison = compare_results(result)
        record = {"status": comparison["status"], "scope": "Full CPU probe refitting, TRAIN selection and frozen held-out validation reproduction", "manifest_sha256": verified["manifest_sha256"], "elapsed_seconds": time.monotonic() - started, "comparison": comparison, "selected": {name: {"C": value["C"], "layer": value["layer"]} for name, value in selection["selected"].items()}, "cpu_cv_fits": 640, "final_train_fits": 5, "gpu_calls": 0, "test_loaded": False}
        write_once(output / "reproduction-result.json", record)
        if comparison["status"] != "PASS":
            raise ValueError("CPU result differs from historical numerical reference; preserve failure")
        return record
    except Exception as error:
        write_once(output / "reproduction-failure.json", {"error_type": type(error).__name__, "automatic_retry": False, "elapsed_seconds": time.monotonic() - started})
        raise


def check_cpu_runtime() -> None:
    expected = {"numpy": "2.2.6", "scipy": "1.14.0", "scikit-learn": "1.5.1", "joblib": "1.4.2", "threadpoolctl": "3.5.0"}
    if sys.version_info[:2] != (3, 12) or any(importlib.metadata.version(key) != version for key, version in expected.items()):
        raise ValueError("CPU runtime differs from the pinned Python 3.12 numerical environment")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory thinking-completion probes; never an automatic cloud/GPU launcher")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Check the report, optionally restore saved models and recompute metrics")
    check.add_argument("--bundle", type=Path)
    delivery = commands.add_parser("download", help="Download the fixed anonymized numeric release")
    delivery.add_argument("--destination", type=Path, required=True)
    fit = commands.add_parser("reproduce", help="Refit the historical CPU procedure using released numerical inputs")
    fit.add_argument("--bundle", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    for command in ("prepare", "plan", "generate"):
        sub = commands.add_parser(command, help="Future opt-in private regeneration: " + command)
        sub.add_argument("--config", type=Path, required=True)
        if command == "prepare":
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--manifest", type=Path, required=True)
            sub.add_argument("--max-rows", type=int)
            if command == "generate":
                sub.add_argument("--output", type=Path, required=True)
                sub.add_argument("--confirm-gpu", action="store_true")
    extract = commands.add_parser("features", help="CPU extraction from private, verified regenerated receipts")
    extract.add_argument("--generation", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            from .verification import verify_reference
            result = verify_saved(args.bundle) if args.bundle else verify_reference()
        elif args.command == "download":
            from .download import download_bundle
            result = download_bundle(args.destination)
        elif args.command == "reproduce":
            result = reproduce(args.bundle, args.output)
        elif args.command == "features":
            from .generation import extract_features
            report = extract_features(args.generation, args.output)
            result = {key: report[key] for key in ("record_type", "support", "exclusions", "gpu_operations", "test_rows")}
        else:
            from . import generation
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if args.command == "prepare":
                manifest = generation.prepare_manifest(config, args.output)
                result = {key: manifest[key] for key in ("record_type", "split_counts", "membership_identity_sha256", "test_generation_permitted")}
            elif args.command == "plan":
                manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
                plan = generation.generation_plan(config, manifest, args.max_rows)
                result = {key: value for key, value in plan.items() if key != "request_keys"}
            else:
                report = generation.generate(config, args.manifest, args.output, args.confirm_gpu, args.max_rows)
                result = {key: value for key, value in report.items() if key not in ("inventory", "payloads", "request_keys")}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("status") == "FAILED_PRESERVED" else 0
    except Exception as error:
        # Avoid leaking raw third-party model text, benchmark rows or machine paths.
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "message": "Check dependency versions, input checksums, command arguments and preserved failure receipts. Outputs are never silently overwritten."}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
