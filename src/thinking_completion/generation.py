"""Opt-in future regeneration of natural-thinking-completion measurements.

This route never provisions infrastructure. Raw material is private and must
remain beneath an ignored artifacts/ directory. It creates a NEW stochastic
experiment, not a byte-identical replay or extension of the historical run.
"""
from __future__ import annotations

import copy
import io
import json
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable

from .generation_helpers.pure import (
    NUMERIC_FEATURE_NAMES, canonical, completion_target, fold_assignments,
    lexical_counts, numeric_features, object_sha, reasoning_boundary,
    rollout_seed, select_cohort, sha, text_sha,
)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "protocol": "natural-thinking-completion-regeneration-v1",
    "dataset": {"id": "zwhe99/DeepMath-103K",
                "revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
                "split": "train", "selection_seed": 20260929},
    "model": {"id": "Qwen/Qwen3-1.7B",
              "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
              "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
              "dtype": "bfloat16", "device": "cuda:0",
              "num_hidden_layers": 28, "hidden_size": 2048},
    "prompt": {"template": "Solve the following mathematics problem. Work through the reasoning carefully.\n"
               "After the thinking section, give only a concise final answer in exactly one\n"
               "\\boxed{...} expression, with no explanation after the box.\n\nProblem:\n{problem}",
               "enable_thinking": True, "add_generation_prompt": True},
    "generation": {"max_new_tokens": 4096, "do_sample": True, "temperature": 0.6,
                   "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                   "seed_root": 20260929, "rollout_indices": [0, 1, 2, 3, 4, 5]},
    "readout": {"positions": [128, 256, 512], "max_new_tokens": 64,
                "do_sample": False, "cue": "</think>\n\n"},
    "activation": {"position": 512, "layers": 28, "hidden_size": 2048,
                   "dtype": "bfloat16", "input": "prompt-plus-exact-prefix-no-cue",
                   "representation": "post-block-final-position-before-final-norm"},
    "splits": {"seed": 20260930, "train_per_difficulty": 48,
               "validation_per_difficulty": 16, "test_per_difficulty": 16,
               "generate": ["train", "validation"]},
    "runtime": {"expected_versions": {"python": "3.12.3", "torch": "2.11.0+cu128",
                "transformers": "4.52.4", "datasets": "4.3.0", "accelerate": "1.12.0",
                "huggingface_hub": "0.36.2", "safetensors": "0.4.3"}},
    "local_files_only": False,
}


# Independently verified semantic projections of the original fixed cohort.
# These disclose no row identities or benchmark text and are NOT hashes of
# this separately versioned private-manifest representation.
ORIGINAL_MEMBERSHIP_SHA256 = "853f858abf5662b3ceac967cbd6c0cdd7af160ffdd216f36aec70702705f5952"
ORIGINAL_SPLITS_SHA256 = "7ef8bbbe1f9e04fff61005fd284009c0764eec44d98574dbe8bfefbc0dfb17e9"
ORIGINAL_FOLDS_SHA256 = "e72c1c1e826da1234f9f5bde09c4918044d1ec49deca156eb863afc431b4fc21"


def _verify_original_membership(records: list[dict], folds: dict) -> None:
    fields = ("problem_id", "source_index", "difficulty", "topic", "question_sha256",
              "reference_answer_sha256", "metadata_sha256", "duplicate_cluster_sha256")
    membership = sorted([{field: r[field] for field in fields} for r in records],
                        key=lambda r: r["problem_id"])
    assignments = sorted([{field: r[field] for field in ("problem_id", "source_index",
                         "difficulty", "duplicate_cluster_sha256", "split")} for r in records],
                         key=lambda r: r["problem_id"])
    if object_sha(membership) != ORIGINAL_MEMBERSHIP_SHA256 or \
            object_sha(assignments) != ORIGINAL_SPLITS_SHA256 or \
            object_sha(folds) != ORIGINAL_FOLDS_SHA256:
        raise ValueError("Manifest does not match independently trusted original cohort/splits/folds")


def _config(config: dict) -> dict:
    """Reject scientific drift; non-scientific explanatory metadata is retained."""
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
    result = copy.deepcopy(DEFAULT_CONFIG)
    def same(value, expected):
        if isinstance(expected, list):
            return isinstance(value, list) and len(value) == len(expected) and all(
                same(a, b) for a, b in zip(value, expected))
        if isinstance(expected, float):
            return type(value) in (int, float) and value == expected
        return type(value) is type(expected) and value == expected
    def merge(target, supplied, expected, prefix=""):
        for key, value in supplied.items():
            if key in expected and isinstance(expected[key], dict):
                if not isinstance(value, dict):
                    raise ValueError("Frozen config object differs: " + prefix + key)
                merge(target[key], value, expected[key], prefix + key + ".")
            elif key in expected and key != "local_files_only" and not same(value, expected[key]):
                raise ValueError("Frozen generation definition differs: " + prefix + key)
            elif key not in expected and not (prefix == "" and key == "note"):
                raise ValueError("Unknown generation config field: " + prefix + key)
            else:
                target[key] = copy.deepcopy(value)
    merge(result, config, DEFAULT_CONFIG)
    if type(result["local_files_only"]) is not bool:
        raise ValueError("local_files_only must be Boolean")
    return result


def _publication_root() -> Path:
    """Require execution from this publication checkout, not the old workspace."""
    root = Path.cwd().resolve()
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        ignores = [(line.strip()) for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()]
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Run from the thinking-completion-probes publication checkout root") from error
    if project.get("project", {}).get("name") != "thinking-completion-probes":
        raise ValueError("Raw output requires the thinking-completion-probes checkout root")
    if not any(line in ("artifacts/", "/artifacts/") for line in ignores) or any(
            line.startswith("!") for line in ignores):
        raise ValueError("Publication artifacts/ must be ignored without ignore negations")
    return root


def _private(path: Path) -> Path:
    root = _publication_root()
    logical_artifacts = root / "artifacts"
    artifacts = logical_artifacts.resolve()
    path = Path(path).resolve()
    if artifacts != logical_artifacts or path == artifacts or not path.is_relative_to(artifacts):
        raise ValueError("Raw outputs must stay inside this publication checkout's ignored artifacts/")
    return path


def _write(path: Path, content: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    return {"bytes": len(content), "sha256": sha(content)}


def _json(path: Path, value: object) -> dict:
    return _write(path, canonical(value))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_from_rows(config: dict, rows, *, fingerprint: str | None = None) -> dict:
    """Pure fixture/cache adapter. It neither imports datasets nor downloads anything."""
    config = _config(config)
    records = select_cohort(rows, selection_seed=config["dataset"]["selection_seed"],
                           split_seed=config["splits"]["seed"])
    return {"schema_version": 1, "record_type": "PRIVATE_REGENERATION_MANIFEST_V1",
            "config_sha256": object_sha(config), "dataset": config["dataset"],
            "dataset_fingerprint": fingerprint, "records": records,
            "split_counts": dict(Counter(r["split"] for r in records)),
            "membership_identity_sha256": object_sha([
                {k: v for k, v in r.items() if k != "question"} for r in records]),
            "fold_assignment": fold_assignments(records),
            "test_generation_permitted": False, "private_not_for_publication": True}


def prepare_manifest(config: dict, output: Path) -> dict:
    """Future opt-in dataset preparation; no model/GPU is loaded.

    Only this function may resolve/download the fixed dataset. For offline
    preparation set local_files_only=true; a missing cache fails without a
    network fallback. All 160 identities are fixed before answer validation.
    """
    config, output = _config(config), _private(output)
    if output.exists():
        raise FileExistsError("Manifest output is write-once")
    from datasets import DownloadConfig, load_dataset
    from huggingface_hub import HfApi
    if not config["local_files_only"]:
        resolved = HfApi().dataset_info(config["dataset"]["id"],
                                        revision=config["dataset"]["revision"]).sha
        if resolved != config["dataset"]["revision"]:
            raise RuntimeError("Dataset revision resolution differs")
    dataset = load_dataset(config["dataset"]["id"], revision=config["dataset"]["revision"],
                           split="train", download_config=DownloadConfig(
                               local_files_only=config["local_files_only"]))
    manifest = manifest_from_rows(config, dataset, fingerprint=getattr(dataset, "_fingerprint", None))
    _validate_manifest(manifest, config)
    _json(output, manifest)
    return manifest


def _validate_manifest(manifest: dict, config: dict) -> None:
    if manifest.get("record_type") != "PRIVATE_REGENERATION_MANIFEST_V1" or \
            manifest.get("config_sha256") != object_sha(config) or manifest.get("dataset") != config["dataset"]:
        raise ValueError("Manifest source/config identity differs")
    records = manifest["records"]
    if len(records) != 160 or len({r["problem_id"] for r in records}) != 160 or \
            len({r["duplicate_cluster_sha256"] for r in records}) != 160:
        raise ValueError("Frozen 160 unique identities are not intact")
    if records != sorted(records, key=lambda r: r["problem_id"]):
        raise ValueError("Manifest canonical order differs")
    if Counter(r["split"] for r in records) != {"train": 96, "validation": 32, "test": 32}:
        raise ValueError("Grouped split counts differ")
    for difficulty in (7.0, 8.0):
        stratum = [r for r in records if r["difficulty"] == difficulty]
        ranked = sorted(stratum, key=lambda r: (
            text_sha(f"20260930\0{difficulty:.1f}\0{r['duplicate_cluster_sha256']}\0{r['problem_id']}"),
            r["source_index"]))
        if len(ranked) != 80:
            raise ValueError("Frozen difficulty quota differs")
        for index, record in enumerate(ranked):
            expected = "train" if index < 48 else "validation" if index < 64 else "test"
            if record["split"] != expected:
                raise ValueError("Frozen grouped split assignment differs")
    for record in records:
        if type(record["source_index"]) is not int or record["source_index"] < 0 or \
                record["problem_id"] != f"deepmath:{record['source_index']}:{record['question_sha256'][:16]}":
            raise ValueError("Stable original source identity differs")
        if record["metadata_sha256"] != object_sha({"difficulty": record["difficulty"],
                                                    "topic": record["topic"]}):
            raise ValueError("Source metadata identity differs")
        if record["split"] != "test" and text_sha(record["question"]) != record["question_sha256"]:
            raise ValueError("Problem text identity differs")
        if record["split"] != "test" and text_sha(" ".join(record["question"].split())) != record["duplicate_cluster_sha256"]:
            raise ValueError("Normalized source cluster differs")
    if manifest["fold_assignment"] != fold_assignments(records):
        raise ValueError("Original-ID TRAIN folds differ")
    identity = object_sha([{k: v for k, v in r.items() if k != "question"} for r in records])
    if identity != manifest["membership_identity_sha256"]:
        raise ValueError("Manifest membership identity differs")
    _verify_original_membership(records, manifest["fold_assignment"])


def generation_plan(config: dict, manifest: dict, max_rows: int | None = None) -> dict:
    """Inspect bounded planned operation count without importing CUDA or a model."""
    config = _config(config)
    _validate_manifest(manifest, config)
    requests = [(r, i) for r in manifest["records"] if r["split"] in ("train", "validation")
                for i in config["generation"]["rollout_indices"]]
    if max_rows is not None:
        if type(max_rows) is not int or not 1 <= max_rows <= len(requests):
            raise ValueError("max_rows must bound rollout count between 1 and 768")
        requests = requests[:max_rows]
    return {"record_type": "REGENERATION_OPERATION_PLAN_V1", "created_at_utc": _utc(),
            "config_sha256": object_sha(config), "manifest_sha256": object_sha(manifest),
            "rollouts": len(requests), "base_requests": len(requests),
            "readout_requests_maximum": 3 * len(requests),
            "activation_requests_maximum": len(requests),
            "planned_model_operations_maximum": 5 * len(requests),
            "splits": dict(Counter(r["split"] for r, _ in requests)),
            "maximum_generated_tokens": (4096 + 3 * 64) * len(requests),
            "partial_diagnostic_not_full_study": len(requests) != 768,
            "test_requests": 0, "no_retry_no_replacement": True,
            "cloud_provisioning": False,
            "request_keys": [{"problem_id": r["problem_id"], "rollout_index": i,
                               "seed": rollout_seed(r["problem_id"], i)} for r, i in requests]}


def _load_runtime(config: dict):
    from .generation_helpers.runtime import QwenRuntime
    return QwenRuntime(config)


def _inventory(output: Path) -> dict:
    payloads = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            if path.is_symlink():
                raise ValueError("Private payload symlinks are forbidden")
            content = path.read_bytes()
            payloads[path.relative_to(output).as_posix()] = {"bytes": len(content), "sha256": sha(content)}
    return {"record_type": "REGENERATION_PAYLOAD_INVENTORY_V1", "payloads": payloads}


def generate(config: dict, manifest_path: Path, output: Path, confirm_gpu: bool = False,
             max_rows: int | None = None) -> dict:
    """Execute once only AFTER explicit GPU consent; failed/uncertain work is not replayed.

    Plan is written before importing the GPU backend. Existing output always
    fails closed, including a previous intent without a receipt. All failures
    are retained; there is no launch, repair, retry or sample substitution.
    """
    if confirm_gpu is not True:
        raise PermissionError("GPU regeneration requires explicit confirm_gpu=True")
    config, output = _config(config), _private(output)
    if output.exists():
        raise FileExistsError("Generation directory is write-once; never replay uncertain intents")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    plan = generation_plan(config, manifest, max_rows)
    output.mkdir(parents=True)
    _json(output / "plan.json", plan)  # Durable BEFORE CUDA/model import.
    _json(output / "config.json", config)
    _write(output / "manifest.json", manifest_bytes)
    _json(output / "source.json", {"generation_source_sha256": sha(Path(__file__).read_bytes()),
                                  "pure_source_sha256": sha((Path(__file__).parent / "generation_helpers/pure.py").read_bytes()),
                                  "runtime_source_sha256": sha((Path(__file__).parent / "generation_helpers/runtime.py").read_bytes())})
    counters, failed, runtime = Counter(), False, None
    by_id = {r["problem_id"]: r for r in manifest["records"]}
    def operation(name: str, kind: str, ids: list[int], call: Callable, extra: dict) -> dict:
        intent = {"record_type": "MODEL_OPERATION_INTENT_V1", "kind": kind,
                  "input_sha256": object_sha(ids), "input_token_count": len(ids),
                  "config_sha256": object_sha(config), "created_at_utc": _utc(), **extra}
        _json(output / "intents" / (name + ".json"), intent)
        try:
            value = call()
            if kind == "activation":
                content, metadata = value
                relative = "matrices/" + name + ".pt"
                tensor = _write(output / relative, content)
                value = {"matrix": {"path": relative, **tensor, **metadata}}
            result = {"status": "COMPLETED", **value, **extra}
            counters[kind + "_completed"] += 1
        except Exception as error:
            result = {"status": "FAILED", "error_type": type(error).__name__, **extra}
            counters[kind + "_failed"] += 1
            raise_error[0] = True
        result.update(kind=kind, intent_sha256=object_sha(intent), input_sha256=object_sha(ids),
                      completed_at_utc=_utc())
        _json(output / "receipts" / (name + ".json"), result)
        return result
    raise_error = [False]
    try:
        runtime = _load_runtime(config)
        _json(output / "runtime.json", runtime.provenance)
        for row_index, key in enumerate(plan["request_keys"]):
            record = by_id[key["problem_id"]]
            prefix = f"row-{row_index:04d}"
            identity = {"row_index": row_index, "problem_id": record["problem_id"],
                        "split": record["split"], "rollout_index": key["rollout_index"]}
            prompt = runtime.prompt(record["question"])
            _json(output / "prompts" / (prefix + ".json"), {**prompt, **identity,
                   "token_ids_sha256": object_sha(prompt["token_ids"])})
            base = operation(prefix + "-base", "base", prompt["token_ids"],
                             lambda: runtime.reasoning(prompt["token_ids"], key["seed"]),
                             {**identity, "seed": key["seed"]})
            if raise_error[0]:
                failed = True
                break
            boundary = reasoning_boundary(base["tokens"], runtime.marker, runtime.eos, base["termination"])
            _json(output / "boundaries" / (prefix + ".json"), boundary)
            for position in (128, 256, 512):
                name = prefix + f"-readout-{position}"
                if boundary["terminal_position"] is None or boundary["terminal_position"] < position:
                    _json(output / "receipts" / (name + ".json"), {
                        "status": "UNAVAILABLE", "kind": "readout", "position": position,
                        "reason": boundary["status"], **identity})
                    counters["readout_unavailable"] += 1
                    continue
                saved = base["tokens"][:position]
                ids = prompt["token_ids"] + saved + runtime.cue
                operation(name, "readout", ids, lambda ids=ids: runtime.readout(ids),
                          {**identity, "position": position, "prefix_sha256": object_sha(saved),
                           "cue_sha256": object_sha(runtime.cue)})
                if raise_error[0]:
                    break
            if raise_error[0]:
                failed = True
                break
            if boundary["terminal_position"] is None or boundary["terminal_position"] < 512:
                _json(output / "receipts" / (prefix + "-activation.json"), {
                    "status": "UNAVAILABLE", "kind": "activation", "position": 512,
                    "reason": boundary["status"], **identity})
                counters["activation_unavailable"] += 1
            else:
                saved = base["tokens"][:512]
                ids = prompt["token_ids"] + saved  # Deliberately NO readout cue.
                operation(prefix + "-activation", "activation", ids,
                          lambda: runtime.activation(ids), {**identity, "position": 512,
                          "prefix_sha256": object_sha(saved), "input_has_readout_cue": False})
                if raise_error[0]:
                    failed = True
                    break
    except Exception as error:
        failed = True
        _json(output / "run-failure.json", {"status": "FAILED", "error_type": type(error).__name__,
                                           "retry_permitted": False, "created_at_utc": _utc()})
    finally:
        del runtime
    inventory = _inventory(output)
    inventory_meta = _json(output / "inventory.json", inventory)
    summary = {"record_type": "REGENERATION_AGGREGATE_SUMMARY_V1",
               "status": "FAILED_PRESERVED" if failed else "COLLECTION_COMPLETE",
               "planned_rollouts": plan["rollouts"], "operations": dict(counters),
               "partial_diagnostic_not_full_study": plan["partial_diagnostic_not_full_study"],
               "inventory_sha256": inventory_meta["sha256"], "test_requests": 0,
               "not_historical_bit_identical": True, "new_scientific_conclusions": False}
    _json(output / "summary.json", summary)
    return summary


def _verified_payloads(directory: Path) -> tuple[dict, dict]:
    summary = json.loads((directory / "summary.json").read_bytes())
    inventory_bytes = (directory / "inventory.json").read_bytes()
    if sha(inventory_bytes) != summary["inventory_sha256"]:
        raise ValueError("Private inventory hash differs")
    inventory = json.loads(inventory_bytes)
    payloads = {}
    for relative, expected in inventory["payloads"].items():
        logical = PurePosixPath(relative)
        if logical.is_absolute() or ".." in logical.parts or "\\" in relative or ":" in relative:
            raise ValueError("Unsafe payload path")
        path = directory.joinpath(*logical.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Payload escapes private extraction root")
        content = path.read_bytes()
        if len(content) != expected["bytes"] or sha(content) != expected["sha256"]:
            raise ValueError("Private payload hash/size differs")
        payloads[relative] = content
    return summary, payloads


def extract_features(generation_directory: Path, output: Path) -> dict:
    """CPU-only authenticated numeric derivative; no fit, evaluator or TEST data.

    Features are constructed before the outcome join. Serialized BF16 is
    deserialized with weights_only=True and converted losslessly to float32.
    Fresh numerical derivatives carry private original IDs; they MUST NOT use
    the historical public-release manifest or be silently added to that study.
    """
    import numpy as np
    import torch
    directory, output = _private(generation_directory), _private(output)
    if output.exists():
        raise FileExistsError("Feature output is write-once")
    summary, payloads = _verified_payloads(directory)
    config = _config(json.loads(payloads["config.json"]))
    manifest = json.loads(payloads["manifest.json"])
    _validate_manifest(manifest, config)
    plan, runtime = json.loads(payloads["plan.json"]), json.loads(payloads.get("runtime.json", b"{}"))
    expected_plan = generation_plan(config, manifest, plan["rollouts"])
    if any(plan[field] != expected_plan[field] for field in expected_plan if field != "created_at_utc"):
        raise ValueError("Recorded operation plan differs from fixed manifest/config")
    records = {r["problem_id"]: r for r in manifest["records"]}
    features, exclusions = [], Counter()
    readout_nonfinite, feature_nonfinite = Counter(), Counter()
    for relative, content in payloads.items():
        if relative.startswith("receipts/") and relative.endswith(".json"):
            receipt = json.loads(content)
            if receipt.get("kind") == "readout" and receipt.get("status") == "COMPLETED":
                fields = receipt.get("nonfinite_scalar_fields_encoded_missing", [])
                if not isinstance(fields, list) or len(fields) != len(set(fields)) or any(
                        field not in NUMERIC_FEATURE_NAMES[4:9] for field in fields):
                    raise ValueError("Nonfinite readout conversion metadata differs")
                count = receipt.get("nonfinite_scalar_count_encoded_missing", len(fields))
                if type(count) is not int or count != len(fields):
                    raise ValueError("Nonfinite readout conversion count differs")
                readout_nonfinite.update(fields)
    def checked_receipt(stem: str, suffix: str, identity: dict) -> dict:
        receipt = json.loads(payloads[f"receipts/{stem}-{suffix}.json"])
        if any(receipt.get(field) != value for field, value in identity.items()):
            raise ValueError("Receipt identity differs from planned request")
        if receipt["status"] in ("COMPLETED", "FAILED"):
            intent = json.loads(payloads[f"intents/{stem}-{suffix}.json"])
            if receipt["intent_sha256"] != object_sha(intent) or \
                    receipt["input_sha256"] != intent["input_sha256"] or \
                    intent["config_sha256"] != object_sha(config) or \
                    any(intent.get(field) != value for field, value in identity.items()):
                raise ValueError("Receipt does not bind the write-once intent")
        return receipt
    # This phase has no label/termination access. It projects only saved
    # prefix, prior readouts and matrix links, never future reasoning length.
    for row_index, key in enumerate(plan["request_keys"]):
        stem = f"row-{row_index:04d}"
        base_path = f"receipts/{stem}-base.json"
        activation_path = f"receipts/{stem}-activation.json"
        if base_path not in payloads or activation_path not in payloads:
            exclusions["INCOMPLETE_OR_FAILED_OPERATIONS"] += 1
            continue
        record = records[key["problem_id"]]
        identity = {"row_index": row_index, "problem_id": record["problem_id"],
                    "split": record["split"], "rollout_index": key["rollout_index"]}
        activation = checked_receipt(stem, "activation", {**identity, "position": 512})
        if activation["status"] != "COMPLETED":
            exclusions["ACTIVATION_UNAVAILABLE"] += 1
            continue
        base = checked_receipt(stem, "base", {**identity, "seed": key["seed"]})
        prompt = json.loads(payloads[f"prompts/{stem}.json"])
        if any(prompt.get(field) != value for field, value in identity.items()) or \
                object_sha(prompt["token_ids"]) != prompt["token_ids_sha256"] or \
                base["input_sha256"] != prompt["token_ids_sha256"]:
            raise ValueError("Prompt does not bind the planned base generation")
        prefix = base["tokens"][:512]
        if len(prefix) != 512 or activation["prefix_sha256"] != object_sha(prefix):
            raise ValueError("Activation prefix identity differs")
        exact_input = prompt["token_ids"] + prefix
        if activation["input_sha256"] != object_sha(exact_input) or activation["input_has_readout_cue"]:
            raise ValueError("Activation input is not exact prompt+prefix without cue")
        if record["split"] not in ("train", "validation"):
            raise ValueError("TEST identity reached feature extraction")
        readouts = []
        for position in (128, 256, 512):
            readout = checked_receipt(stem, f"readout-{position}", {**identity, "position": position})
            if readout["status"] == "COMPLETED":
                expected_input = prompt["token_ids"] + prefix[:position] + runtime["cue_token_ids"]
                if readout["input_sha256"] != object_sha(expected_input) or \
                        readout["prefix_sha256"] != object_sha(prefix[:position]) or \
                        readout["cue_sha256"] != object_sha(runtime["cue_token_ids"]):
                    raise ValueError("Checkpoint readout contains changed/future prefix")
            readouts.append(readout)
        numeric = numeric_features(record, prompt["problem_token_count"], readouts,
                                   nonfinite_counts=feature_nonfinite)
        matrix_metadata = activation["matrix"]
        content = payloads[matrix_metadata["path"]]
        if sha(content) != matrix_metadata["sha256"] or len(content) != matrix_metadata["bytes"]:
            raise ValueError("Matrix serialized identity differs")
        matrix = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        if not isinstance(matrix, torch.Tensor) or matrix.dtype != torch.bfloat16 or \
                tuple(matrix.shape) != (28, 2048) or not torch.isfinite(matrix).all():
            raise ValueError("Matrix BF16 geometry/finiteness differs")
        header = canonical({"shape": [28, 2048], "dtype": "bfloat16",
                            "representation": "post-block-last-reasoning-token-v1"})
        content_sha = sha(header + b"\0" + matrix.contiguous().view(torch.uint8).numpy().tobytes())
        if content_sha != matrix_metadata["matrix_sha256"]:
            raise ValueError("Matrix content identity differs")
        features.append({"row_index": row_index, "record": record, "numeric": numeric,
                         "lexical": lexical_counts(prefix), "matrix": matrix.float().numpy(),
                         "row_id": record["problem_id"] + f"|rollout-{key['rollout_index']}|checkpoint-512"})
    output.mkdir(parents=True)
    before = [{k: v for k, v in row.items() if k not in ("matrix", "record")}
              for row in features]
    before_meta = _json(output / "features-before-labels.json", {
        "features": before, "future_labels_joined": False,
        "terminal_correctness_or_evaluator_features": False})
    accepted = []
    for row in features:
        stem = f"row-{row['row_index']:04d}"
        base = json.loads(payloads[f"receipts/{stem}-base.json"])
        boundary = reasoning_boundary(base["tokens"], runtime["close_marker_token_ids"],
                                       runtime["eos_token_ids"], base["termination"])
        if boundary != json.loads(payloads[f"boundaries/{stem}.json"]):
            raise ValueError("Recorded boundary differs from original emitted tokens")
        label, status = completion_target(boundary)
        if label is None:
            exclusions[status] += 1
            continue
        accepted.append({**row, "label": label})
    outputs, support = {}, {}
    for split in ("train", "validation"):
        rows = [r for r in accepted if r["record"]["split"] == split]
        if not rows:
            support[split] = {"rows": 0, "problems": 0, "positive_rows": 0}
            continue
        buffer = io.BytesIO()
        np.savez_compressed(buffer, numeric=np.asarray([
            [np.nan if value is None else value for value in r["numeric"]] for r in rows], dtype=np.float64),
            topics=np.asarray([r["record"]["topic"] for r in rows], dtype=str),
            groups=np.asarray([r["record"]["problem_id"] for r in rows], dtype=str),
            lexical=np.asarray([r["lexical"] for r in rows], dtype=np.float64),
            activations=np.stack([r["matrix"] for r in rows]).astype(np.float32),
            labels=np.asarray([r["label"] for r in rows], dtype=np.int8),
            row_ids=np.asarray([r["row_id"] for r in rows], dtype=str),
            fold_ids=np.asarray([manifest["fold_assignment"][r["record"]["problem_id"]]
                                 if split == "train" else -1 for r in rows], dtype=np.int8),
            difficulty=np.asarray([r["record"]["difficulty"] for r in rows], dtype=np.float64),
            numeric_feature_names=np.asarray(NUMERIC_FEATURE_NAMES, dtype=str))
        outputs[split + ".npz"] = _write(output / (split + ".npz"), buffer.getvalue())
        support[split] = {"rows": len(rows), "problems": len({r["record"]["problem_id"] for r in rows}),
                          "positive_rows": sum(r["label"] for r in rows)}
    report = {"record_type": "PRIVATE_REGENERATION_DERIVATIVE_V1",
              "generation_inventory_sha256": summary["inventory_sha256"],
              "generation_status": summary["status"], "outputs": outputs, "support": support,
              "exclusions": dict(exclusions), "features_before_labels_sha256": before_meta["sha256"],
              "nonfinite_readout_scalars_encoded_missing": dict(readout_nonfinite),
              "nonfinite_primary_feature_values_encoded_missing": dict(feature_nonfinite),
              "consumed_payloads": {path: {"bytes": len(content), "sha256": sha(content)}
                                    for path, content in payloads.items()},
              "gpu_operations": 0, "test_rows": 0, "not_original_historical_release": True,
              "partial_diagnostic_not_full_study": plan["partial_diagnostic_not_full_study"]}
    _json(output / "derivative-report.json", report)
    return report
