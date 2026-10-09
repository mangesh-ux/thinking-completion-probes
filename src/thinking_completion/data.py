"""Verified, numeric-only publication bundle for the completed frozen study.

This module never fits a model, calls a GPU, reads raw benchmark records, or
writes into the historical source repository. Bundle checksums establish
integrity relative to the manifest, not authenticity without an independently
trusted manifest hash. Anonymization is not a claim of non-invertibility.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile

import numpy as np

ARTIFACT_ID = "completion-forecasting-public-v1"
REPOSITORY = Path(__file__).resolve().parents[2]
FEATURE_NAMES = (
    "checkpoint_token_position", "normalized_checkpoint_position",
    "problem_token_count", "difficulty", "mean_answer_token_logprob",
    "min_answer_token_logprob", "answer_token_count",
    "first_readout_token_entropy_nats", "first_readout_token_probability_margin",
    "previous_readout_agreement", "previous_readout_available",
)
DATA_KEYS = frozenset({
    "numeric", "topics", "groups", "lexical", "activations", "labels",
    "row_ids", "fold_ids", "difficulty", "numeric_feature_names",
})
MODEL_NAMES = ("A", "B", "Blex", "C", "Clex")
BUNDLE_FILES = frozenset({
    "train.npz", "validation.npz", "fitted-models.json", "heldout-predictions.npz",
})
MANIFEST_KEYS = frozenset({"schema_version", "artifact_id", "files",
                           "original_inputs_sha256", "shapes", "support", "privacy"})
PROVENANCE_KEYS = frozenset({
    "protocol", "configuration", "protocol_source_lock", "data_source", "analysis_source",
    "train_audit", "validation_audit", "train_prepared", "validation_prepared",
    "train_data", "validation_data", "selection", "fitted_models", "heldout_result",
    "private_predictions", "independent_verification", "original_eligibility_reproduction",
    "original_eligibility_auditor", "public_report",
})
SOURCE_FILES = {
    "train_data": "artifacts/completion-forecasting/v1/train/data.npz",
    "validation_data": "artifacts/completion-forecasting/v1/validation/data.npz",
    "fitted_models": "artifacts/completion-forecasting/v1/analysis/fitted-models.json",
    "private_predictions": "artifacts/completion-forecasting/v1/analysis/heldout-predictions.npz",
    "public_report": "reports/completion_forecasting_v1.json",
}
SOURCE_SHA256 = {
    "train_data": "56167195435342dc12945b6e930fd761ac977a940df90d674999b5e6b3b5b762",
    "validation_data": "9adfdc9efc39d039b88e74ecc6f95d7109f4256564146d6a779fc0a65db3f734",
    "fitted_models": "da1b0e502732a2c779cba6b263e0df1094b744aaac5b93f454960d087b5d7843",
    "private_predictions": "e09a5dd446af2043559f2126316d0e89531d3312fbb0f165562d2e9100ebda04",
    "public_report": "b45f10622d92cc66b69617c70f2a9b449f532675bb1eadbd113389a4de6a31a8",
}
PRIVACY = {
    "original_problem_identifiers_released": False,
    "original_row_identifiers_released": False,
    "raw_text_answers_or_token_sequences_released": False,
    "machine_paths_or_credentials_released": False,
    "identifier_mapping_released": False,
    "numeric_arrays_preserved_exactly": True,
    "group_order_preserved_for_seeded_bootstrap": True,
    "scope": "Anonymized numeric derivative, not a guarantee of non-invertibility",
}


class BundleError(ValueError):
    """Invalid publication input; messages never expose private values or paths."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256_file(path: Path) -> str:
    """Calculate a full streaming SHA256 without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_hash(path: Path, expected: str) -> None:
    _require(_valid_hash(expected), "Invalid expected SHA256")
    _require(sha256_file(path) == expected, "Input SHA256 mismatch")


def _json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise BundleError("Nonfinite JSON number")

    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleError("Unavailable or invalid JSON input") from error
    _require(isinstance(value, dict), "JSON root must be an object")
    return value


def _npz(path: Path, keys: frozenset[str]) -> dict[str, np.ndarray]:
    """Load only exact NPZ members; no pickle, extraction or arbitrary metadata."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            _require(len(names) == len(set(names)), "Duplicate NPZ member")
            _require(set(names) == {key + ".npy" for key in keys}, "Unexpected NPZ schema")
            _require(all(not member.is_dir() and member.file_size <= 1024 ** 3
                         for member in members), "Invalid NPZ member size")
            _require(sum(member.file_size for member in members) <= 1024 ** 3,
                     "NPZ exceeds bounded numeric release size")
            # A tiny ZIP member can otherwise claim a huge shape in its NPY
            # header and trigger allocation before the later schema checks.
            for member in members:
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    _require(version in ((1, 0), (2, 0)), "Unsupported NPY member version")
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(stream, max_header_size=4096)
                    _require(not dtype.hasobject and dtype.itemsize > 0,
                             "Object or zero-size array dtype is forbidden")
                    _require(len(shape) <= 3 and all(type(dim) is int and dim >= 0 for dim in shape),
                             "Invalid NPY shape declaration")
                    payload_bytes = math.prod(shape) * dtype.itemsize
                    _require(payload_bytes <= 1024 ** 3
                             and stream.tell() + payload_bytes == member.file_size,
                             "NPY shape and payload byte count disagree")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in keys}
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as error:
        if isinstance(error, BundleError):
            raise
        raise BundleError("Invalid numeric NPZ input") from error
    _require(all(isinstance(value, np.ndarray) and not value.dtype.hasobject
                 for value in arrays.values()), "Object arrays are forbidden")
    return arrays


def _text_array(value: np.ndarray, length: int) -> bool:
    return value.shape == (length,) and value.dtype.kind == "U" and all(
        isinstance(item, str) and bool(item) and not any(ord(c) < 32 for c in item)
        for item in value.tolist()
    )


def _group_folds(data: dict[str, np.ndarray]) -> None:
    for group in np.unique(data["groups"]):
        mask = data["groups"] == group
        _require(len(np.unique(data["fold_ids"][mask])) == 1, "Problem crosses stored folds")
        _require(len(np.unique(data["difficulty"][mask])) == 1, "Problem difficulty is inconsistent")
        _require(len(np.unique(data["topics"][mask])) == 1, "Problem topic is inconsistent")


def load_dataset(path: Path, expected_sha: str | None = None) -> dict[str, np.ndarray]:
    """Load the exact public feature schema, preserving dtype, row order and NaNs."""
    path = Path(path)
    if expected_sha is not None:
        _check_hash(path, expected_sha)
    data = _npz(path, DATA_KEYS)
    labels = data["labels"]
    _require(labels.ndim == 1 and len(labels) > 0, "Dataset has no rows")
    n = len(labels)
    expected = {
        "numeric": ((n, 11), np.dtype("float64")),
        "lexical": ((n, 512), np.dtype("float64")),
        "activations": ((n, 28, 2048), np.dtype("float32")),
        "labels": ((n,), np.dtype("int8")),
        "fold_ids": ((n,), np.dtype("int8")),
        "difficulty": ((n,), np.dtype("float64")),
    }
    for key, (shape, dtype) in expected.items():
        _require(data[key].shape == shape and data[key].dtype == dtype, "Numeric shape or dtype changed")
    for key in ("topics", "groups", "row_ids"):
        _require(_text_array(data[key], n), "Invalid string-array schema")
    names = data["numeric_feature_names"]
    _require(_text_array(names, 11) and tuple(names.tolist()) == FEATURE_NAMES,
             "Numeric feature names changed")
    _require(len(set(data["row_ids"].tolist())) == n, "Duplicate row identifiers")
    _require(np.isin(labels, [0, 1]).all(), "Targets must be binary")
    _require(np.isin(data["fold_ids"], [-1, 0, 1, 2, 3, 4]).all(), "Invalid stored fold")
    _require(np.isin(data["difficulty"], [7.0, 8.0]).all(), "Unexpected frozen difficulty")
    _require(not np.isinf(data["numeric"]).any(), "Infinite numeric features")
    _require(np.isfinite(data["numeric"][:, :4]).all(), "Structural features are missing")
    _require((data["numeric"][:, 0] == 512).all()
             and (data["numeric"][:, 1] == 512 / 4096).all()
             and np.array_equal(data["numeric"][:, 3], data["difficulty"]),
             "Structural coordinate or difficulty changed")
    count = data["numeric"][:, 2]
    _require((count >= 0).all() and (count == np.floor(count)).all(), "Invalid problem token counts")
    lexical = data["lexical"]
    _require(np.isfinite(lexical).all() and (lexical >= 0).all()
             and (lexical == np.floor(lexical)).all()
             and (lexical.sum(axis=1) == 128).all(), "Invalid hashed-prefix counts")
    _require(np.isfinite(data["activations"]).all(), "Nonfinite activations")
    _group_folds(data)
    return data


def _support(data: dict[str, np.ndarray]) -> dict[str, Any]:
    groups, labels = data["groups"], data["labels"]
    unique = np.unique(groups)
    return {
        "rows": len(labels), "problems": len(unique),
        "class_rows": {str(y): int(np.sum(labels == y)) for y in (0, 1)},
        "class_problems": {str(y): len(np.unique(groups[labels == y])) for y in (0, 1)},
        "mixed_outcome_problems": sum(len(np.unique(labels[groups == group])) == 2 for group in unique),
        "fold_problem_counts": {str(int(fold)): len(np.unique(groups[data["fold_ids"] == fold]))
                                for fold in np.unique(data["fold_ids"])},
    }


def _shapes(data: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {key: {"shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in sorted(data.items())}


def _validate_partition(data: dict[str, np.ndarray], split: str, public: bool) -> None:
    folds = set(data["fold_ids"].tolist())
    _require(folds == set(range(5)) if split == "train" else folds == {-1},
             "Partition stored-fold contract changed")
    _require(set(data["labels"].tolist()) == {0, 1}, "Partition is single class")
    _require(len(np.unique(data["groups"])) <= (96 if split == "train" else 32),
             "Partition exceeds frozen problem scope")
    if public:
        unique = sorted(set(data["groups"].tolist()))
        _require(unique == [f"{split}-g{i:03d}" for i in range(len(unique))],
                 "Public problem identifiers are not canonical anonymous IDs")
        _require(data["row_ids"].tolist() == [f"{split}-r{i:04d}" for i in range(len(data["row_ids"]))],
                 "Public row identifiers are not canonical row-order IDs")


def _models(path: Path, train: dict[str, np.ndarray]) -> dict[str, Any]:
    value = _json(path)
    _require(set(value) == {"models", "train_problem_prevalence"}, "Unexpected model JSON metadata")
    _require(isinstance(value["models"], dict) and set(value["models"]) == set(MODEL_NAMES),
             "Unexpected model set")
    prevalence = value["train_problem_prevalence"]
    _require(type(prevalence) in (int, float) and math.isfinite(prevalence) and 0 < prevalence < 1,
             "Invalid TRAIN prevalence")
    means = [train["labels"][train["groups"] == group].mean() for group in np.unique(train["groups"])]
    _require(math.isclose(prevalence, float(np.mean(means)), rel_tol=0, abs_tol=1e-15),
             "TRAIN prevalence disagrees with released labels")
    state_keys = {"model", "layer", "C", "numeric_medians", "topic_vocabulary",
                  "kept_columns", "means", "scales", "coefficients", "intercept"}
    vocabulary = sorted(set(train["topics"].tolist()))
    for name, state in value["models"].items():
        _require(isinstance(state, dict) and set(state) == state_keys and state["model"] == name,
                 "Unexpected linear-model state")
        activation = name in ("C", "Clex")
        layer = state["layer"]
        _require(type(layer) is int and 0 <= layer < 28 if activation else layer is None,
                 "Invalid selected layer")
        _require(type(state["C"]) in (int, float) and state["C"] in (0.001, 0.01, 0.1, 1.0),
                 "Unexpected regularization strength")
        _require(state["topic_vocabulary"] == vocabulary, "TRAIN topic vocabulary changed")
        n_numeric = 4 if name == "A" else 11
        full_width = 2 * n_numeric + len(vocabulary) + (512 if name in ("Blex", "Clex") else 0) + (2048 if activation else 0)
        kept = state["kept_columns"]
        _require(isinstance(kept, list) and bool(kept)
                 and all(type(i) is int and 0 <= i < full_width for i in kept)
                 and kept == sorted(set(kept)), "Invalid retained model columns")
        for field, length in (("numeric_medians", n_numeric), ("means", len(kept)),
                              ("scales", len(kept)), ("coefficients", len(kept))):
            vector = state[field]
            _require(isinstance(vector, list) and len(vector) == length
                     and all(type(item) in (int, float) and math.isfinite(item) for item in vector),
                     "Invalid numerical model vector")
        _require(all(item > 0 for item in state["scales"]), "Invalid model scales")
        _require(type(state["intercept"]) in (int, float) and math.isfinite(state["intercept"]),
                 "Invalid model intercept")
    _require(value["models"]["C"]["layer"] == value["models"]["Clex"]["layer"],
             "Secondary model uses a different layer")
    return value


def _predictions(path: Path, validation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = _npz(path, frozenset({"labels", "groups", "row_ids", *MODEL_NAMES}))
    for key in ("labels", "groups", "row_ids"):
        _require(result[key].dtype == validation[key].dtype
                 and np.array_equal(result[key], validation[key]), "Predictions are not paired to validation rows")
    for name in MODEL_NAMES:
        probabilities = result[name]
        _require(probabilities.shape == (len(validation["labels"]),)
                 and probabilities.dtype == np.dtype("float64")
                 and np.isfinite(probabilities).all()
                 and ((probabilities >= 0) & (probabilities <= 1)).all(),
                 "Invalid saved probability vector")
    return result


def _relative_file(root: Path, name: str) -> Path:
    logical = PurePosixPath(name)
    _require(not logical.is_absolute() and ".." not in logical.parts and "\\" not in name
             and ":" not in name and str(logical) == name, "Unsafe manifest path")
    path = (root / name).resolve(strict=True)
    _require(path.is_relative_to(root.resolve(strict=True)), "Manifest path escapes bundle")
    _require(path.is_file(), "Manifest path is not a file")
    return path


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Verify all sizes/hashes, numeric schemas, private-data boundaries and pairing."""
    root = Path(bundle).resolve(strict=True)
    manifest = _json(root / "manifest.json")
    _require(set(manifest) == MANIFEST_KEYS, "Unexpected manifest metadata")
    _require(manifest.get("schema_version") == 1 and manifest.get("artifact_id") == ARTIFACT_ID,
             "Unknown publication bundle version")
    files = manifest.get("files")
    _require(isinstance(files, dict), "Malformed manifest file map")
    # Reject traversal before enforcing the exact allowed member names.
    for name in files:
        _require(isinstance(name, str), "Invalid manifest filename")
        logical = PurePosixPath(name)
        _require(not logical.is_absolute() and ".." not in logical.parts
                 and "\\" not in name and ":" not in name and str(logical) == name,
                 "Unsafe manifest path")
    _require(set(files) == BUNDLE_FILES, "Unexpected release file set")
    _require({path.name for path in root.iterdir()} == BUNDLE_FILES | {"manifest.json"},
             "Unexpected file outside release allowlist")
    for name, expected in files.items():
        _require(isinstance(expected, dict) and set(expected) == {"bytes", "sha256"}
                 and type(expected["bytes"]) is int and expected["bytes"] >= 0
                 and _valid_hash(expected["sha256"]), "Malformed checksum entry")
        path = _relative_file(root, name)
        _require(path.stat().st_size == expected["bytes"], "Release file size mismatch")
        _check_hash(path, expected["sha256"])
    original = manifest.get("original_inputs_sha256")
    _require(isinstance(original, dict) and set(original).issubset(PROVENANCE_KEYS)
             and all(_valid_hash(v) for v in original.values()),
             "Malformed historical provenance hashes")
    _require(all(original.get(key) == value for key, value in SOURCE_SHA256.items()),
             "Historical source pins changed")
    train = load_dataset(root / "train.npz")
    validation = load_dataset(root / "validation.npz")
    _validate_partition(train, "train", True)
    _validate_partition(validation, "validation", True)
    _require(not set(train["groups"].tolist()) & set(validation["groups"].tolist()),
             "Problems overlap TRAIN and VALIDATION")
    _models(root / "fitted-models.json", train)
    _predictions(root / "heldout-predictions.npz", validation)
    shapes = {"train": _shapes(train), "validation": _shapes(validation)}
    support = {"train": _support(train), "validation": _support(validation)}
    _require(manifest.get("shapes") == shapes and manifest.get("support") == support,
             "Manifest shape/support summary changed")
    _require(manifest.get("privacy") == PRIVACY, "Unknown privacy declaration")
    return {"status": "VERIFIED", "artifact_id": ARTIFACT_ID,
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "files": files, "support": support, "shapes": shapes,
            "original_test_used": False, "new_gpu_calls": 0}


def _anonymize(data: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    unique = sorted(set(data["groups"].tolist()))
    mapping = {group: f"{split}-g{i:03d}" for i, group in enumerate(unique)}
    result = dict(data)
    result["groups"] = np.asarray([mapping[group] for group in data["groups"].tolist()])
    result["row_ids"] = np.asarray([f"{split}-r{i:04d}" for i in range(len(data["row_ids"]))])
    _require(np.unique(result["groups"]).tolist() == [mapping[group] for group in unique],
             "Anonymous group sorting changed")
    for key in DATA_KEYS - {"groups", "row_ids"}:
        _require(result[key].dtype == data[key].dtype
                 and result[key].tobytes() == data[key].tobytes(), "Numeric export is not lossless")
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)


def export_public_bundle(source_repo: Path, destination: Path) -> dict[str, Any]:
    """Export once from pinned numerical evidence, without changing any source.

    Identifier maps live only in memory. The destination must be a fresh child
    of this new repository's ignored data/ directory, outside the source repo.
    A partial failed export is preserved and must not be silently overwritten.
    """
    source = Path(source_repo).resolve(strict=True)
    destination = Path(destination).resolve()
    private_root = (REPOSITORY / "data").resolve()
    _require(destination.is_relative_to(private_root) and destination != private_root,
             "Export must use the new repository data namespace")
    _require(not destination.is_relative_to(source), "Export overlaps historical source")
    _require(not destination.exists(), "Write-once destination already exists")
    inputs = {key: _relative_file(source, name) for key, name in SOURCE_FILES.items()}
    for key, path in inputs.items():
        _check_hash(path, SOURCE_SHA256[key])
    report = _json(inputs["public_report"])
    evidence = report.get("evidence_sha256")
    _require(isinstance(evidence, dict) and set(evidence).issubset(PROVENANCE_KEYS)
             and all(_valid_hash(v) for v in evidence.values()),
             "Invalid reviewed report provenance")
    for key in ("train_data", "validation_data", "fitted_models", "private_predictions"):
        _require(evidence.get(key) == SOURCE_SHA256[key], "Reviewed report/source disagreement")
    train = load_dataset(inputs["train_data"])
    validation = load_dataset(inputs["validation_data"])
    _validate_partition(train, "train", False)
    _validate_partition(validation, "validation", False)
    _require(not set(train["groups"].tolist()) & set(validation["groups"].tolist()),
             "Historical problem split overlaps")
    models = _models(inputs["fitted_models"], train)
    predictions = _predictions(inputs["private_predictions"], validation)
    public_train, public_validation = _anonymize(train, "train"), _anonymize(validation, "validation")
    public_predictions = dict(predictions)
    for key in ("groups", "row_ids"):
        public_predictions[key] = public_validation[key]
    destination.mkdir(parents=True, exist_ok=False)
    _write_npz(destination / "train.npz", public_train)
    _write_npz(destination / "validation.npz", public_validation)
    _write_json(destination / "fitted-models.json", models)
    _write_npz(destination / "heldout-predictions.npz", public_predictions)
    originals = {**evidence, **SOURCE_SHA256}
    manifest = {
        "schema_version": 1, "artifact_id": ARTIFACT_ID,
        "files": {name: {"bytes": (destination / name).stat().st_size,
                         "sha256": sha256_file(destination / name)} for name in sorted(BUNDLE_FILES)},
        "original_inputs_sha256": originals,
        "shapes": {"train": _shapes(public_train), "validation": _shapes(public_validation)},
        "support": {"train": _support(public_train), "validation": _support(public_validation)},
        "privacy": PRIVACY,
    }
    _write_json(destination / "manifest.json", manifest)
    verify_bundle(destination)
    # Recheck immutable original bytes after export; no original file is opened for writing.
    for key, path in inputs.items():
        _check_hash(path, SOURCE_SHA256[key])
    return manifest
