"""Write-once CPU reproduction of the frozen exploratory probe procedure.

Only whitelisted checkpoint predictors enter a design matrix. Selection uses
the supplied, historically frozen TRAIN fold assignments: anonymous problem
identifiers are never rehashed to construct new folds. Held-out prediction
restores serialized TRAIN models without fitting or selecting anything.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Mapping
import warnings

# Setting these before numerical imports also supports the standalone module.
# threadpool_limits below handles libraries already imported by an API caller.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from . import metrics as statistics

GRID = (0.001, 0.01, 0.1, 1.0)
MODELS = statistics.MODELS
RUNTIME = {"numpy": "2.2.6", "scipy": "1.14.0", "scikit-learn": "1.5.1"}
FEATURE_NAMES = (
    "checkpoint_token_position", "normalized_checkpoint_position", "problem_token_count", "difficulty",
    "mean_answer_token_logprob", "min_answer_token_logprob", "answer_token_count",
    "first_readout_token_entropy_nats", "first_readout_token_probability_margin",
    "previous_readout_agreement", "previous_readout_available",
)
ROW_KEYS = ("numeric", "topics", "groups", "lexical", "activations", "labels", "row_ids", "fold_ids", "difficulty")
ALLOWED_KEYS = frozenset(ROW_KEYS + ("numeric_feature_names",))
PROCEDURE = {
    "schema_version": 1, "checkpoint": 512, "reasoning_cap": 4096,
    "folds": 5, "folds_are_stored_not_reconstructed": True,
    "grid": GRID, "activation_layers_zero_based": list(range(28)),
    "solver": "lbfgs", "penalty": "l2", "max_iter": 10000, "tol": 1e-6, "random_state": 0,
    "variance_keep_threshold": 1e-12, "missing_numeric_imputation": "unweighted_observed_TRAIN_median_or_zero",
    "scale_weighting": "equal_problem_mean_one", "primary": "Clex_vs_Blex",
    "secondary_C_layer": "selected_Clex_layer", "selection_ties": ["smaller_C", "shallower_layer"],
    "bootstrap_resamples": statistics.BOOTSTRAP_RESAMPLES, "bootstrap_seed": statistics.BOOTSTRAP_SEED,
    "interval_conditions_on_selected_frozen_model": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sources() -> dict[str, str]:
    return {"modeling.py": _sha_file(Path(__file__)), "metrics.py": _sha_file(Path(statistics.__file__))}


def _runtime() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in RUNTIME}
    _require(actual == RUNTIME, "CPU numerical runtime differs from the frozen procedure")
    return actual


def _write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _new_output(path: Path) -> Path:
    _require(isinstance(path, Path), "An explicit Path output directory is required")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _failure(output: Path, phase: str, error: Exception) -> None:
    # Do not publish exception text, file paths or input row identities.
    _write_once(output / "failure.json", {"failed_at_utc": _utc(), "phase": phase, "error_type": type(error).__name__, "automatic_retry_allowed": False})


def _feature_names(data: Mapping) -> tuple[str, ...]:
    _require("numeric_feature_names" in data, "Explicit numeric feature whitelist required")
    names = np.asarray(data["numeric_feature_names"])
    _require(names.shape == (11,) and names.dtype.kind in "US", "Invalid numeric feature names")
    _require(tuple(names) == FEATURE_NAMES, "Numeric predictor whitelist differs")
    return FEATURE_NAMES


def _validate_design(data: Mapping) -> int:
    _require(isinstance(data, Mapping), "Array mapping required")
    _require(not (set(data) - ALLOWED_KEYS), "Unrecognized metadata or predictor route")
    _feature_names(data)
    for key in ("numeric", "topics", "lexical", "activations"):
        _require(key in data and isinstance(data[key], np.ndarray), "Missing array in predictor schema")
    numeric = data["numeric"]
    _require(numeric.ndim == 2 and numeric.shape[1] == 11 and len(numeric) > 0, "Numeric feature shape differs")
    n = len(numeric)
    _require(numeric.dtype == np.dtype("float64") and not np.isinf(numeric).any(), "Numeric predictors must be finite or NaN float64")
    _require(data["topics"].shape == (n,) and data["topics"].dtype.kind == "U", "Topic schema differs")
    _require(all(str(value).strip() for value in data["topics"]), "Empty topic")
    lexical = data["lexical"]
    _require(lexical.shape == (n, 512) and lexical.dtype == np.dtype("float64") and np.isfinite(lexical).all(), "Lexical feature schema differs")
    _require((lexical >= 0).all() and (lexical == np.floor(lexical)).all() and (lexical.sum(axis=1) == 128).all(), "Hashed trailing-prefix count contract differs")
    _require(data["activations"].shape == (n, 28, 2048) and data["activations"].dtype == np.dtype("float32") and np.isfinite(data["activations"]).all(), "Activation feature schema differs")
    return n


def validate_dataset(data: Mapping, split: str) -> None:
    """Validate the frozen data contract and population support before fitting."""
    _require(split in ("train", "validation"), "Only TRAIN/VALIDATION are supported")
    n = _validate_design(data)
    _require(all(key in data and isinstance(data[key], np.ndarray) for key in ROW_KEYS), "Dataset row fields absent")
    _require(all(len(data[key]) == n for key in ROW_KEYS), "Dataset row counts differ")
    for key in ("groups", "row_ids"):
        _require(data[key].shape == (n,) and data[key].dtype.kind == "U", "Identifier schema differs")
        _require(all(str(value).strip() for value in data[key]), "Empty identifier")
    _require(len(set(data["row_ids"])) == n, "Duplicate row identity")
    y, groups, folds = data["labels"], data["groups"], data["fold_ids"]
    _require(y.shape == (n,) and y.dtype == np.dtype("int8") and set(y) == {0, 1}, "Both binary classes required")
    _require(folds.shape == (n,) and folds.dtype == np.dtype("int8"), "Stored fold schema differs")
    _require(data["difficulty"].shape == (n,) and data["difficulty"].dtype == np.dtype("float64") and np.isin(data["difficulty"], [7.0, 8.0]).all(), "Difficulty schema differs")
    _require(np.array_equal(data["difficulty"], data["numeric"][:, 3]), "Difficulty metadata and structural predictor differ")
    _require(np.isfinite(data["numeric"][:, :4]).all(), "Structural predictors missing")
    _require((data["numeric"][:, 0] == 512).all() and (data["numeric"][:, 1] == 512 / 4096).all(), "Frozen checkpoint coordinate differs")
    token_count = data["numeric"][:, 2]
    _require((token_count >= 0).all() and (token_count == np.floor(token_count)).all(), "Invalid problem token count")
    _require(len(set(groups)) <= (96 if split == "train" else 32), "Frozen split population exceeded")
    for group in set(groups):
        mask = groups == group
        _require(len(set(folds[mask])) == 1, "A problem crosses stored folds")
        _require(len(set(data["difficulty"][mask])) == 1 and len(set(data["topics"][mask])) == 1, "Problem covariate changes across rows")
    row_floor, problem_floor = (50, 15) if split == "train" else (20, 8)
    for label in (0, 1):
        _require(int(np.sum(y == label)) >= row_floor and len(set(groups[y == label])) >= problem_floor, "Predeclared target diversity gate failed")
    if split == "train":
        _require(set(folds) == set(range(5)), "All five stored TRAIN folds required")
        for fold in range(5):
            _require(set(y[folds != fold]) == {0, 1}, "A grouped fit partition is single class")
    else:
        _require(set(folds) == {-1}, "VALIDATION must not carry TRAIN CV folds")


def dataset_fingerprint(data: Mapping) -> str:
    """Canonical array identity, explicitly not an NPZ byte-file identity."""
    _feature_names(data)
    digest = hashlib.sha256()
    for key in ROW_KEYS + ("numeric_feature_names",):
        value = np.asarray(FEATURE_NAMES) if key == "numeric_feature_names" else data[key]
        header = json.dumps({"name": key, "dtype": value.dtype.str, "shape": value.shape}, separators=(",", ":"), sort_keys=True).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = np.ascontiguousarray(value).tobytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def subset(data: Mapping, mask: np.ndarray) -> dict:
    """Row fields are sliced; the feature-name schema is not a row field."""
    return {key: value[mask] if key in ROW_KEYS else value for key, value in data.items()}


class Preprocessor:
    """Historical TRAIN-only imputation, vocabulary and weighted standardization."""

    def __init__(self, model: str, layer: int | None = None):
        _validate_model(model, layer)
        self.model, self.layer = model, layer

    def numeric(self, data: Mapping) -> np.ndarray:
        return data["numeric"][:, :4] if self.model == "A" else data["numeric"]

    def fit(self, data: Mapping) -> "Preprocessor":
        _validate_design(data)
        raw = self.numeric(data)
        self.medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0 for column in raw.T])
        self.topics = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        self.topics.fit(data["topics"].reshape(-1, 1))
        design = self.unscaled(data)
        w = statistics.weights(data["groups"])
        self.mean = np.average(design, axis=0, weights=w)
        variance = np.average((design - self.mean) ** 2, axis=0, weights=w)
        _require(np.isfinite(self.mean).all() and np.isfinite(variance).all(), "Preprocessing emitted nonfinite state")
        self.keep = variance > 1e-12
        self.scale = np.sqrt(variance[self.keep])
        _require(self.keep.any(), "All training features constant")
        return self

    def unscaled(self, data: Mapping) -> np.ndarray:
        raw = self.numeric(data)
        missing = ~np.isfinite(raw)
        blocks = [np.where(missing, self.medians, raw), missing.astype(float), self.topics.transform(data["topics"].reshape(-1, 1))]
        if self.model in ("Blex", "Clex"):
            blocks.append(data["lexical"])
        if self.model in ("C", "Clex"):
            blocks.append(data["activations"][:, self.layer, :])
        return np.concatenate(blocks, axis=1)

    def transform(self, data: Mapping) -> np.ndarray:
        _validate_design(data)
        design = (self.unscaled(data)[:, self.keep] - self.mean[self.keep]) / self.scale
        _require(np.isfinite(design).all(), "Transformed predictors nonfinite")
        return design


def _validate_model(model: str, layer: int | None) -> None:
    _require(model in MODELS, "Unknown model family")
    if model in ("C", "Clex"):
        _require(type(layer) is int and 0 <= layer < 28, "One valid activation layer required")
    else:
        _require(layer is None, "Non-activation model cannot select a layer")


def fit_predict(train: Mapping, evaluation: Mapping, model: str, strength: float, layer: int | None = None) -> tuple[np.ndarray, dict]:
    """One bounded fit; callers control TRAIN-only CV or final TRAIN fitting."""
    _validate_model(model, layer)
    _require(strength in GRID and np.isfinite(strength), "Regularization outside the frozen grid")
    _require(set(train["labels"]) == {0, 1}, "Fit partition is single class")
    with threadpool_limits(limits=1):
        preprocessing = Preprocessor(model, layer).fit(train)
        estimator = LogisticRegression(C=strength, solver="lbfgs", penalty="l2", max_iter=10000, tol=1e-6, random_state=0)
        w = statistics.weights(train["groups"])
        w *= len(w) / w.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(preprocessing.transform(train), train["labels"], sample_weight=w)
        _require((estimator.n_iter_ < 10000).all(), "Optimizer iteration cap reached")
        probabilities = estimator.predict_proba(preprocessing.transform(evaluation))[:, 1]
    _require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), "Model emitted invalid probabilities")
    state = {
        "model": model, "layer": layer, "C": strength,
        "numeric_medians": preprocessing.medians.tolist(),
        "topic_vocabulary": preprocessing.topics.categories_[0].tolist(),
        "kept_columns": np.flatnonzero(preprocessing.keep).tolist(),
        "means": preprocessing.mean[preprocessing.keep].tolist(), "scales": preprocessing.scale.tolist(),
        "coefficients": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
    }
    return probabilities, state


def predict_frozen(data: Mapping, state: Mapping) -> np.ndarray:
    """Restore a linear model without calling any fitting or vocabulary learner."""
    _validate_design(data)
    expected = {"model", "layer", "C", "numeric_medians", "topic_vocabulary", "kept_columns", "means", "scales", "coefficients", "intercept"}
    _require(isinstance(state, Mapping) and set(state) == expected, "Saved model schema differs")
    _validate_model(state["model"], state["layer"])
    _require(state["C"] in GRID, "Saved regularization outside frozen grid")
    raw = data["numeric"][:, :4] if state["model"] == "A" else data["numeric"]
    missing = ~np.isfinite(raw)
    medians = np.asarray(state["numeric_medians"])
    vocabulary = state["topic_vocabulary"]
    _require(medians.shape == (raw.shape[1],) and np.isfinite(medians).all(), "Saved numeric imputation state differs")
    _require(isinstance(vocabulary, list) and len(vocabulary) > 0 and all(isinstance(value, str) and value.strip() for value in vocabulary) and vocabulary == sorted(set(vocabulary)), "Saved topic vocabulary differs")
    topics = np.array([[float(topic == category) for category in vocabulary] for topic in data["topics"]])
    blocks = [np.where(missing, medians, raw), missing.astype(float), topics]
    if state["model"] in ("Blex", "Clex"):
        blocks.append(data["lexical"])
    if state["model"] in ("C", "Clex"):
        blocks.append(data["activations"][:, state["layer"], :])
    design = np.concatenate(blocks, axis=1)
    kept = np.asarray(state["kept_columns"])
    _require(kept.ndim == 1 and len(kept) > 0 and kept.dtype.kind in "iu" and (kept >= 0).all() and (kept < design.shape[1]).all() and np.array_equal(kept, np.unique(kept)), "Saved feature selection differs")
    means, scales, coefficients = (np.asarray(state[name]) for name in ("means", "scales", "coefficients"))
    _require(all(value.shape == kept.shape and np.isfinite(value).all() for value in (means, scales, coefficients)) and (scales > 0).all() and np.isfinite(state["intercept"]), "Invalid saved linear state")
    with threadpool_limits(limits=1):
        scaled = (design[:, kept] - means) / scales
        probabilities = expit(scaled @ coefficients + state["intercept"])
    _require(np.isfinite(probabilities).all(), "Saved model emitted invalid probabilities")
    return probabilities


def _best(records: list[dict]) -> dict:
    return min(records, key=lambda record: (record["cv_problem_weighted_log_loss_nats"], record["C"], -1 if record["layer"] is None else record["layer"]))


def _cv(train: Mapping, model: str, strength: float, layer: int | None) -> dict:
    folds = train["fold_ids"]
    probabilities = np.zeros(len(folds))
    for fold in range(5):
        probabilities[folds == fold], _ = fit_predict(subset(train, folds != fold), subset(train, folds == fold), model, strength, layer)
    return {"model": model, "layer": layer, "C": strength, "cv_problem_weighted_log_loss_nats": statistics.score(train["labels"], probabilities, train["groups"])}


def select(train: dict, output: Path) -> dict:
    """Select 128 TRAIN CV configurations, then freeze all five final models.

    A fresh output directory is required. Any failed configuration is recorded
    and prevents a selection from being released; candidates are never dropped.
    """
    runtime = _runtime()
    validate_dataset(train, "train")
    fingerprint, sources = dataset_fingerprint(train), _sources()
    output = _new_output(output)
    try:
        assignments = {group: int(train["fold_ids"][np.flatnonzero(train["groups"] == group)[0]]) for group in sorted(set(train["groups"]))}
        _write_once(output / "selection-intent.json", {
            "created_at_utc": _utc(), "record_type": "PUBLIC_COMPLETION_SELECTION_INTENT_V1",
            "procedure_sha256": _canonical(PROCEDURE), "source_sha256": sources,
            "train_array_fingerprint_sha256": fingerprint, "runtime": runtime,
            "fold_assignment": assignments, "grid": GRID, "layers_zero_based": list(range(28)), "paid_gpu_calls": 0,
        })
        scores, failures = [], []
        for model in ("A", "B", "Blex", "Clex"):
            for layer in (range(28) if model == "Clex" else (None,)):
                for strength in GRID:
                    try:
                        record = _cv(train, model, strength, layer)
                        scores.append(record)
                    except Exception as error:
                        record = {"model": model, "layer": layer, "C": strength, "error_type": type(error).__name__}
                        failures.append(record)
                    _write_once(output / f"cv-config-{len(scores) + len(failures):03d}.json", record)
        _require(not failures, "A predeclared configuration failed; no selection is permitted")
        chosen = {model: _best([record for record in scores if record["model"] == model]) for model in ("A", "B", "Blex", "Clex")}
        layer, secondary = chosen["Clex"]["layer"], []
        for strength in GRID:
            try:
                record = _cv(train, "C", strength, layer)
                secondary.append(record)
            except Exception as error:
                record = {"model": "C", "layer": layer, "C": strength, "error_type": type(error).__name__}
                _write_once(output / f"cv-secondary-C-{len(secondary) + 1:02d}.json", record)
                raise
            _write_once(output / f"cv-secondary-C-{len(secondary):02d}.json", record)
        chosen["C"] = _best(secondary)
        _require(len(scores + secondary) == 128, "Frozen configuration count differs")
        fitted = {}
        for model in MODELS:
            config = chosen[model]
            expected, fitted[model] = fit_predict(train, train, model, config["C"], config["layer"])
            _require(np.allclose(expected, predict_frozen(train, fitted[model]), rtol=1e-10, atol=1e-12), "Serialized model prediction mismatch")
        _require(dataset_fingerprint(train) == fingerprint, "TRAIN arrays changed during selection")
        prevalence = float(statistics.problem_means(train["labels"], train["groups"])[1].mean())
        _write_once(output / "fitted-models.json", {"models": fitted, "train_problem_prevalence": prevalence})
        record = {
            "created_at_utc": _utc(), "record_type": "PUBLIC_COMPLETION_TRAIN_SELECTION_V1",
            "procedure_sha256": _canonical(PROCEDURE), "source_sha256": sources,
            "train_array_fingerprint_sha256": fingerprint, "runtime": runtime,
            "fitted_models_sha256": _sha_file(output / "fitted-models.json"),
            "selected": chosen, "cv_scores": scores + secondary, "failures": failures,
            "fold_assignment": assignments, "validation_loaded": False, "original_test_loaded": False,
            "selection_cv_is_not_unbiased_performance": True,
        }
        _write_once(output / "selection.json", record)
        _write_once(output / "selection-lock.json", {
            "created_at_utc": _utc(), "record_type": "PUBLIC_COMPLETION_SELECTION_LOCK_V1",
            "selection_sha256": _sha_file(output / "selection.json"),
            "fitted_models_sha256": record["fitted_models_sha256"],
            "train_array_fingerprint_sha256": fingerprint, "source_sha256": sources,
        })
        return record
    except Exception as error:
        _failure(output, "select", error)
        raise


def _validate_selection(selection: Mapping) -> None:
    _require(selection.get("record_type") == "PUBLIC_COMPLETION_TRAIN_SELECTION_V1", "Unsupported selection schema")
    _require(selection["procedure_sha256"] == _canonical(PROCEDURE) and selection["source_sha256"] == _sources(), "Selection numerical procedure or source changed")
    _require(selection["runtime"] == RUNTIME and selection["failures"] == [] and selection["validation_loaded"] is False and selection["original_test_loaded"] is False, "Selection provenance or failure gate differs")
    records, selected = selection["cv_scores"], selection["selected"]
    _require(len(records) == 128 and set(selected) == set(MODELS), "Selection family coverage differs")
    _require(Counter(record["model"] for record in records) == {"A": 4, "B": 4, "Blex": 4, "Clex": 112, "C": 4}, "Selection configuration coverage differs")
    _require(len({(record["model"], record["layer"], record["C"]) for record in records}) == 128, "Duplicate configuration")
    for record in records:
        _validate_model(record["model"], record["layer"])
        _require(record["C"] in GRID and np.isfinite(record["cv_problem_weighted_log_loss_nats"]), "Invalid CV record")
    for model in MODELS:
        _require(selected[model] == _best([record for record in records if record["model"] == model]), "Stored selection violates TRAIN tie policy")
    _require(selected["C"]["layer"] == selected["Clex"]["layer"], "Secondary C changed primary layer")
    secondary = [record for record in records if record["model"] == "C"]
    _require(all(record["layer"] == selected["Clex"]["layer"] for record in secondary) and {record["C"] for record in secondary} == set(GRID), "Secondary C search differs")


def evaluate(train: dict, validation: dict, selection_directory: Path, output: Path) -> dict:
    """One held-out exploratory evaluation of frozen TRAIN models, without refit."""
    _runtime()
    validate_dataset(train, "train")
    selection_directory = selection_directory.resolve()
    selection_path, models_path = selection_directory / "selection.json", selection_directory / "fitted-models.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _validate_selection(selection)
    lock = json.loads((selection_directory / "selection-lock.json").read_text(encoding="utf-8"))
    _require(lock.get("record_type") == "PUBLIC_COMPLETION_SELECTION_LOCK_V1"
             and lock.get("selection_sha256") == _sha_file(selection_path)
             and lock.get("fitted_models_sha256") == selection["fitted_models_sha256"]
             and lock.get("train_array_fingerprint_sha256") == selection["train_array_fingerprint_sha256"]
             and lock.get("source_sha256") == selection["source_sha256"], "Frozen selection lock changed")
    _require(dataset_fingerprint(train) == selection["train_array_fingerprint_sha256"], "TRAIN arrays changed since selection")
    _require(_sha_file(models_path) == selection["fitted_models_sha256"], "Frozen model file changed")
    fitted = json.loads(models_path.read_text(encoding="utf-8"))
    _require(set(fitted) == {"models", "train_problem_prevalence"} and set(fitted["models"]) == set(MODELS), "Frozen model bundle differs")
    expected_prevalence = float(statistics.problem_means(train["labels"], train["groups"])[1].mean())
    _require(fitted["train_problem_prevalence"] == expected_prevalence, "TRAIN prevalence changed")
    for model in MODELS:
        state, chosen = fitted["models"][model], selection["selected"][model]
        _require(state["model"] == model and state["C"] == chosen["C"] and state["layer"] == chosen["layer"], "Frozen model and selected configuration differ")
    output = _new_output(output)
    try:
        selection_hash = _sha_file(selection_path)
        intent = {
            "created_at_utc": _utc(), "record_type": "PUBLIC_COMPLETION_EVALUATION_INTENT_V1",
            "selection_sha256": selection_hash, "fitted_models_sha256": selection["fitted_models_sha256"],
            "selection_lock_sha256": _sha_file(selection_directory / "selection-lock.json"),
            "no_refit_on_validation": True, "original_test_loaded": False,
            "evaluation_reservation_consumed": True, "automatic_retry_allowed": False,
        }
        # The marker reserves this selection once even if the held-out support
        # gate later fails. A deliberate fresh reproduction must rerun TRAIN
        # selection in a fresh run root, never silently retry this evaluation.
        _write_once(selection_directory / "evaluation-consumed.json", intent)
        _write_once(output / "evaluation-intent.json", intent)
        validate_dataset(validation, "validation")
        _require(not (set(train["groups"]) & set(validation["groups"])), "Problem leakage across TRAIN/VALIDATION")
        _require(not (set(train["row_ids"]) & set(validation["row_ids"])), "Row identity leakage across TRAIN/VALIDATION")
        validation_fingerprint = dataset_fingerprint(validation)
        predictions = {model: predict_frozen(validation, fitted["models"][model]) for model in MODELS}
        result = statistics.evaluate_predictions(validation, predictions, fitted["train_problem_prevalence"])
        _require(dataset_fingerprint(validation) == validation_fingerprint and dataset_fingerprint(train) == selection["train_array_fingerprint_sha256"], "Input arrays changed during evaluation")
        _require(_sha_file(selection_path) == selection_hash and _sha_file(models_path) == selection["fitted_models_sha256"], "Frozen model/selection changed during evaluation")
        record = {
            "record_type": "PUBLIC_COMPLETION_VALIDATION_HELDOUT_RESULT_V1", "completed_at_utc": _utc(),
            "question": "Future natural thinking completion before the frozen token cap, not correctness or recovery",
            "procedure_sha256": selection["procedure_sha256"], "source_sha256": selection["source_sha256"],
            "selection_sha256": selection_hash, "fitted_models_sha256": selection["fitted_models_sha256"],
            "selection_lock_sha256": _sha_file(selection_directory / "selection-lock.json"),
            "train_array_fingerprint_sha256": selection["train_array_fingerprint_sha256"],
            "validation_array_fingerprint_sha256": validation_fingerprint,
            **result, "original_test_loaded": False, "gpu_calls": 0, "no_validation_refit": True,
            "intervals_condition_on_selected_frozen_model": True,
        }
        with (output / "heldout-predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, labels=validation["labels"], groups=validation["groups"], row_ids=validation["row_ids"], **predictions)
        record["heldout_predictions_sha256"] = _sha_file(output / "heldout-predictions.npz")
        _write_once(output / "heldout-result.json", record)
        return record
    except Exception as error:
        _failure(output, "evaluate", error)
        raise
