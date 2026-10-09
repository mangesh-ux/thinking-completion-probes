"""Problem-weighted metrics for bounded natural-thinking completion.

These functions reproduce the historical exploratory statistics. They neither
fit a predictor nor reinterpret completion as correctness or answer recovery.
Bootstrap intervals condition on the already selected, frozen trained model.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ("A", "B", "Blex", "C", "Clex")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261008


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _groups(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups)
    _require(values.ndim == 1 and len(values) > 0, "Nonempty one-dimensional groups required")
    _require(values.dtype.kind in "US", "Group identifiers must be strings")
    _require(all(str(value).strip() for value in values), "Empty group identifier")
    return values


def _targets(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    _require("labels" in data and "groups" in data, "Labels and groups required")
    y, groups = np.asarray(data["labels"]), _groups(data["groups"])
    _require(y.ndim == 1 and len(y) == len(groups), "Label/group row counts differ")
    _require(y.dtype.kind in "bifu" and np.isfinite(y).all(), "Invalid label array")
    _require(set(y) == {0, 1}, "Both binary classes required")
    return y, groups


def weights(groups: np.ndarray) -> np.ndarray:
    """Mean-one row weights, each problem contributing equal total weight."""
    groups = _groups(groups)
    counts = Counter(groups)
    return np.array([(len(groups) / len(counts)) / counts[group] for group in groups])


def binary_losses(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    y, p = np.asarray(y), np.asarray(probabilities)
    _require(y.ndim == p.ndim == 1 and y.shape == p.shape and len(y) > 0, "Prediction/target shape differs")
    _require(y.dtype.kind in "bifu" and np.isfinite(y).all() and set(y) <= {0, 1}, "Nonbinary target")
    _require(p.dtype.kind in "fiu" and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid prediction probabilities")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def problem_means(values: np.ndarray, groups: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Sorted group order is significant for the frozen bootstrap draw stream."""
    groups, values = _groups(groups), np.asarray(values)
    _require(values.ndim == 1 and len(values) == len(groups), "Value/group shape differs")
    _require(np.isfinite(values).all(), "Nonfinite problem statistic")
    unique = sorted(set(groups))
    return unique, np.array([np.mean(values[groups == group]) for group in unique])


def score(y: np.ndarray, probabilities: np.ndarray, groups: np.ndarray) -> float:
    return float(problem_means(binary_losses(y, probabilities), groups)[1].mean())


def bootstrap_interval(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> list[float]:
    """Retain duplicate sampled problems rather than deduplicating each draw."""
    values = np.asarray(values)
    _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Invalid bootstrap values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    replicates = np.mean(values[indices], axis=1)
    return [float(value) for value in np.quantile(replicates, [0.025, 0.975])]


def paired_bootstrap(
    values: np.ndarray,
    positive_support: np.ndarray,
    negative_support: np.ndarray,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    values = np.asarray(values)
    positive, negative = np.asarray(positive_support), np.asarray(negative_support)
    _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Invalid paired statistics")
    _require(positive.shape == negative.shape == values.shape and positive.dtype.kind == negative.dtype.kind == "b", "Invalid group class support")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    valid = positive[indices].any(axis=1) & negative[indices].any(axis=1)
    count = int(valid.sum())
    interval = (
        [float(value) for value in np.quantile(values[indices[valid]].mean(axis=1), [0.025, 0.975])]
        if count >= 1900 else None
    )
    return {
        "problem_cluster_bootstrap_95_percentile_ci": interval,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "valid_resamples": count,
        "invalid_fraction": 1 - count / BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def metrics(data: Mapping[str, np.ndarray], probabilities: np.ndarray) -> dict:
    """PR-AUC is the historical weighted average-precision convention."""
    y, groups = _targets(data)
    loss = score(y, probabilities, groups)
    w = weights(groups)
    return {
        "problem_weighted_log_loss_nats": loss,
        "problem_weighted_brier": float(np.average((probabilities - y) ** 2, weights=w)),
        "problem_weighted_pr_auc_average_precision": float(average_precision_score(y, probabilities, sample_weight=w)),
        "problem_weighted_roc_auc": float(roc_auc_score(y, probabilities, sample_weight=w)),
    }


def matched(data: Mapping[str, np.ndarray], probabilities: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Equal-problem positive-versus-negative concordance, with half-credit ties."""
    y, all_groups = _targets(data)
    binary_losses(y, probabilities)  # Probability and pairing validation only.
    values, groups = [], []
    for group in sorted(set(all_groups)):
        mask = all_groups == group
        targets, scores = y[mask], probabilities[mask]
        if set(targets) != {0, 1}:
            continue
        delta = scores[targets == 1, None] - scores[targets == 0][None, :]
        values.append(float(np.mean((delta > 0) + 0.5 * (delta == 0))))
        groups.append(group)
    return groups, np.array(values)


def evaluate_predictions(
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    train_prevalence: float,
) -> dict:
    """Recompute the predeclared aggregate result from paired saved predictions.

    No feature arrays are accessed. This is also the lightweight reproduction
    route for an anonymized derivative bundle without repeating any model fit.
    """
    y, groups = _targets(data)
    _require(set(predictions) == set(MODELS), "Exactly five frozen model predictions required")
    _require(np.isfinite(train_prevalence) and 0 <= train_prevalence <= 1, "Invalid TRAIN prevalence")
    predictions = {name: np.asarray(predictions[name]) for name in MODELS}
    for values in predictions.values():
        binary_losses(y, values)
    results = {name: metrics(data, values) for name, values in predictions.items()}
    results["TRAIN_prevalence"] = metrics(data, np.full(len(y), train_prevalence))
    paired = {}
    for baseline, added in (("Blex", "Clex"), ("B", "C"), ("B", "Blex")):
        ordered, deltas = problem_means(binary_losses(y, predictions[baseline]) - binary_losses(y, predictions[added]), groups)
        positive = np.array([np.any(y[groups == group] == 1) for group in ordered])
        negative = np.array([np.any(y[groups == group] == 0) for group in ordered])
        paired[added + "_vs_" + baseline] = {
            "mean_problem_log_loss_improvement_nats": float(deltas.mean()),
            **paired_bootstrap(deltas, positive, negative),
            "problem_count": len(deltas),
            "positive_favors_added_predictors": True,
        }
    mg, concordance = matched(data, predictions["Clex"])
    bg, baseline_concordance = matched(data, predictions["Blex"])
    _require(mg == bg, "Matched model populations differ")
    matched_result = {
        "matched_problem_count": len(mg),
        "adequate_support": len(mg) >= 8,
        "Clex_concordance": float(concordance.mean()) if len(mg) else None,
        "Blex_concordance": float(baseline_concordance.mean()) if len(mg) else None,
        "paired_concordance_difference": float((concordance - baseline_concordance).mean()) if len(mg) else None,
        "difference_bootstrap_95_percentile_ci": bootstrap_interval(concordance - baseline_concordance) if len(mg) >= 8 else None,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES if len(mg) >= 8 else 0,
        "bootstrap_valid_resamples": BOOTSTRAP_RESAMPLES if len(mg) >= 8 else 0,
    }
    return {
        "metrics": results,
        "paired_comparisons": paired,
        "matched_control": matched_result,
        "rows": len(y),
        "problems": len(set(groups)),
        "positive_rows": int(y.sum()),
        "problem_weighted_prevalence": float(problem_means(y, groups)[1].mean()),
    }
