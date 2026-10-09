"""Lightweight report checks and transparent numerical reproduction comparison."""
from __future__ import annotations

from importlib.resources import files
import json
import math


def reference_results() -> dict:
    return json.loads(files("thinking_completion").joinpath("reference/historical_v1.json").read_text(encoding="utf-8"))


def verify_reference() -> dict:
    """Check report consistency; this alone does not reproduce the empirical study."""
    result = reference_results()
    if result["original_recovery_status"] != "NON_EVALUABLE" or result["original_test_used"] is not False:
        raise ValueError("Historical scientific scope changed")
    for split, expected in (("train", (576, 96)), ("validation", (192, 32))):
        support = result["support"][split]
        if (support["rows"], support["problems"]) != expected or support["positive_rows"] + support["negative_rows"] != support["rows"]:
            raise ValueError("Historical support record is inconsistent")
    for comparison, baseline, added in (("Clex_vs_Blex", "Blex", "Clex"), ("C_vs_B", "B", "C"), ("Blex_vs_B", "B", "Blex")):
        actual = result["metrics"][baseline]["problem_weighted_log_loss_nats"] - result["metrics"][added]["problem_weighted_log_loss_nats"]
        expected = result["paired_comparisons"][comparison]["mean_problem_log_loss_improvement_nats"]
        if not math.isclose(actual, expected, abs_tol=1e-14):
            raise ValueError("Historical paired effect is inconsistent")
    return {"status": "PASS", "scope": "Report consistency only, not numerical/data/GPU reproduction", "original_recovery": "NON_EVALUABLE", "original_test_used": False, "gpu_calls": 0}


def compare_results(actual: dict, *, tolerance: float = 1e-8) -> dict:
    """Compare all reported metrics, paired and matched effects—not just one gain."""
    expected = reference_results()
    if not math.isfinite(tolerance) or tolerance <= 0 or tolerance > 1e-4:
        raise ValueError("Tolerance must be finite, positive, and at most 1e-4")
    checks: list[dict] = []

    def check(label: str, observed: float, target: float) -> None:
        if not math.isfinite(float(observed)):
            raise ValueError("Reproduction emitted a nonfinite statistic")
        error = abs(float(observed) - float(target))
        checks.append({"statistic": label, "absolute_error": error, "passed": error <= tolerance})

    for model, values in expected["metrics"].items():
        for metric, target in values.items():
            check(f"metrics.{model}.{metric}", actual["metrics"][model][metric], target)
    for name, values in expected["paired_comparisons"].items():
        found = actual["paired_comparisons"][name]
        check(f"paired.{name}.effect", found["mean_problem_log_loss_improvement_nats"], values["mean_problem_log_loss_improvement_nats"])
        for index in range(2):
            check(f"paired.{name}.ci.{index}", found["problem_cluster_bootstrap_95_percentile_ci"][index], values["problem_cluster_bootstrap_95_percentile_ci"][index])
    matched = actual["matched_control"]
    for key in ("Blex_concordance", "Clex_concordance", "paired_concordance_difference", "matched_problem_count"):
        check(f"matched.{key}", matched[key], expected["matched_control"][key])
    for index in range(2):
        check(f"matched.ci.{index}", matched["difference_bootstrap_95_percentile_ci"][index], expected["matched_control"]["difference_bootstrap_95_percentile_ci"][index])
    return {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "tolerance": tolerance, "checks": len(checks), "max_absolute_error": max(c["absolute_error"] for c in checks), "mismatches": [c for c in checks if not c["passed"]]}
