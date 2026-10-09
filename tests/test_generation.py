"""Synthetic contracts only: no model, CUDA invocation, downloads, or benchmark rows."""
import copy
import io
import importlib.util
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="CPU BF16 fixture needs optional torch")

from thinking_completion.generation import (
    DEFAULT_CONFIG, extract_features, generate, generation_plan,
    manifest_from_rows, prepare_manifest, _verify_original_membership, _private,
)
from thinking_completion.generation_helpers.pure import (
    boxed_boundary, canonical, completion_target, fold_assignments,
    lexical_counts, numeric_features, object_sha, reasoning_boundary,
    reduce_readout, rollout_seed, sha, encode_nonfinite_scalars,
)


@pytest.fixture(autouse=True)
def synthetic_membership_trust_mock(monkeypatch, tmp_path):
    """Mock only the official-cohort trust check for synthetic operation fixtures.

    Fixed scientific config, row bindings, structural invariants and I/O
    verification remain real. The verifier itself rejects fixtures below.
    """
    monkeypatch.setattr("thinking_completion.generation._verify_original_membership",
                        lambda records, folds: None)
    # Every synthetic operation gets a separate explicit publication checkout;
    # the production containment/cwd/project-name guard remains unmodified.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "thinking-completion-probes"\n')
    (tmp_path / ".gitignore").write_text("artifacts/\n")
    monkeypatch.chdir(tmp_path)


def synthetic_rows(count=90):
    return [{"question": f"Synthetic fixture {difficulty}:{i}", "topic": "fixture",
             "difficulty": difficulty, "final_answer": "fixture-answer"}
            for difficulty in (7.0, 8.0) for i in range(count)]


def save_manifest(tmp_path):
    path = tmp_path / "artifacts" / "manifest.json"
    path.parent.mkdir()
    manifest = manifest_from_rows({}, synthetic_rows())
    path.write_bytes(canonical(manifest))
    return path, manifest


class FakeRuntime:
    """CPU synthetic model-operation fixture, not evidence of CUDA success."""
    marker, cue, eos = [99], [99, 100], [98]
    def __init__(self, *, fail=False, early=False):
        self.fail, self.early = fail, early
        self.activation_inputs = []
        self.readout_inputs = []
        self.provenance = {"close_marker_token_ids": self.marker,
                           "cue_token_ids": self.cue, "eos_token_ids": self.eos,
                           "real_gpu_regeneration_exercised": False, "fixture": True}
    def prompt(self, question):
        return {"token_ids": [1, 2], "problem_token_count": 3}
    def reasoning(self, ids, seed):
        if self.fail:
            raise RuntimeError("synthetic failure")
        return {"tokens": [3] * (100 if self.early else 512) + [99],
                "termination": "THINK_CLOSE_MARKER"}
    def readout(self, ids):
        self.readout_inputs.append(list(ids))
        return {"tokens": [4, 98], "termination": "EOS",
                "scores": {"normalized_answer": r"\boxed{fixture}",
                           "mean_answer_token_logprob": -0.3,
                           "min_answer_token_logprob": -0.4, "answer_token_count": 1,
                           "first_readout_token_entropy_nats": 0.7,
                           "first_readout_token_probability_margin": 0.5}}
    def activation(self, ids):
        self.activation_inputs.append(list(ids))
        matrix = torch.zeros((28, 2048), dtype=torch.bfloat16)
        stream = io.BytesIO()
        torch.save(matrix, stream)
        header = canonical({"shape": [28, 2048], "dtype": "bfloat16",
                            "representation": "post-block-last-reasoning-token-v1"})
        return stream.getvalue(), {"shape": [28, 2048], "dtype": "bfloat16",
                "matrix_sha256": sha(header + b"\0" + matrix.view(torch.uint8).numpy().tobytes())}


def test_manifest_exact_counts_and_grouping():
    manifest = manifest_from_rows({}, synthetic_rows())
    assert len(manifest["records"]) == 160
    assert manifest["split_counts"] == {"train": 96, "validation": 32, "test": 32}
    assert len({r["duplicate_cluster_sha256"] for r in manifest["records"]}) == 160
    assert all("question" not in r for r in manifest["records"] if r["split"] == "test")
    assert len(manifest["fold_assignment"]) == 96
    assert set(manifest["fold_assignment"].values()) == set(range(5))
    assert manifest == manifest_from_rows({}, synthetic_rows())


def test_production_trust_anchor_rejects_synthetic_cohort():
    manifest = manifest_from_rows({}, synthetic_rows())
    with pytest.raises(ValueError, match="independently trusted"):
        _verify_original_membership(manifest["records"], manifest["fold_assignment"])


def test_answer_validation_after_membership_no_replacement():
    rows = synthetic_rows(80)
    rows[0]["final_answer"] = None
    with pytest.raises(ValueError, match="replacement_count=0"):
        manifest_from_rows({}, rows)
    rows = synthetic_rows(90)
    ids = {r["source_index"] for r in manifest_from_rows({}, rows)["records"]}
    nonmember = next(i for i in range(len(rows)) if i not in ids)
    rows[nonmember]["final_answer"] = None
    assert len(manifest_from_rows({}, rows)["records"]) == 160


def test_duplicate_control_is_global():
    rows = synthetic_rows(81)
    rows[81]["question"] = rows[0]["question"]
    manifest = manifest_from_rows({}, rows)
    assert len({r["duplicate_cluster_sha256"] for r in manifest["records"]}) == 160


def test_scientific_config_drift_rejected():
    with pytest.raises(ValueError, match="Frozen generation"):
        manifest_from_rows({"generation": {"max_new_tokens": 8192}}, synthetic_rows())
    with pytest.raises(ValueError):
        manifest_from_rows({"splits": {"generate": ["test"]}}, synthetic_rows())
    with pytest.raises(ValueError, match="Unknown generation"):
        manifest_from_rows({"generation": {"reasoning_max_new_tokens": 4096}}, synthetic_rows())
    with pytest.raises(ValueError, match="Frozen generation"):
        manifest_from_rows({"generation": {"rollout_indices": [False, True, 2, 3, 4, 5]}}, synthetic_rows())


def test_plan_no_gpu_and_no_test():
    manifest = manifest_from_rows({}, synthetic_rows())
    plan = generation_plan({}, manifest)
    assert plan["planned_model_operations_maximum"] == 3840
    assert plan["rollouts"] == 768 and plan["test_requests"] == 0
    assert plan["splits"] == {"train": 576, "validation": 192}
    assert generation_plan({}, manifest, 2)["planned_model_operations_maximum"] == 10
    assert generation_plan({}, manifest, 2)["partial_diagnostic_not_full_study"]


def test_consent_fails_before_files_or_runtime(tmp_path):
    with patch("thinking_completion.generation._load_runtime") as load:
        with pytest.raises(PermissionError):
            generate({}, tmp_path / "missing", tmp_path / "not-created")
        load.assert_not_called()
    assert not (tmp_path / "not-created").exists()


def test_uncertain_previous_directory_is_never_resumed(tmp_path):
    manifest_path, _ = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    output.mkdir()
    (output / "unresolved-intent.json").write_text("{}")
    with patch("thinking_completion.generation._load_runtime") as load:
        with pytest.raises(FileExistsError):
            generate({}, manifest_path, output, True, 1)
        load.assert_not_called()


def test_private_raw_output_namespace_enforced(tmp_path):
    with pytest.raises(ValueError, match="artifacts/"):
        prepare_manifest({}, tmp_path / "unsafe-public-manifest.json")
    with pytest.raises(ValueError, match="artifacts/"):
        _private(tmp_path / "artifacts")  # A FILE here would bypass a directory-only ignore rule.


def test_historical_or_external_artifacts_root_rejected(tmp_path):
    with pytest.raises(ValueError, match="this publication checkout"):
        _private(tmp_path.parent / "historical-repository" / "artifacts" / "unsafe")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "reasoning-recovery-probes"\n')
    with pytest.raises(ValueError, match="checkout root"):
        _private(tmp_path / "artifacts" / "unsafe")


def test_artifact_ignore_policy_required(tmp_path):
    (tmp_path / ".gitignore").write_text("*.pt\n")
    with pytest.raises(ValueError, match="must be ignored"):
        _private(tmp_path / "artifacts" / "unsafe")
    (tmp_path / ".gitignore").write_text("artifacts/\n!artifacts/unsafe\n")
    with pytest.raises(ValueError, match="without ignore negations"):
        _private(tmp_path / "artifacts" / "unsafe")


@pytest.mark.parametrize("target_inside", [False, True])
def test_resolved_artifacts_symlink_cannot_escape_checkout(tmp_path, target_inside):
    external = (tmp_path / "public-folder") if target_inside else tmp_path.parent / (tmp_path.name + "-external-private")
    external.mkdir()
    try:
        (tmp_path / "artifacts").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("This host does not permit synthetic directory symlinks")
    with pytest.raises(ValueError, match="this publication checkout"):
        _private(tmp_path / "artifacts" / "unsafe")


def test_config_file_matches_frozen_canonical_contract():
    from thinking_completion.generation import _config
    config = json.loads((Path(__file__).parents[1] / "configs/regeneration_v1.json").read_text())
    normalized = _config(config)
    assert {k: v for k, v in normalized.items() if k != "note"} == DEFAULT_CONFIG


@requires_torch
def test_plan_exists_before_runtime_and_write_once(tmp_path):
    manifest_path, _ = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    runtime = FakeRuntime()
    def load(config):
        assert (output / "plan.json").exists()
        assert not (output / "receipts").exists()
        return runtime
    with patch("thinking_completion.generation._load_runtime", side_effect=load):
        summary = generate({}, manifest_path, output, True, 1)
    assert summary["status"] == "COLLECTION_COMPLETE"
    assert len(runtime.readout_inputs) == 3
    assert len(runtime.activation_inputs[0]) == 514
    assert runtime.activation_inputs[0][-1] == 3  # Not a close cue.
    assert all(ids[-2:] == [99, 100] for ids in runtime.readout_inputs)
    with pytest.raises(FileExistsError):
        generate({}, manifest_path, output, True, 1)


def test_failure_preserved_no_retry_or_replacement(tmp_path):
    manifest_path, _ = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    with patch("thinking_completion.generation._load_runtime", return_value=FakeRuntime(fail=True)):
        summary = generate({}, manifest_path, output, True, 2)
    assert summary["status"] == "FAILED_PRESERVED"
    assert summary["operations"]["base_failed"] == 1
    assert len(list((output / "intents").iterdir())) == 1
    assert json.loads((output / "receipts/row-0000-base.json").read_bytes())["status"] == "FAILED"
    with pytest.raises(FileExistsError):
        generate({}, manifest_path, output, True, 2)


def test_early_closure_excluded_without_readouts(tmp_path):
    manifest_path, _ = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    runtime = FakeRuntime(early=True)
    with patch("thinking_completion.generation._load_runtime", return_value=runtime):
        generate({}, manifest_path, output, True, 1)
    assert runtime.readout_inputs == [] and runtime.activation_inputs == []
    assert json.loads((output / "receipts/row-0000-activation.json").read_bytes())["status"] == "UNAVAILABLE"


@requires_torch
def test_authenticated_cpu_feature_schema(tmp_path):
    manifest_path, manifest = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    with patch("thinking_completion.generation._load_runtime", return_value=FakeRuntime()):
        generate({}, manifest_path, output, True, 1)
    features = tmp_path / "artifacts" / "features"
    report = extract_features(output, features)
    split = next(r["split"] for r in manifest["records"] if r["split"] != "test")
    with np.load(features / (split + ".npz"), allow_pickle=False) as data:
        assert data["numeric"].shape == (1, 11)
        assert data["lexical"].shape == (1, 512)
        assert data["activations"].shape == (1, 28, 2048)
        assert data["activations"].dtype == np.float32
        assert data["labels"].tolist() == [1]
        assert data["numeric"][0, -2:].tolist() == [1, 1]
        assert data["lexical"].sum() == 128
    assert report["test_rows"] == 0 and report["gpu_operations"] == 0
    with pytest.raises(FileExistsError):
        extract_features(output, features)


@requires_torch
def test_tamper_before_parsing_is_rejected(tmp_path):
    manifest_path, _ = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    with patch("thinking_completion.generation._load_runtime", return_value=FakeRuntime()):
        generate({}, manifest_path, output, True, 1)
    (output / "receipts/row-0000-base.json").write_bytes(b"not json")
    with pytest.raises(ValueError, match="hash/size"):
        extract_features(output, tmp_path / "artifacts" / "features")


def test_boundary_target_exact_coordinate_and_eos():
    close = reasoning_boundary([3] * 512 + [99], [99], [98], "THINK_CLOSE_MARKER")
    assert completion_target(close) == (1, "CLOSE_WITHIN_4096")
    assert completion_target(reasoning_boundary([3] * 511 + [99], [99], [98], "THINK_CLOSE_MARKER"))[0] is None
    assert completion_target(reasoning_boundary([3] * 4096, [99], [98], "MAX_NEW_TOKENS"))[0] == 0
    assert completion_target(reasoning_boundary([3] * 4095 + [98], [99], [98], "EOS"))[0] is None


def test_last_balanced_box_and_boundary_token_exclusion():
    assert boxed_boundary(r"\boxed{one} \boxed{broken")["answer"] == r"\boxed{one}"
    assert boxed_boundary(r"\boxed{outer \boxed{inner}}")["answer"] == r"\boxed{inner}"
    assert boxed_boundary(r"\boxedness{one}")["status"] == "MISSING_BOXED_ANSWER"
    pieces = [r"\boxed{", "12", "}", "EOS"]
    result = reduce_readout([0, 1, 2, 3], [-0.1, -0.2, -0.3, -0.4], 1, 0.5,
                           lambda ids: "".join(pieces[i] for i in ids))
    assert result["answer_token_count"] == 1
    assert result["mean_answer_token_logprob"] == -0.2


def test_missing_answer_breaks_immediate_prior_agreement():
    readouts = [{"position": p, "status": "COMPLETED", "scores": {"normalized_answer": answer}}
                for p, answer in [(128, "a"), (256, None), (512, "a")]]
    assert numeric_features({"difficulty": 7}, 12, readouts)[-2:] == [0, 0]
    readouts[1]["status"] = "UNAVAILABLE"
    assert numeric_features({"difficulty": 7}, 12, readouts)[-2:] == [1, 1]


def test_seed_and_lexical_determinism():
    assert rollout_seed("synthetic-id", 0) == rollout_seed("synthetic-id", 0)
    assert rollout_seed("synthetic-id", 0) != rollout_seed("synthetic-id", 1)
    assert 1 <= rollout_seed("synthetic-id", 0) <= 2147483646
    assert sum(lexical_counts([3] * 512)) == 128


def test_nonfinite_scalar_conversion_has_explicit_field_audit():
    converted, fields = encode_nonfinite_scalars({
        "mean_answer_token_logprob": float("inf"),
        "first_readout_token_entropy_nats": float("nan"),
        "answer_token_count": 4, "normalized_answer": "fixture"})
    assert fields == ["mean_answer_token_logprob", "first_readout_token_entropy_nats"]
    assert converted["mean_answer_token_logprob"] is None
    assert converted["first_readout_token_entropy_nats"] is None
    assert converted["answer_token_count"] == 4
    counts = {}
    rows = [{"position": 512, "status": "COMPLETED", "scores": {
        "mean_answer_token_logprob": float("inf"), "min_answer_token_logprob": -1}}]
    numeric = numeric_features({"difficulty": 7}, 12, rows, nonfinite_counts=counts)
    assert numeric[4] is None and numeric[5] == -1
    assert counts == {"mean_answer_token_logprob": 1}


@requires_torch
def test_derivative_aggregates_nonfinite_conversion_receipts(tmp_path):
    class MissingEntropyFixture(FakeRuntime):
        def readout(self, ids):
            result = super().readout(ids)
            result["scores"]["first_readout_token_entropy_nats"] = None
            result["nonfinite_scalar_fields_encoded_missing"] = ["first_readout_token_entropy_nats"]
            result["nonfinite_scalar_count_encoded_missing"] = 1
            return result
    manifest_path, _ = save_manifest(tmp_path)
    output = tmp_path / "artifacts" / "generation"
    with patch("thinking_completion.generation._load_runtime", return_value=MissingEntropyFixture()):
        generate({}, manifest_path, output, True, 1)
    report = extract_features(output, tmp_path / "artifacts" / "features")
    assert report["nonfinite_readout_scalars_encoded_missing"] == {
        "first_readout_token_entropy_nats": 3}
    assert report["nonfinite_primary_feature_values_encoded_missing"] == {}
