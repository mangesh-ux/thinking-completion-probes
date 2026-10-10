"""Mocked delivery checks only: no network, original data, fitting or GPU work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from thinking_completion import cli, data


SPEC = importlib.util.spec_from_file_location(
    "huggingface_cpu_example",
    Path(__file__).resolve().parents[1] / "docs/huggingface/cpu_reproduce.py",
)
assert SPEC is not None and SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def mocked_delivery(monkeypatch):
    assets = {name: b"synthetic-numeric-delivery-fixture" for name in EXAMPLE.PUBLIC_FILES}
    assets["manifest.json"] = b'{"synthetic":true}'
    manifest_sha = hashlib.sha256(assets["manifest.json"]).hexdigest()
    monkeypatch.setattr(EXAMPLE, "EXPECTED_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(EXAMPLE.importlib.metadata, "version", lambda _: EXAMPLE.HUB_VERSION)
    runtime = Mock()
    saved = Mock(return_value={"status": "PASS", "statistic_comparison": {"checks": 39}})
    refit = Mock(return_value={"status": "PASS", "cpu_cv_fits": 640})
    monkeypatch.setattr(cli, "check_cpu_runtime", runtime)
    monkeypatch.setattr(cli, "verify_saved", saved)
    monkeypatch.setattr(cli, "reproduce", refit)
    monkeypatch.setattr(data, "verify_bundle", Mock(return_value={"status": "VERIFIED"}))
    api = Mock()
    api.return_value.dataset_info.return_value = SimpleNamespace(sha="a" * 40)

    def download(**kwargs):
        staging = Path(kwargs["local_dir"])
        staging.mkdir()
        metadata = staging / ".cache/huggingface"
        metadata.mkdir(parents=True)
        (metadata / "synthetic.metadata").write_text("delivery metadata", encoding="utf-8")
        for name, content in assets.items():
            (staging / name).write_bytes(content)
        return str(staging)

    snapshot = Mock(side_effect=download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=api, snapshot_download=snapshot))
    return api, snapshot, saved, refit, assets


def test_default_anonymous_exact_files_saved_verification_only(tmp_path, monkeypatch):
    api, snapshot, saved, refit, assets = mocked_delivery(monkeypatch)
    output = tmp_path / "example"
    result = EXAMPLE.run(output)
    assert result["status"] == "PASS"
    assert result["requested_revision"] == "v1.0.0"
    assert result["resolved_revision"] == "a" * 40
    assert api.call_args.kwargs["token"] is False
    assert api.return_value.dataset_info.call_args.kwargs["token"] is False
    arguments = snapshot.call_args.kwargs
    assert arguments["repo_type"] == "dataset"
    assert arguments["revision"] == "a" * 40
    assert arguments["token"] is False
    assert set(arguments["allow_patterns"]) == set(EXAMPLE.PUBLIC_FILES)
    assert {path.name for path in (output / "bundle").iterdir()} == set(EXAMPLE.PUBLIC_FILES)
    assert (output / "_download/.cache/huggingface/synthetic.metadata").exists()
    for name, content in assets.items():
        assert (output / "bundle" / name).read_bytes() == content
    saved.assert_called_once_with(output / "bundle")
    refit.assert_not_called()
    assert not (output / "failure.json").exists()


def test_refit_is_explicit_and_uses_the_unchanged_cpu_entrypoint(tmp_path, monkeypatch):
    api, _, saved, refit, _ = mocked_delivery(monkeypatch)
    api.return_value.dataset_info.return_value = SimpleNamespace(sha="b" * 40)
    output = tmp_path / "refit-example"
    result = EXAMPLE.run(output, revision="b" * 40, refit=True)
    saved.assert_called_once()
    refit.assert_called_once_with(output / "bundle", output / "refit")
    assert result["refit"]["cpu_cv_fits"] == 640


def test_manifest_mismatch_preserves_failure_and_prevents_verification(tmp_path, monkeypatch):
    _, _, saved, refit, _ = mocked_delivery(monkeypatch)
    monkeypatch.setattr(EXAMPLE, "EXPECTED_MANIFEST_SHA256", "0" * 64)
    output = tmp_path / "failed-checksum"
    with pytest.raises(RuntimeError, match="manifest SHA256"):
        EXAMPLE.run(output)
    saved.assert_not_called()
    refit.assert_not_called()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "FAILED_PRESERVED"
    assert failure["automatic_retry"] is False
    assert (output / "_download/manifest.json").exists()
    with pytest.raises(FileExistsError):
        EXAMPLE.run(output)


def test_runtime_failure_is_visible_preserved_and_does_not_download(tmp_path, monkeypatch):
    _, snapshot, saved, _, _ = mocked_delivery(monkeypatch)
    monkeypatch.setattr(EXAMPLE.importlib.metadata, "version", lambda _: "not-pinned")
    output = tmp_path / "bad-runtime"
    with pytest.raises(RuntimeError, match="huggingface_hub==1.19.0"):
        EXAMPLE.run(output)
    snapshot.assert_not_called()
    saved.assert_not_called()
    assert (output / "intent.json").exists()
    assert (output / "failure.json").exists()


def test_existing_output_and_unpinned_revision_are_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        EXAMPLE.run(tmp_path)
    for revision in ("main", "../unsafe", "v2.0.0", "abc"):
        with pytest.raises(Exception):
            EXAMPLE._revision(revision)
