import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from thinking_completion.download import MEMBERS, unpack_verified
from thinking_completion.verification import verify_reference, reference_results, compare_results


def archive(tmp_path: Path, extra=None):
    path = tmp_path / "bundle.zip"
    manifest = b'{"fixture":true}'
    with zipfile.ZipFile(path, "w") as stream:
        for name in sorted(MEMBERS):
            stream.writestr(name, manifest if name == "manifest.json" else b"fixture")
        if extra:
            stream.writestr(extra, b"untrusted")
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(manifest).hexdigest()


def test_delivery_and_write_once(tmp_path):
    path, checksum, manifest = archive(tmp_path)
    output = tmp_path / "output"
    assert unpack_verified(path, output, checksum, manifest)["files"] == 5
    with pytest.raises(FileExistsError):
        unpack_verified(path, output, checksum, manifest)


def test_hash_before_extraction(tmp_path):
    path, _, manifest = archive(tmp_path)
    with pytest.raises(ValueError):
        unpack_verified(path, tmp_path / "output", "0" * 64, manifest)
    assert not (tmp_path / "output").exists()


def test_path_traversal(tmp_path):
    path, checksum, manifest = archive(tmp_path, "../secret")
    with pytest.raises(ValueError):
        unpack_verified(path, tmp_path / "output", checksum, manifest)
    assert not (tmp_path / "secret").exists()


def test_reference_scope_and_all_statistics():
    assert verify_reference()["original_recovery"] == "NON_EVALUABLE"
    result = reference_results()
    assert compare_results(result)["checks"] == 39
    result["metrics"]["A"]["problem_weighted_brier"] += 0.01
    assert compare_results(result)["status"] == "FAIL"


def test_reproduction_cannot_write_inside_bundle(tmp_path, monkeypatch):
    from thinking_completion import cli
    monkeypatch.setattr(cli, "check_cpu_runtime", lambda: None)
    with pytest.raises(ValueError, match="cannot modify"):
        cli.reproduce(tmp_path, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_cli_gpu_consent_is_not_implicit(tmp_path, capsys):
    from thinking_completion.cli import main
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "artifacts" / "generation"
    assert main(["generate", "--config", str(config), "--manifest", str(tmp_path / "absent.json"), "--output", str(output)]) == 2
    assert not output.exists()
    assert 'FAILED' in capsys.readouterr().err
