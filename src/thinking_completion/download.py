"""Bounded, checksum-pinned release delivery; no scientific generation."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.request import urlopen
import zipfile

MEMBERS = frozenset({"manifest.json", "train.npz", "validation.npz", "fitted-models.json", "heldout-predictions.npz"})
LIMIT = 200 * 1024 * 1024


def release_info() -> dict:
    return json.loads(files("thinking_completion").joinpath("reference/release_v1.json").read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_verified(archive: Path, destination: Path, expected_sha: str, manifest_sha: str) -> dict:
    """Verify the whole archive BEFORE allocating or extracting any member."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise FileExistsError("Download destination must be fresh")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (expected_sha, manifest_sha)):
        raise ValueError("Invalid release checksum")
    if archive.stat().st_size > LIMIT or file_sha(archive) != expected_sha:
        raise ValueError("Release archive checksum or size differs")
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or set(names) != MEMBERS:
            raise ValueError("Release member allowlist differs")
        if any(entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > LIMIT for entry in entries):
            raise ValueError("Unsafe release member")
        if sum(entry.file_size for entry in entries) > LIMIT:
            raise ValueError("Expanded release exceeds size limit")
        if hashlib.sha256(source.read("manifest.json")).hexdigest() != manifest_sha:
            raise ValueError("Release manifest checksum differs")
        destination.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            with source.open(entry) as src, (destination / entry.filename).open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    return {"status": "DELIVERED", "archive_sha256": expected_sha, "manifest_sha256": manifest_sha, "files": len(MEMBERS)}


def download_bundle(destination: Path) -> dict:
    """Download only the fixed GitHub numerical release; no model/dataset access."""
    if Path(destination).exists():
        raise FileExistsError("Download destination must be fresh")
    info = release_info()
    if not info["url"].startswith("https://github.com/mangesh-ux/thinking-completion-probes/releases/download/v1.0.0/"):
        raise ValueError("Unexpected release endpoint")
    with tempfile.TemporaryDirectory(prefix="tcp-release-") as temporary:
        archive = Path(temporary) / "bundle.zip"
        size = 0
        with urlopen(info["url"], timeout=60) as response, archive.open("xb") as stream:
            if not response.geturl().startswith("https://"):
                raise ValueError("Release transport is not HTTPS")
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > LIMIT:
                    raise ValueError("Download exceeds release size limit")
                stream.write(block)
        if size != info["bytes"]:
            raise ValueError("Release download byte count differs")
        return unpack_verified(archive, Path(destination), info["sha256"], info["manifest_sha256"])


def pack_bundle(bundle: Path, destination: Path) -> dict:
    """Maintainer delivery packaging of an already reviewed numerical bundle."""
    from .data import verify_bundle
    verified = verify_bundle(bundle)
    if destination.exists():
        raise FileExistsError("Release archive is write-once")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(MEMBERS):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 9, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (bundle / name).read_bytes())
    return {"url": "https://github.com/mangesh-ux/thinking-completion-probes/releases/download/v1.0.0/completion-forecasting-public-v1.zip", "sha256": file_sha(destination), "bytes": destination.stat().st_size, "manifest_sha256": verified["manifest_sha256"]}
