"""Synthetic publication tests; never read historical scientific inputs."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from thinking_completion import data


def fixture(split: str) -> dict[str, np.ndarray]:
    group_count = 5 if split == "train" else 2
    n = group_count * 2
    groups = np.asarray([f"private-{split}-problem-{i // 2:03d}" for i in range(n)])
    difficulty = np.asarray([7.0 + (i // 2) % 2 for i in range(n)], dtype=np.float64)
    numeric = np.zeros((n, 11), dtype=np.float64)
    numeric[:, :4] = np.column_stack((np.full(n, 512), np.full(n, 0.125),
                                     100 + np.arange(n) // 2, difficulty))
    numeric[:, 4] = np.linspace(-1, -2, n)
    numeric[:, 5] = np.nan
    # Include signed zero and a precisely representable, nontrivial float.
    numeric[0, 4] = -0.0
    numeric[-1, 4] = np.nextafter(np.float64(-1.0), np.float64(-2.0))
    lexical = np.zeros((n, 512), dtype=np.float64)
    lexical[:, 17] = 128
    activations = np.zeros((n, 28, 2048), dtype=np.float32)
    activations[:, 27, 2047] = np.arange(n) / np.float32(8)
    activations[0, 0, 0] = np.float32(-0.0)
    return {
        "numeric": numeric,
        "topics": np.asarray(["synthetic-algebra" if d == 7 else "synthetic-geometry" for d in difficulty]),
        "groups": groups,
        "lexical": lexical,
        "activations": activations,
        "labels": np.asarray(np.arange(n) % 2, dtype=np.int8),
        "row_ids": np.asarray([f"private-{split}-rollout-{i:04d}" for i in range(n)]),
        "fold_ids": np.asarray(np.arange(n) // 2 if split == "train" else np.full(n, -1), dtype=np.int8),
        "difficulty": difficulty,
        "numeric_feature_names": np.asarray(data.FEATURE_NAMES),
    }


def models(train: dict[str, np.ndarray]) -> dict:
    vocabulary = sorted(set(train["topics"].tolist()))
    result = {}
    for name in data.MODEL_NAMES:
        result[name] = {
            "model": name, "layer": 15 if name in ("C", "Clex") else None,
            "C": 0.001, "numeric_medians": [0.0] * (4 if name == "A" else 11),
            "topic_vocabulary": vocabulary, "kept_columns": [2],
            "means": [0.25], "scales": [1.0], "coefficients": [0.5], "intercept": -0.1,
        }
    return {"models": result, "train_problem_prevalence": 0.5}


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = data.REPOSITORY / "data" / "synthetic-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)

    def source(self) -> tuple[Path, dict[str, str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        source = self.root / "synthetic-source"
        train, validation = fixture("train"), fixture("validation")
        self.save(source / data.SOURCE_FILES["train_data"], train)
        self.save(source / data.SOURCE_FILES["validation_data"], validation)
        model_path = source / data.SOURCE_FILES["fitted_models"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(models(train)), encoding="utf-8")
        predictions = {key: validation[key] for key in ("labels", "groups", "row_ids")}
        predictions.update({name: np.linspace(0.1, 0.9, len(validation["labels"]), dtype=np.float64)
                            for name in data.MODEL_NAMES})
        self.save(source / data.SOURCE_FILES["private_predictions"], predictions)
        pins = {key: data.sha256_file(source / name) for key, name in data.SOURCE_FILES.items()
                if key != "public_report"}
        report_path = source / data.SOURCE_FILES["public_report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"evidence_sha256": pins}), encoding="utf-8")
        pins["public_report"] = data.sha256_file(report_path)
        return source, pins, train, validation

    def test_loader_preserves_float_bits_and_all_layers(self) -> None:
        expected = fixture("train")
        path = self.root / "numeric.npz"
        self.save(path, expected)
        result = data.load_dataset(path, data.sha256_file(path))
        for key in expected:
            self.assertEqual(result[key].dtype, expected[key].dtype)
            self.assertEqual(result[key].tobytes(), expected[key].tobytes())
        self.assertEqual(result["activations"].shape[1:], (28, 2048))

    def test_hash_corruption_is_detected_before_npz(self) -> None:
        path = self.root / "numeric.npz"
        self.save(path, fixture("train"))
        expected = data.sha256_file(path)
        with path.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(data.BundleError, "SHA256"):
            data.load_dataset(path, expected)

    def test_object_arrays_and_extra_metadata_are_forbidden(self) -> None:
        for kind in ("object", "extra"):
            with self.subTest(kind=kind):
                arrays = fixture("train")
                if kind == "object":
                    arrays["topics"] = arrays["topics"].astype(object)
                else:
                    arrays["raw_answer"] = np.asarray(["private-not-allowed"] * len(arrays["labels"]))
                path = self.root / f"{kind}.npz"
                self.save(path, arrays)
                with self.assertRaises(data.BundleError):
                    data.load_dataset(path)

    def test_npz_traversal_member_is_rejected_without_extraction(self) -> None:
        path = self.root / "malicious.npz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../labels.npy", b"not-array")
        with self.assertRaisesRegex(data.BundleError, "schema"):
            data.load_dataset(path)
        self.assertFalse((self.root.parent / "labels.npy").exists())

    def test_malicious_npy_shape_is_rejected_before_array_allocation(self) -> None:
        original = self.root / "normal.npz"
        self.save(original, fixture("train"))
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": "|i1", "fortran_order": False, "shape": (10 ** 12,),
        })
        malicious = self.root / "huge-shape.npz"
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(malicious, "w") as dest:
            for name in source.namelist():
                dest.writestr(name, header.getvalue() if name == "labels.npy" else source.read(name))
        with patch.object(data.np, "load", side_effect=AssertionError("allocation must not begin")):
            with self.assertRaisesRegex(data.BundleError, "shape and payload"):
                data.load_dataset(malicious)

    def test_duplicate_rows_crossing_folds_and_lossy_dtypes_fail(self) -> None:
        for problem in ("rows", "folds", "dtype", "nonfinite"):
            with self.subTest(problem=problem):
                arrays = fixture("train")
                if problem == "rows":
                    arrays["row_ids"][1] = arrays["row_ids"][0]
                elif problem == "folds":
                    arrays["fold_ids"][1] = 4
                elif problem == "dtype":
                    arrays["numeric"] = arrays["numeric"].astype(np.float32)
                else:
                    arrays["activations"][0, 0, 1] = np.inf
                path = self.root / f"{problem}.npz"
                self.save(path, arrays)
                with self.assertRaises(data.BundleError):
                    data.load_dataset(path)

    def test_export_is_lossless_anonymous_paired_and_write_once(self) -> None:
        source, pins, train, validation = self.source()
        destination = self.root / "bundle"
        with patch.dict(data.SOURCE_SHA256, pins, clear=True):
            manifest = data.export_public_bundle(source, destination)
            summary = data.verify_bundle(destination)
            self.assertEqual(summary["status"], "VERIFIED")
            self.assertEqual(manifest["support"]["train"]["rows"], 10)
            public_train = data.load_dataset(destination / "train.npz")
            public_validation = data.load_dataset(destination / "validation.npz")
            for original, exported in ((train, public_train), (validation, public_validation)):
                for key in data.DATA_KEYS - {"groups", "row_ids"}:
                    self.assertEqual(original[key].dtype, exported[key].dtype)
                    self.assertEqual(original[key].tobytes(), exported[key].tobytes())
                self.assertEqual(np.argsort(original["groups"], kind="stable").tolist(),
                                 np.argsort(exported["groups"], kind="stable").tolist())
            self.assertEqual(public_train["row_ids"].tolist(), [f"train-r{i:04d}" for i in range(10)])
            self.assertEqual(np.unique(public_train["groups"]).tolist(), [f"train-g{i:03d}" for i in range(5)])
            with np.load(destination / "heldout-predictions.npz", allow_pickle=False) as p:
                np.testing.assert_array_equal(p["groups"], public_validation["groups"])
                np.testing.assert_array_equal(p["row_ids"], public_validation["row_ids"])
                np.testing.assert_array_equal(p["labels"], public_validation["labels"])
            for name in data.BUNDLE_FILES:
                self.assertNotIn(b"private-train-problem", (destination / name).read_bytes())
            for key, original_path in data.SOURCE_FILES.items():
                self.assertEqual(data.sha256_file(source / original_path), pins[key])
            with self.assertRaisesRegex(data.BundleError, "already exists"):
                data.export_public_bundle(source, destination)

    def test_destination_cannot_overlap_source_or_leave_new_data_namespace(self) -> None:
        source, pins, _, _ = self.source()
        with patch.dict(data.SOURCE_SHA256, pins, clear=True):
            with self.assertRaisesRegex(data.BundleError, "overlaps"):
                data.export_public_bundle(source, source / "bundle")
            with self.assertRaisesRegex(data.BundleError, "namespace"):
                data.export_public_bundle(source, data.REPOSITORY / "unsafe-public-output")

    def test_manifest_traversal_and_unlisted_files_fail(self) -> None:
        source, pins, _, _ = self.source()
        destination = self.root / "bundle"
        with patch.dict(data.SOURCE_SHA256, pins, clear=True):
            manifest = data.export_public_bundle(source, destination)
            (destination / "unexpected.txt").write_text("synthetic", encoding="utf-8")
            with self.assertRaisesRegex(data.BundleError, "allowlist"):
                data.verify_bundle(destination)
            (destination / "unexpected.txt").unlink()
            manifest["files"]["../escaped.json"] = {"bytes": 0, "sha256": "a" * 64}
            (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(data.BundleError, "Unsafe"):
                data.verify_bundle(destination)

    def test_predictions_wrong_row_order_fail(self) -> None:
        validation = fixture("validation")
        arrays = {key: validation[key].copy() for key in ("labels", "groups", "row_ids")}
        arrays.update({name: np.full(len(validation["labels"]), 0.5) for name in data.MODEL_NAMES})
        arrays["row_ids"] = arrays["row_ids"][::-1]
        path = self.root / "predictions.npz"
        self.save(path, arrays)
        with self.assertRaisesRegex(data.BundleError, "paired"):
            data._predictions(path, validation)

    def test_manifest_cannot_carry_unexpected_private_metadata(self) -> None:
        source, pins, _, _ = self.source()
        destination = self.root / "bundle"
        with patch.dict(data.SOURCE_SHA256, pins, clear=True):
            manifest = data.export_public_bundle(source, destination)
            manifest["raw_question"] = "synthetic-private-not-allowed"
            (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(data.BundleError, "manifest metadata"):
                data.verify_bundle(destination)

    def test_models_reject_private_metadata_and_invalid_probability(self) -> None:
        train = fixture("train")
        value = models(train)
        value["raw_question"] = "synthetic-private-not-allowed"
        path = self.root / "models.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(data.BundleError, "metadata"):
            data._models(path, train)
        value = models(train)
        value["models"]["C"]["topic_vocabulary"] = ["not-the-original-topic"]
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(data.BundleError, "vocabulary"):
            data._models(path, train)


if __name__ == "__main__":
    unittest.main()
