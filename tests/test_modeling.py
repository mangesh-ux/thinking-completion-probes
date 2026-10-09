"""Small synthetic fits and mocked orchestration, never historical refitting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from thinking_completion import modeling


def fixture(groups=20, rows=6, split="train"):
    n = groups * rows
    group_index = np.repeat(np.arange(groups), rows)
    labels = np.tile(np.arange(rows) % 2, groups).astype(np.int8)
    numeric = np.zeros((n, 11), dtype=np.float64)
    numeric[:, 0], numeric[:, 1] = 512, 0.125
    numeric[:, 2] = 20 + group_index
    numeric[:, 3] = 7 + group_index % 2
    numeric[:, 4] = np.nan
    numeric[:, 7] = np.arange(n) / n
    lexical = np.zeros((n, 512), dtype=np.float64)
    lexical[:, 0] = 128 - np.arange(n) % 3
    lexical[:, 1] = np.arange(n) % 3
    activations = np.zeros((n, 28, 2048), dtype=np.float32)
    activations[:, :, 0] = (np.arange(n) % 4)[:, None]
    return {
        "numeric": numeric, "topics": np.array(["algebra" if g % 2 else "geometry" for g in group_index]),
        "groups": np.array([f"{split}-g{g:03d}" for g in group_index]),
        "lexical": lexical, "activations": activations, "labels": labels,
        "row_ids": np.array([f"{split}-r{i:03d}" for i in range(n)]),
        "fold_ids": (group_index % 5).astype(np.int8) if split == "train" else np.full(n, -1, dtype=np.int8),
        "difficulty": numeric[:, 3].copy(), "numeric_feature_names": np.asarray(modeling.FEATURE_NAMES),
    }


def constant_state(model, strength, layer, data):
    return {"model": model, "layer": layer, "C": strength,
            "numeric_medians": [0.0] * (4 if model == "A" else 11),
            "topic_vocabulary": sorted(set(data["topics"])), "kept_columns": [0],
            "means": [512.0], "scales": [1.0], "coefficients": [0.0], "intercept": 0.0}


def mocked_fit(train, evaluation, model, strength, layer=None):
    # Orchestration test only: final TRAIN restore agrees with a constant model.
    p = np.full(len(evaluation["labels"]), 0.5)
    if train is not evaluation:
        quality = 0.1 / (1 + strength)
        if model == "Clex":
            quality += 0.1 / (1 + abs(layer - 15))
        p += (2 * evaluation["labels"] - 1) * quality
    return p, constant_state(model, strength, layer, train)


class ModelingTests(unittest.TestCase):
    def test_support_and_stored_folds(self):
        data = fixture()
        modeling.validate_dataset(data, "train")
        changed = copy.deepcopy(data)
        changed["fold_ids"][0] = 4
        with self.assertRaisesRegex(ValueError, "crosses"):
            modeling.validate_dataset(changed, "train")
        with self.assertRaisesRegex(ValueError, "diversity"):
            modeling.validate_dataset(fixture(groups=10), "train")

    def test_feature_whitelist_and_future_metadata_fail_closed(self):
        data = fixture(groups=3)
        with self.assertRaisesRegex(ValueError, "metadata"):
            modeling.Preprocessor("B").fit({**data, "future_terminal_status": np.zeros(18)})
        changed = copy.deepcopy(data)
        changed["numeric_feature_names"] = np.array(list(modeling.FEATURE_NAMES[:-1]) + ["future_outcome"])
        with self.assertRaisesRegex(ValueError, "whitelist"):
            modeling.Preprocessor("B").fit(changed)

    def test_nonfinite_and_duplicate_input_rejected(self):
        data = fixture()
        for field in ("numeric", "lexical", "activations"):
            changed = copy.deepcopy(data)
            changed[field].flat[0] = np.inf
            with self.subTest(field=field), self.assertRaises(ValueError):
                modeling.validate_dataset(changed, "train")
        changed = copy.deepcopy(data)
        changed["row_ids"][0] = changed["row_ids"][1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            modeling.validate_dataset(changed, "train")

    def test_metadata_is_not_row_sliced(self):
        data = fixture(groups=3)
        sliced = modeling.subset(data, np.arange(18) < 4)
        self.assertEqual(sliced["numeric"].shape, (4, 11))
        self.assertEqual(sliced["numeric_feature_names"].shape, (11,))

    def test_all_missing_zero_imputation_and_unknown_topics(self):
        train = fixture(groups=3)
        prep = modeling.Preprocessor("B").fit(train)
        self.assertEqual(prep.medians[4], 0.0)
        self.assertFalse(prep.keep[4])
        self.assertFalse(prep.keep[11 + 4])
        heldout = copy.deepcopy(train)
        heldout["topics"] = np.full(18, "unseen")
        width = len(prep.topics.categories_[0])
        self.assertTrue((prep.unscaled(heldout)[:, 22:22 + width] == 0).all())
        self.assertTrue(np.isfinite(prep.transform(heldout)).all())

    def test_imputation_and_scaling_use_train_only(self):
        train = fixture(groups=3)
        train["numeric"][:, 5] = np.arange(18)
        train["numeric"][0, 5] = np.nan
        prep = modeling.Preprocessor("B").fit(train)
        original_mean, original_scale = prep.mean.copy(), prep.scale.copy()
        heldout = copy.deepcopy(train)
        heldout["numeric"][:, 5] = 1e9
        prep.transform(heldout)
        np.testing.assert_array_equal(prep.mean, original_mean)
        np.testing.assert_array_equal(prep.scale, original_scale)
        self.assertEqual(prep.medians[5], 9.0)

    def test_only_selected_activation_layer_enters_predictors(self):
        train = fixture(groups=3)
        prep = modeling.Preprocessor("Clex", 15).fit(train)
        changed = copy.deepcopy(train)
        changed["activations"][:, 14, :] = 999
        np.testing.assert_array_equal(prep.transform(changed), prep.transform(train))

    def test_small_real_fit_restores_without_any_fitting(self):
        train = fixture(groups=4)
        for model, layer in (("A", None), ("B", None), ("Blex", None), ("C", 15), ("Clex", 15)):
            p, state = modeling.fit_predict(train, train, model, 0.001, layer)
            with patch.object(modeling.LogisticRegression, "fit", side_effect=AssertionError("refit forbidden")), patch.object(modeling.OneHotEncoder, "fit", side_effect=AssertionError("vocabulary refit forbidden")):
                np.testing.assert_allclose(modeling.predict_frozen(train, state), p, rtol=1e-10, atol=1e-12)
            changed = copy.deepcopy(train)
            changed["labels"] = 1 - changed["labels"]
            np.testing.assert_array_equal(modeling.predict_frozen(changed, state), modeling.predict_frozen(train, state))

    def test_invalid_saved_states_are_rejected(self):
        data = fixture(groups=3)
        state = constant_state("B", 0.001, None, data)
        for key, value in (("scales", [0.0]), ("means", [np.inf]), ("kept_columns", [9999]), ("coefficients", [np.nan]), ("layer", 15)):
            changed = {**state, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                modeling.predict_frozen(data, changed)

    def test_convergence_warning_is_not_silently_accepted(self):
        data = fixture(groups=3)
        def warning_fit(*args, **kwargs):
            import warnings
            warnings.warn("synthetic", ConvergenceWarning)
        with patch.object(modeling.LogisticRegression, "fit", side_effect=warning_fit), self.assertRaises(ConvergenceWarning):
            modeling.fit_predict(data, data, "B", 0.001)

    def test_fingerprint_changes_when_stored_folds_change(self):
        data = fixture()
        changed = copy.deepcopy(data)
        changed["fold_ids"] = (changed["fold_ids"] + 1) % 5
        self.assertNotEqual(modeling.dataset_fingerprint(data), modeling.dataset_fingerprint(changed))

    def test_all_128_configs_and_models_are_frozen_before_evaluate(self):
        train, validation = fixture(), fixture(groups=10, split="validation")
        calls = [0]
        def counted_fit(*args, **kwargs):
            calls[0] += 1
            return mocked_fit(*args, **kwargs)
        # A plain replacement does not retain 645 full array call arguments.
        with tempfile.TemporaryDirectory() as temporary, patch.object(modeling, "fit_predict", new=counted_fit):
            selection_directory = Path(temporary) / "selection"
            selected = modeling.select(train, selection_directory)
            self.assertEqual(calls[0], 645)
            self.assertEqual(len(selected["cv_scores"]), 128)
            self.assertEqual(len(list(selection_directory.glob("cv-config-*.json"))), 124)
            self.assertEqual(len(list(selection_directory.glob("cv-secondary-C-*.json"))), 4)
            self.assertEqual(selected["selected"]["Clex"]["layer"], 15)
            self.assertEqual(selected["selected"]["C"]["layer"], 15)
            self.assertTrue(all(record["C"] == 0.001 for record in selected["selected"].values()))
            self.assertFalse(selected["validation_loaded"])
            self.assertEqual(selected["fitted_models_sha256"], modeling._sha_file(selection_directory / "fitted-models.json"))
            lock = json.loads((selection_directory / "selection-lock.json").read_text())
            self.assertEqual(lock["selection_sha256"], modeling._sha_file(selection_directory / "selection.json"))
            # Selection reads stored assignments, not a hash of these names.
            self.assertEqual(selected["fold_assignment"]["train-g000"], 0)
            with patch.object(modeling, "fit_predict", side_effect=AssertionError("heldout refit forbidden")), patch.object(modeling.LogisticRegression, "fit", side_effect=AssertionError("heldout estimator fit forbidden")):
                output = Path(temporary) / "evaluation"
                result = modeling.evaluate(train, validation, selection_directory, output)
            self.assertTrue(result["no_validation_refit"])
            self.assertEqual(result["rows"], 60)
            self.assertEqual(result["matched_control"]["matched_problem_count"], 10)
            self.assertTrue((output / "heldout-predictions.npz").is_file())
            self.assertTrue((selection_directory / "evaluation-consumed.json").is_file())
            with self.assertRaises(FileExistsError):
                modeling.select(train, selection_directory)
            with self.assertRaises(FileExistsError):
                modeling.evaluate(train, validation, selection_directory, output)
            with self.assertRaises(FileExistsError):
                modeling.evaluate(train, validation, selection_directory, Path(temporary) / "repeat-evaluation")

    def test_failed_candidate_retained_and_no_selection_published(self):
        def failing_fit(*args, **kwargs):
            if args[2:4] == ("A", 0.01):
                raise ValueError("synthetic bad configuration")
            return mocked_fit(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, patch.object(modeling, "fit_predict", new=failing_fit):
            output = Path(temporary) / "selection"
            with self.assertRaisesRegex(ValueError, "configuration failed"):
                modeling.select(fixture(), output)
            self.assertEqual(len(list(output.glob("cv-config-*.json"))), 124)
            self.assertFalse((output / "selection.json").exists())
            self.assertTrue((output / "failure.json").exists())
            self.assertEqual(json.loads((output / "cv-config-002.json").read_text())["error_type"], "ValueError")

    def test_heldout_overlap_and_support_gate_fail_before_predictions(self):
        train = fixture()
        with tempfile.TemporaryDirectory() as temporary, patch.object(modeling, "fit_predict", new=mocked_fit):
            bad = fixture(groups=10, split="validation")
            bad["groups"][bad["groups"] == "validation-g000"] = "train-g000"
            selected = Path(temporary) / "selection-overlap"
            modeling.select(train, selected)
            with patch.object(modeling, "predict_frozen", side_effect=AssertionError("prediction before support gate")), self.assertRaisesRegex(ValueError, "leakage"):
                modeling.evaluate(train, bad, selected, Path(temporary) / "overlap")
            self.assertTrue((selected / "evaluation-consumed.json").is_file())
            self.assertTrue((Path(temporary) / "overlap" / "failure.json").is_file())
            selected = Path(temporary) / "selection-support"
            modeling.select(train, selected)
            with patch.object(modeling, "predict_frozen", side_effect=AssertionError("prediction before support gate")), self.assertRaisesRegex(ValueError, "diversity"):
                modeling.evaluate(train, fixture(groups=7, split="validation"), selected, Path(temporary) / "support")


if __name__ == "__main__":
    unittest.main()
