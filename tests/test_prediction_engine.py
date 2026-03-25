from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from field_data_store import data_quality_issues, import_pfz_observations, normalize_pfz_dataset
from juvenile_risk_utils import lookup_maturity_length, maturity_risk_label
from prediction_engine import predict_fishing_zone
from project_data_utils import balance_classification_frame, balanced_risk_labels
from validate_project import _class_metrics


class PredictionEngineTests(unittest.TestCase):
    def test_fishbase_lookup_returns_known_species_length(self) -> None:
        length, _ = lookup_maturity_length("Sardinella longiceps")
        self.assertIsNotNone(length)
        self.assertAlmostEqual(float(length), 16.3, places=1)

    def test_maturity_risk_high_for_subadult_length(self) -> None:
        label, score = maturity_risk_label(12.0, 16.3)
        self.assertEqual(label, "High")
        self.assertIsNotNone(score)
        self.assertGreater(float(score), 0.2)

    def test_prediction_uses_exact_maturity_rule_when_species_and_length_are_available(self) -> None:
        result = predict_fishing_zone(
            location="Vizag",
            sst=28.0,
            salinity=34.0,
            dissolved_oxygen=6.2,
            historical_catch=250.0,
            species="Sardinella longiceps",
            observed_length_cm=12.0,
            model_choice="random_forest",
        )
        self.assertEqual(result.juvenile_risk, "High")
        self.assertIn("Exact maturity rule", result.juvenile_method)
        self.assertAlmostEqual(float(result.maturity_length_cm), 16.3, places=1)

    def test_prediction_falls_back_without_species_or_lengths(self) -> None:
        result = predict_fishing_zone(
            location="Vizag",
            sst=28.0,
            salinity=34.0,
            dissolved_oxygen=6.2,
            historical_catch=250.0,
            model_choice="random_forest",
        )
        self.assertEqual(result.juvenile_method, "Environmental juvenile model fallback")

    def test_all_pipeline_choices_execute_without_error(self) -> None:
        for model_choice in ["random_forest", "xgboost", "hybrid"]:
            with self.subTest(model_choice=model_choice):
                result = predict_fishing_zone(
                    location="Vizag",
                    sst=28.0,
                    salinity=34.0,
                    dissolved_oxygen=6.2,
                    historical_catch=250.0,
                    species="Sardinella longiceps",
                    observed_length_cm=12.0,
                    latitude=17.6868,
                    longitude=83.2185,
                    model_choice=model_choice,
                )
                self.assertIn(result.juvenile_risk, {"High", "Medium", "Low"})
                self.assertIsInstance(result.quantity, float)
                self.assertTrue(result.model_pipeline)

    def test_imported_observation_rows_are_normalized_and_exact_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample.csv"
            pd.DataFrame(
                [
                    {
                        "State": "Andhra Pradesh",
                        "Date": "2024-01-15",
                        "PFZ_Count": 1,
                        "Distance_km": 18.5,
                        "Depth_m": 55,
                        "Species": "Sardinella longiceps",
                        "Observed_Length_cm": 12.0,
                        "Maturity_Length_cm": 16.3,
                    }
                ]
            ).to_csv(sample_path, index=False)

            frame = normalize_pfz_dataset(pd.read_csv(sample_path))
            issues = data_quality_issues(frame)
            self.assertTrue(all("No exact-ready rows" not in issue for issue in issues))

    def test_class_metrics_confusion_matrix_tracks_numeric_labels(self) -> None:
        metrics = _class_metrics(pd.Series([0, 1, 1, 0]), pd.Series([0, 1, 0, 0]))
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"]["0"]["0"], 2)
        self.assertEqual(metrics["confusion_matrix"]["1"]["0"], 1)
        self.assertEqual(metrics["confusion_matrix"]["1"]["1"], 1)

    def test_balanced_risk_labels_produce_three_classes_for_ranked_scores(self) -> None:
        labels = balanced_risk_labels(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        self.assertEqual(set(labels.tolist()), {"Low", "Medium", "High"})

    def test_balance_classification_frame_upsamples_minority_class(self) -> None:
        features = pd.DataFrame({"value": [1, 2, 3, 4]})
        target = pd.Series(["High", "Low", "Low", "Low"], name="Juvenile_Risk")
        balanced_features, balanced_target = balance_classification_frame(features, target, random_state=42)
        counts = balanced_target.value_counts().to_dict()
        self.assertEqual(counts["High"], counts["Low"])
        self.assertEqual(len(balanced_features), len(balanced_target))


if __name__ == "__main__":
    unittest.main()
