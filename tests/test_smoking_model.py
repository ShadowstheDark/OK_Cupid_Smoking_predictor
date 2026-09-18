"""
tests/test_smoking_model.py
===========================
Unit and integration tests for smoking_model.py ML pipeline and inference.
"""

import pytest
import numpy as np
import pandas as pd
from smoking_model import (
    prepare_training_data,
    encode_single_profile,
    train_model_suite,
    predict_single_profile,
    compute_threshold_curve,
    DRUG_OPTIONS,
    DRINK_OPTIONS,
    BODY_TYPE_OPTIONS,
)


@pytest.fixture(scope="module")
def model_suite():
    return train_model_suite()


class TestDataPreparation:
    """Tests for data preprocessing and encoding schemas."""

    def test_prepare_training_data_shapes(self):
        X, y, feature_names, median_age = prepare_training_data()
        assert len(X) == len(y)
        assert len(feature_names) == X.shape[1]
        assert median_age > 18 and median_age < 70
        assert set(y.unique()).issubset({0, 1})

    def test_target_smoker_base_rate(self):
        _, y, _, _ = prepare_training_data()
        base_rate = float(y.mean())
        # The true smoker rate in the 2012 extract is ~19.4%
        assert 0.17 <= base_rate <= 0.22, f"Unexpected smoker base rate: {base_rate}"

    def test_unspecified_drugs_encoded_explicitly(self):
        # Verify the "did not answer" bug fix: drugs_unspecified should be an explicit feature
        X, _, feature_names, _ = prepare_training_data()
        assert "drugs_unspecified" in feature_names

    def test_encode_single_profile_columns_match(self):
        _, _, feature_names, median_age = prepare_training_data()
        profile = {
            "age": 25,
            "drugs": "never",
            "drinks": "socially",
            "body_type": "fit",
        }
        encoded = encode_single_profile(profile, feature_names, median_age)
        assert list(encoded.columns) == feature_names
        assert encoded.shape == (1, len(feature_names))
        assert encoded["age"].values[0] == 25.0
        assert encoded["body_type_fit"].values[0] == 1.0


class TestModelTrainingAndMetrics:
    """Tests for ML benchmark training and evaluation."""

    def test_all_models_trained(self, model_suite):
        expected_models = [
            "Baseline (Always 'No')",
            "Logistic Regression (Unweighted)",
            "Logistic Regression (Balanced)",
            "Random Forest (Balanced, max_depth=5)",
            "HistGradientBoosting (Balanced)",
        ]
        for name in expected_models:
            assert name in model_suite["models"], f"Missing model: {name}"
            assert name in model_suite["metrics"], f"Missing metrics for: {name}"

    def test_baseline_accuracy_trap(self, model_suite):
        # Baseline should have ~80.6% accuracy, but exactly 0 recall and precision
        baseline_m = model_suite["metrics"]["Baseline (Always 'No')"]
        assert baseline_m["accuracy"] > 0.75
        assert baseline_m["precision"] == 0.0
        assert baseline_m["recall"] == 0.0
        assert baseline_m["f1"] == 0.0

    def test_balanced_models_catch_smokers(self, model_suite):
        # Balanced Logistic Regression must have much higher recall than unweighted
        unweighted_rec = model_suite["metrics"]["Logistic Regression (Unweighted)"]["recall"]
        balanced_rec = model_suite["metrics"]["Logistic Regression (Balanced)"]["recall"]
        assert balanced_rec > unweighted_rec
        assert balanced_rec > 0.50

    def test_hist_gradient_boosting_metrics(self, model_suite):
        hgb_m = model_suite["metrics"]["HistGradientBoosting (Balanced)"]
        assert hgb_m["roc_auc"] > 0.65
        assert hgb_m["f1"] > 0.40


class TestLiveInference:
    """Tests for single profile prediction and risk scoring."""

    def test_predict_single_profile_clean_living(self, model_suite):
        clean_profile = {
            "age": 28,
            "drugs": "never",
            "drinks": "not at all",
            "body_type": "athletic",
        }
        res = predict_single_profile(clean_profile, "Logistic Regression (Balanced)", 0.50, model_suite)
        assert res["probability"] < 0.35
        assert res["prediction"] == 0
        assert res["prediction_label"] == "Non-Smoker"
        assert res["risk_level"] in ["Very Low Risk", "Low Risk"]

    def test_predict_single_profile_heavy_partier(self, model_suite):
        party_profile = {
            "age": 24,
            "drugs": "often",
            "drinks": "often",
            "body_type": "used up",
        }
        res = predict_single_profile(party_profile, "Logistic Regression (Balanced)", 0.50, model_suite)
        assert res["probability"] > 0.60
        assert res["prediction"] == 1
        assert res["prediction_label"] == "Smoker / Social Smoker"
        assert res["risk_level"] in ["High Risk", "Very High Risk"]

    def test_dynamic_threshold_adjustments(self, model_suite):
        profile = {"age": 26, "drugs": "sometimes", "drinks": "socially", "body_type": "fit"}
        # High threshold (e.g. 0.85) should predict non-smoker
        res_high = predict_single_profile(profile, "Logistic Regression (Balanced)", 0.85, model_suite)
        assert res_high["prediction"] == 0

        # Low threshold (e.g. 0.15) should predict smoker
        res_low = predict_single_profile(profile, "Logistic Regression (Balanced)", 0.15, model_suite)
        assert res_low["prediction"] == 1

    def test_threshold_curve_generation(self, model_suite):
        curve_df = model_suite["threshold_curve"]
        assert isinstance(curve_df, pd.DataFrame)
        assert "threshold" in curve_df.columns
        assert "precision" in curve_df.columns
        assert "recall" in curve_df.columns
        # As threshold increases, recall generally decreases
        first_recall = curve_df["recall"].iloc[0]
        last_recall = curve_df["recall"].iloc[-1]
        assert first_recall >= last_recall
