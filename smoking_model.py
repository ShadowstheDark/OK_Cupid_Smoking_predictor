"""
smoking_model.py
================
Production Machine Learning Pipeline for OkCupid Smoking Prediction.

Features:
- Encodes demographic and lifestyle features (drugs, drinks, body_type, age).
- Fixes the "did not answer" data flaw by treating missing values as an explicit
  "unspecified" category rather than conflating them with "never".
- Trains balanced and unweighted benchmark models:
    1. Baseline (DummyClassifier - always "no")
    2. Logistic Regression (Unweighted)
    3. Logistic Regression (Balanced)
    4. Random Forest (Balanced, max_depth=5)
    5. HistGradientBoosting (Balanced, LightGBM-style modern booster)
- Evaluates metrics on stratified held-out test split (Accuracy, Precision, Recall, F1, ROC-AUC).
- Computes confusion matrices, precision-recall threshold trade-offs, and feature attributions.
- Provides real-time inference via `predict_single_profile()`.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from data_loader import load_data

# Canonical categories for deterministic encoding (matching UI options)
DRUG_OPTIONS: List[str] = ["never", "sometimes", "often", "unspecified"]
DRINK_OPTIONS: List[str] = ["socially", "rarely", "often", "not at all", "very often", "desperately", "unspecified"]
BODY_TYPE_OPTIONS: List[str] = [
    "average",
    "fit",
    "athletic",
    "thin",
    "curvy",
    "a little extra",
    "skinny",
    "full figured",
    "overweight",
    "jacked",
    "used up",
    "rather not say",
    "unspecified",
]

SMOKER_POSITIVE_LABELS = {"sometimes", "when drinking", "yes", "trying to quit"}

# Module-level cache to avoid redundant retraining across Streamlit runs
_MODEL_SUITE_CACHE: Optional[Dict[str, Any]] = None


def prepare_training_data(df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], float]:
    """
    Prepare feature matrix X and target y from the dataset.
    
    Returns:
        X: One-hot encoded feature DataFrame with fixed column schema
        y: Binary target Series (1 = smoker/social smoker, 0 = non-smoker)
        feature_names: List of column names in X
        median_age: Median age used for imputation
    """
    if df is None:
        df = load_data()

    # Drop rows where smoking status was unspecified (cannot train/score without ground truth)
    df_clean = df[df["smokes"].isin(SMOKER_POSITIVE_LABELS | {"no"})].copy()

    # Binary target: 1 = smoker / social smoker / trying to quit, 0 = no
    y = df_clean["smokes"].apply(lambda s: 1 if s in SMOKER_POSITIVE_LABELS else 0).astype(int)

    # Impute and sanitize age
    median_age = float(df_clean["age"].median())
    age_series = df_clean["age"].fillna(median_age).clip(lower=18, upper=80)

    # Sanitize categorical lifestyle features
    drugs_series = df_clean["drugs"].apply(lambda x: x if x in DRUG_OPTIONS else "unspecified")
    drinks_series = df_clean["drinks"].apply(lambda x: x if x in DRINK_OPTIONS else "unspecified")
    body_series = df_clean["body_type"].apply(lambda x: x if x in BODY_TYPE_OPTIONS else "unspecified")

    # Construct one-hot encoding with fixed columns
    feature_dict: Dict[str, Any] = {"age": age_series.values}

    # Reference categories dropped for linear model identifiability:
    # drugs: "never", drinks: "socially", body_type: "average"
    for cat in DRUG_OPTIONS:
        if cat != "never":
            feature_dict[f"drugs_{cat}"] = (drugs_series == cat).astype(int).values

    for cat in DRINK_OPTIONS:
        if cat != "socially":
            feature_dict[f"drinks_{cat}"] = (drinks_series == cat).astype(int).values

    for cat in BODY_TYPE_OPTIONS:
        if cat != "average":
            feature_dict[f"body_type_{cat}"] = (body_series == cat).astype(int).values

    X = pd.DataFrame(feature_dict, index=df_clean.index)
    feature_names = list(X.columns)

    return X, y, feature_names, median_age


def encode_single_profile(
    profile: Dict[str, Any],
    feature_names: List[str],
    median_age: float = 30.0
) -> pd.DataFrame:
    """
    Encode a single user profile dictionary into the exact feature vector schema.
    """
    age = float(profile.get("age", median_age))
    drugs = str(profile.get("drugs", "unspecified")).lower().strip()
    drinks = str(profile.get("drinks", "unspecified")).lower().strip()
    body_type = str(profile.get("body_type", "unspecified")).lower().strip()

    row: Dict[str, float] = {"age": age}

    for col in feature_names:
        if col == "age":
            continue
        elif col.startswith("drugs_"):
            val = col.replace("drugs_", "")
            row[col] = 1.0 if drugs == val else 0.0
        elif col.startswith("drinks_"):
            val = col.replace("drinks_", "")
            row[col] = 1.0 if drinks == val else 0.0
        elif col.startswith("body_type_"):
            val = col.replace("body_type_", "")
            row[col] = 1.0 if body_type == val else 0.0
        else:
            row[col] = 0.0

    return pd.DataFrame([row], columns=feature_names)


def train_model_suite(df: Optional[pd.DataFrame] = None, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Train and evaluate all benchmark models.
    """
    X, y, feature_names, median_age = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models: Dict[str, Any] = {
        "Baseline (Always 'No')": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Unweighted)": LogisticRegression(max_iter=1000, random_state=random_state),
        "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "Random Forest (Balanced, max_depth=5)": RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "HistGradientBoosting (Balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", max_depth=5, random_state=random_state
        ),
    }

    metrics: Dict[str, Dict[str, Any]] = {}
    test_probs: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            try:
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                auc = 0.5
        else:
            probs = np.zeros(len(y_test))
            auc = 0.5

        test_probs[name] = probs

        cm = confusion_matrix(y_test, preds)
        cm_norm = confusion_matrix(y_test, preds, normalize="true")

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_norm": cm_norm.tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        }

    # Feature Importance artifacts
    lr_balanced = models["Logistic Regression (Balanced)"]
    lr_unweighted = models["Logistic Regression (Unweighted)"]
    rf_model = models["Random Forest (Balanced, max_depth=5)"]

    feature_impact_df = pd.DataFrame({
        "feature": feature_names,
        "lr_balanced_coef": lr_balanced.coef_[0],
        "lr_balanced_odds_ratio": np.exp(lr_balanced.coef_[0]),
        "lr_unweighted_coef": lr_unweighted.coef_[0],
        "rf_importance": rf_model.feature_importances_,
    }).sort_values(by="lr_balanced_coef", ascending=False).reset_index(drop=True)

    # Threshold curve analysis for Logistic Regression (Balanced)
    threshold_curve = compute_threshold_curve(
        test_probs["Logistic Regression (Balanced)"], y_test.values
    )

    base_rate = float(y.mean())

    suite: Dict[str, Any] = {
        "models": models,
        "metrics": metrics,
        "feature_names": feature_names,
        "median_age": median_age,
        "feature_impact_df": feature_impact_df,
        "threshold_curve": threshold_curve,
        "base_rate": base_rate,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_test": y_test.values.tolist(),
        "test_probs": {k: v.tolist() for k, v in test_probs.items()},
    }

    return suite


def compute_threshold_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Compute Accuracy, Precision, Recall, and F1 across decision thresholds.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": t,
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "predicted_smokers": int(preds.sum()),
        })

    return pd.DataFrame(rows)


def get_model_suite(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Retrieve cached model suite or train once if not loaded.
    """
    global _MODEL_SUITE_CACHE
    if _MODEL_SUITE_CACHE is None or force_retrain:
        _MODEL_SUITE_CACHE = train_model_suite()
    return _MODEL_SUITE_CACHE


def predict_single_profile(
    profile: Dict[str, Any],
    model_name: str = "Logistic Regression (Balanced)",
    threshold: float = 0.50,
    suite: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform live inference on an individual profile dictionary.
    
    Args:
        profile: dict with keys {"age", "drugs", "drinks", "body_type"}
        model_name: Name of model in the suite
        threshold: Decision threshold for positive classification (default 0.50)
        suite: Optional pre-loaded model suite
        
    Returns:
        dict with probability, prediction label, risk tier, color, and feature attributions.
    """
    if suite is None:
        suite = get_model_suite()

    model = suite["models"].get(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(suite['models'].keys())}")

    X_single = encode_single_profile(profile, suite["feature_names"], suite["median_age"])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_single)[0, 1])
    else:
        prob = float(model.predict(X_single)[0])

    prediction = 1 if prob >= threshold else 0
    prediction_label = "Smoker / Social Smoker" if prediction == 1 else "Non-Smoker"

    # Risk categorization
    if prob < 0.25:
        risk_level = "Very Low Risk"
        risk_color = "#10b981"  # Emerald
    elif prob < 0.40:
        risk_level = "Low Risk"
        risk_color = "#34d399"  # Mint
    elif prob < 0.60:
        risk_level = "Moderate / Borderline"
        risk_color = "#f59e0b"  # Amber
    elif prob < 0.75:
        risk_level = "High Risk"
        risk_color = "#f97316"  # Orange
    else:
        risk_level = "Very High Risk"
        risk_color = "#ff2a5f"  # Crimson

    # Feature attributions for Logistic Regression
    attributions = []
    lr_model = suite["models"].get("Logistic Regression (Balanced)")
    if lr_model is not None and hasattr(lr_model, "coef_"):
        coefs = lr_model.coef_[0]
        # Intercept contribution
        intercept = float(lr_model.intercept_[0])
        attributions.append({
            "feature": "Baseline (Population Intercept)",
            "value": "Base Odds",
            "log_odds_impact": round(intercept, 3),
            "odds_ratio": round(float(np.exp(intercept)), 3),
            "direction": "negative" if intercept < 0 else "positive",
        })

        # Feature contributions
        for col, coef in zip(suite["feature_names"], coefs):
            col_val = X_single[col].values[0]
            if col == "age":
                # Relative to age 30
                diff = (col_val - 30.0)
                impact = diff * coef
                if abs(impact) > 0.01:
                    attributions.append({
                        "feature": f"Age ({int(col_val)} yrs vs 30)",
                        "value": f"{int(col_val)} yrs",
                        "log_odds_impact": round(impact, 3),
                        "odds_ratio": round(float(np.exp(impact)), 3),
                        "direction": "positive" if impact > 0 else "negative",
                    })
            else:
                if col_val == 1.0:
                    attributions.append({
                        "feature": col.replace("_", ": ").title(),
                        "value": "Selected",
                        "log_odds_impact": round(coef, 3),
                        "odds_ratio": round(float(np.exp(coef)), 3),
                        "direction": "positive" if coef > 0 else "negative",
                    })

    # Population comparison odds ratio
    base_odds = suite["base_rate"] / (1.0 - suite["base_rate"] + 1e-9)
    profile_odds = prob / (1.0 - prob + 1e-9)
    relative_odds = profile_odds / (base_odds + 1e-9)

    return {
        "model_name": model_name,
        "probability": prob,
        "probability_percent": round(prob * 100, 1),
        "prediction": prediction,
        "prediction_label": prediction_label,
        "threshold": threshold,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "relative_odds_vs_population": round(relative_odds, 2),
        "feature_attributions": attributions,
    }


if __name__ == "__main__":
    print("Training smoking prediction models...")
    suite = train_model_suite()
    print(f"Features: {len(suite['feature_names'])} columns")
    print(f"Base Rate (Smokers): {suite['base_rate']:.1%}")
    print("\nBenchmark Evaluation:")
    for name, m in suite["metrics"].items():
        print(f"  {name:38s} | Acc: {m['accuracy']:.3f} | Prec: {m['precision']:.3f} | Rec: {m['recall']:.3f} | F1: {m['f1']:.3f}")

    # Test single inference
    test_profile = {
        "age": 28,
        "drugs": "often",
        "drinks": "often",
        "body_type": "average",
    }
    pred = predict_single_profile(test_profile, "Logistic Regression (Balanced)", 0.50, suite)
    print("\nTest Profile Prediction (Party Regular):")
    print(f"  Probability: {pred['probability_percent']}% -> {pred['prediction_label']} ({pred['risk_level']})")
    print(f"  Relative Odds vs Pop: {pred['relative_odds_vs_population']}x")
