"""Reusable training and inference helpers for knee-OA screening research."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.50
MIN_TRAINING_ROWS = 20
MIN_CLASS_ROWS = 5

IDENTIFIER_COLUMNS = {
    "patient_id",
    "patient_name",
    "name",
    "record_id",
    "medical_record_number",
    "phone",
    "email",
    "address",
}
LEAKAGE_TOKENS = (
    "diagnosis",
    "radiograph",
    "xray",
    "x_ray",
    "mri",
    "kl_grade",
    "kellgren",
    "doctor_assessment",
    "clinician_assessment",
)
POSITIVE_LABELS = {"1", "yes", "true", "positive", "oa", "osteoarthritis", "knee_oa"}
NEGATIVE_LABELS = {"0", "no", "false", "negative", "non_oa", "no_oa", "healthy"}


class DataValidationError(ValueError):
    """Raised when training or prediction data is unsafe to use."""


def select_numeric_columns(frame: pd.DataFrame) -> pd.Index:
    """Pick numeric feature columns for the persisted preprocessing pipeline."""
    return frame.select_dtypes(include=np.number).columns


def select_categorical_columns(frame: pd.DataFrame) -> pd.Index:
    """Pick categorical feature columns for the persisted preprocessing pipeline."""
    return frame.select_dtypes(exclude=np.number).columns


def read_tabular_file(path: str) -> pd.DataFrame:
    """Load a CSV or Excel file without guessing the target column."""
    file_path = str(path)
    lower_name = file_path.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    raise DataValidationError("Input data must be a .csv, .xlsx, or .xls file.")


def _normalise_binary_target(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise DataValidationError("The target column contains missing values.")

    normalised = values.astype(str).str.strip().str.lower()
    valid_labels = POSITIVE_LABELS | NEGATIVE_LABELS
    invalid_labels = sorted(set(normalised) - valid_labels)
    if invalid_labels:
        raise DataValidationError(
            "Target values must be one of "
            f"{sorted(valid_labels)}. Unexpected values: {invalid_labels[:5]}"
        )

    return normalised.isin(POSITIVE_LABELS).astype(int)


def _is_disallowed_feature(column: str, target_column: str) -> bool:
    normalised = column.strip().lower()
    if normalised == target_column.strip().lower() or normalised in IDENTIFIER_COLUMNS:
        return True
    return any(token in normalised for token in LEAKAGE_TOKENS)


def _validate_training_data(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target_column not in data.columns:
        raise DataValidationError(
            f"Target column '{target_column}' was not found. Available columns: {list(data.columns)}"
        )
    if data.empty:
        raise DataValidationError("The training file has no rows.")
    if len(data) < MIN_TRAINING_ROWS:
        raise DataValidationError(
            f"At least {MIN_TRAINING_ROWS} labeled records are required; found {len(data)}."
        )

    y = _normalise_binary_target(data[target_column])
    class_counts = y.value_counts()
    if len(class_counts) != 2:
        raise DataValidationError("The target must include both OA-positive and OA-negative records.")
    if class_counts.min() < MIN_CLASS_ROWS:
        raise DataValidationError(
            f"At least {MIN_CLASS_ROWS} records are required in each class; found {class_counts.to_dict()}."
        )

    excluded_columns = [
        column for column in data.columns if _is_disallowed_feature(column, target_column)
    ]
    feature_columns = [column for column in data.columns if column not in excluded_columns]
    if not feature_columns:
        raise DataValidationError("No usable feature columns remain after excluding identifiers and leakage fields.")

    features = data.loc[:, feature_columns].copy()
    if features.isna().all(axis=None):
        raise DataValidationError("All feature values are missing.")

    return features, y, feature_columns, excluded_columns


def _build_base_estimator(positive_count: int, negative_count: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, select_numeric_columns),
            ("categorical", categorical_transformer, select_categorical_columns),
        ],
        remainder="drop",
    )

    # Kept deliberately shallow and regularized for small clinical cohorts.
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.25,
        reg_lambda=5.0,
        gamma=0.10,
        scale_pos_weight=negative_count / positive_count,
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _build_calibrated_estimator(positive_count: int, negative_count: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=_build_base_estimator(positive_count, negative_count),
        method="sigmoid",
        cv=3,
        ensemble=True,
    )


def _metric_summary(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "positive_predictive_value": float(precision),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _feature_importance(model: CalibratedClassifierCV) -> pd.DataFrame:
    importances: list[np.ndarray] = []
    transformed_feature_names: np.ndarray | None = None

    for calibrated_classifier in model.calibrated_classifiers_:
        pipeline = calibrated_classifier.estimator
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed_feature_names = preprocessor.get_feature_names_out()
        importances.append(classifier.feature_importances_)

    if transformed_feature_names is None or not importances:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": np.mean(importances, axis=0),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_model(
    data: pd.DataFrame, target_column: str, threshold: float = DEFAULT_THRESHOLD
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train a calibrated screening model and return its portable artifact."""
    if not 0.0 < threshold < 1.0:
        raise DataValidationError("Threshold must be between 0 and 1.")

    features, y, feature_columns, excluded_columns = _validate_training_data(data, target_column)
    class_counts = y.value_counts()
    positive_count = int(class_counts[1])
    negative_count = int(class_counts[0])

    outer_splits = min(5, positive_count, negative_count)
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED
    )
    calibrated_estimator = _build_calibrated_estimator(positive_count, negative_count)
    cross_validated_probabilities = cross_val_predict(
        calibrated_estimator,
        features,
        y,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    metrics = _metric_summary(y, cross_validated_probabilities, threshold)

    final_model = _build_calibrated_estimator(positive_count, negative_count)
    final_model.fit(features, y)
    metrics.update(
        {
            "validation": f"{outer_splits}-fold stratified cross-validation",
            "training_rows": int(len(data)),
            "oa_positive_rows": positive_count,
            "oa_negative_rows": negative_count,
            "warning": (
                "This dataset has fewer than 100 rows; reported probabilities are research-only "
                "until independently validated on a representative held-out cohort."
                if len(data) < 100
                else "Independent external validation is still required before clinical use."
            ),
        }
    )

    artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "excluded_columns": excluded_columns,
        "metrics": metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "Calibrated XGBoost knee-OA screening model",
    }
    return artifact, _feature_importance(final_model)


def predict_risk(
    artifact: dict[str, Any], data: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Return screening probabilities as percentages for one or more patient records."""
    if not 0.0 < threshold < 1.0:
        raise DataValidationError("Threshold must be between 0 and 1.")

    required_columns = artifact["feature_columns"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise DataValidationError(
            "Prediction data is missing training features: " + ", ".join(missing_columns)
        )

    probabilities = artifact["model"].predict_proba(data.loc[:, required_columns])[:, 1]
    result = pd.DataFrame(
        {
            "oa_risk_percent": np.round(probabilities * 100, 1),
            "screen_positive": probabilities >= threshold,
            "screening_threshold_percent": round(threshold * 100, 1),
            "screening_note": "Research screening estimate only; not a clinical diagnosis.",
        }
    )

    for identifier in ("patient_id", "patient_name", "record_id"):
        if identifier in data.columns:
            result.insert(0, identifier, data[identifier].values)
    return result
