"""Train and export the calibrated knee-OA screening model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from oa_risk_model import DEFAULT_THRESHOLD, read_tabular_file, train_model


def build_model_card(artifact: dict, data_path: Path) -> str:
    metrics = artifact["metrics"]
    return f"""# Knee-OA Screening Model Card

## Intended use

Research-only knee osteoarthritis screening support from patient questionnaire data. This output is not a diagnosis and must not replace clinical assessment, examination, or appropriate imaging.

## Training data

- Source: `{data_path.name}`
- Records: {metrics['training_rows']}
- OA-positive records: {metrics['oa_positive_rows']}
- OA-negative records: {metrics['oa_negative_rows']}
- Validation: {metrics['validation']}

## Cross-validated performance

- ROC-AUC: {metrics['roc_auc']:.3f}
- Average precision: {metrics['average_precision']:.3f}
- Brier score: {metrics['brier_score']:.3f}
- Sensitivity at {metrics['threshold'] * 100:.0f}% threshold: {metrics['sensitivity']:.3f}
- Specificity at {metrics['threshold'] * 100:.0f}% threshold: {metrics['specificity']:.3f}

## Important limitation

{metrics['warning']}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a calibrated XGBoost knee-OA screening model from labeled patient data."
    )
    parser.add_argument("--data", required=True, help="Labeled patient CSV/XLSX file.")
    parser.add_argument(
        "--target",
        default="knee_oa_confirmed",
        help="0/1 or yes/no clinician-confirmed OA label column.",
    )
    parser.add_argument(
        "--model-out",
        default="knee_oa_xgboost.joblib",
        help="Path for the exported model artifact.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Screen-positive threshold from 0 to 1; default is 0.50.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model_out)
    data = read_tabular_file(str(data_path))
    artifact, importances = train_model(data, args.target, args.threshold)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    importance_path = model_path.with_name(f"{model_path.stem}_feature_importance.csv")
    model_card_path = model_path.with_name(f"{model_path.stem}_model_card.md")
    metrics_path.write_text(json.dumps(artifact["metrics"], indent=2), encoding="utf-8")
    importances.to_csv(importance_path, index=False)
    model_card_path.write_text(build_model_card(artifact, data_path), encoding="utf-8")

    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved feature importance: {importance_path.resolve()}")
    print(f"Saved model card: {model_card_path.resolve()}")
    print(f"Cross-validated ROC-AUC: {artifact['metrics']['roc_auc']:.3f}")
    print(f"Cross-validated Brier score: {artifact['metrics']['brier_score']:.3f}")


if __name__ == "__main__":
    main()
