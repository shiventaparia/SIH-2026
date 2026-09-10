"""Generate calibrated knee-OA screening percentages from a saved model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from oa_risk_model import DEFAULT_THRESHOLD, predict_risk, read_tabular_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict knee-OA screening risk percentages for patient records."
    )
    parser.add_argument("--model", required=True, help="Exported .joblib model artifact.")
    parser.add_argument("--data", required=True, help="Patient CSV/XLSX file for prediction.")
    parser.add_argument(
        "--out",
        default="oa_risk_predictions.csv",
        help="CSV path for screening predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Screen-positive threshold from 0 to 1; default is 0.50.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    data = read_tabular_file(args.data)
    predictions = predict_risk(artifact, data, args.threshold)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions)} screening predictions: {output_path.resolve()}")


if __name__ == "__main__":
    main()
