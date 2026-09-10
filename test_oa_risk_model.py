"""Smoke test for the OA model workflow using synthetic, non-clinical data."""

import numpy as np
import pandas as pd

from oa_risk_model import predict_risk, train_model


def main() -> None:
    rng = np.random.default_rng(42)
    row_count = 120
    age = rng.integers(35, 81, row_count)
    bmi = rng.normal(26.5, 4.0, row_count).clip(17, 42)
    pain = rng.integers(0, 11, row_count)
    stiffness = rng.integers(0, 91, row_count)
    crepitus = rng.choice(["yes", "no"], row_count)
    logits = -7 + 0.07 * age + 0.10 * bmi + 0.20 * pain + 0.01 * stiffness + (crepitus == "yes")
    probability = 1 / (1 + np.exp(-logits))
    labels = rng.binomial(1, probability)

    data = pd.DataFrame(
        {
            "patient_id": [f"TEST-{index:03d}" for index in range(row_count)],
            "age_years": age,
            "bmi": bmi,
            "pain_score_0_10": pain,
            "stiffness_duration_minutes": stiffness,
            "crepitus_reported": crepitus,
            "knee_oa_confirmed": labels,
        }
    )
    artifact, importances = train_model(data, "knee_oa_confirmed")
    predictions = predict_risk(artifact, data.drop(columns="knee_oa_confirmed").head(4))

    assert len(predictions) == 4
    assert predictions["oa_risk_percent"].between(0, 100).all()
    assert {"roc_auc", "brier_score", "sensitivity", "specificity"} <= set(artifact["metrics"])
    assert not importances.empty
    print("OA model smoke test passed.")


if __name__ == "__main__":
    main()
