"""Convert OAI AllClinical00 into a compact, leakage-controlled training table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


DEFAULT_ARCHIVE = Path(r"C:\Users\HARSH\Downloads\OAICompleteData_ASCII.zip")
DEFAULT_ENTRY = "OAI Complete Data_ASCII/AllClinical00.txt"
OUTCOME_COLUMN = "P01XRKOA"

# These fields are selected before reading the 92 MB ASCII file. The outcome is
# radiographic OA status; no radiographic field appears in MODEL_FEATURES.
SOURCE_COLUMNS = {
    "ID",
    OUTCOME_COLUMN,
    "V00AGE",
    "P01BMI",
    "P01KSX",
    "P02KINJ",
    "P02FAMHXKR",
    "P02PA1",
    "P02PA2",
    "P02PA3",
    "V00WOMTSL",
    "V00WOMTSR",
    "V00KOOSKPR",
    "V00KOOSKPL",
    "V00RKPFCRE",
    "V00LKPFCRE",
    "V00RKFHDEG",
    "V00LKFHDEG",
    "V00RKEFFB",
    "V00LKEFFB",
    "V00CSTIME1",
    "V00CSTIME2",
    "V00400MTIM",
}

FEATURE_DEFINITIONS = {
    "age_years": "V00AGE - age at baseline",
    "bmi": "P01BMI - calculated body mass index",
    "frequent_knee_symptom_status": "P01KSX - person-level frequent knee pain status",
    "prior_knee_injury": "P02KINJ - prior knee injury severe enough to impair walking for at least one week",
    "family_history_knee_replacement": "P02FAMHXKR - blood relative with knee replacement",
    "frequent_stair_climbing": "P02PA1 - climbs at least 10 flights of stairs on most days",
    "frequent_kneeling": "P02PA2 - kneels for 30 minutes or more on most days",
    "frequent_deep_knee_bending": "P02PA3 - squats or deeply bends knees for 30 minutes or more on most days",
    "womac_total_worse_knee": "Maximum of V00WOMTSL and V00WOMTSR",
    "koos_pain_worse_knee": "Minimum of V00KOOSKPR and V00KOOSKPL; lower KOOS means worse pain",
    "patellofemoral_crepitus_any": "Maximum of V00RKPFCRE and V00LKPFCRE",
    "knee_effusion_any": "Maximum of V00RKEFFB and V00LKEFFB",
    "flexion_contracture_worse_deg": "Maximum non-negative flexion contracture from V00RKFHDEG and V00LKFHDEG",
    "chair_stand_time_sec_mean": "Mean of V00CSTIME1 and V00CSTIME2",
    "walk_400m_time_sec": "V00400MTIM - 400 metre walk time",
}


def numeric_code(series: pd.Series) -> pd.Series:
    """Extract the leading numeric code from OAI's '1: Yes' style values."""
    values = series.astype("string").str.strip()
    return pd.to_numeric(
        values.str.extract(r"^(-?(?:\d+(?:\.\d*)?|\.\d+))", expand=False),
        errors="coerce",
    )


def row_max(*columns: pd.Series) -> pd.Series:
    return pd.concat(columns, axis=1).max(axis=1, skipna=True)


def row_mean(*columns: pd.Series) -> pd.Series:
    return pd.concat(columns, axis=1).mean(axis=1, skipna=True)


def load_oai_columns(archive_path: Path, entry_name: str) -> pd.DataFrame:
    if not archive_path.is_file():
        raise FileNotFoundError(f"OAI archive was not found: {archive_path}")

    with ZipFile(archive_path) as archive:
        if entry_name not in archive.namelist():
            matches = [name for name in archive.namelist() if name.endswith("AllClinical00.txt")]
            raise FileNotFoundError(
                f"Archive entry '{entry_name}' was not found. Matching entries: {matches}"
            )
        with archive.open(entry_name) as source:
            data = pd.read_csv(
                source,
                sep="|",
                dtype=str,
                usecols=lambda column: column in SOURCE_COLUMNS,
                low_memory=False,
            )

    missing_columns = SOURCE_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"The OAI file is missing expected columns: {sorted(missing_columns)}")
    return data


def prepare_oai_training_table(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    numeric = {column: numeric_code(data[column]) for column in SOURCE_COLUMNS if column != "ID"}
    outcome_code = numeric[OUTCOME_COLUMN]
    valid_outcome = outcome_code.isin([0, 1, 2, 3])

    flexion_contracture = pd.concat(
        [numeric["V00RKFHDEG"].clip(lower=0), numeric["V00LKFHDEG"].clip(lower=0)], axis=1
    ).max(axis=1, skipna=True)

    outcome_label = outcome_code.map({0: "no", 1: "yes", 2: "yes", 3: "yes"})
    prepared = pd.DataFrame(
        {
            "patient_id": data["ID"].astype("string"),
            "age_years": numeric["V00AGE"],
            "bmi": numeric["P01BMI"],
            "frequent_knee_symptom_status": numeric["P01KSX"],
            "prior_knee_injury": numeric["P02KINJ"],
            "family_history_knee_replacement": numeric["P02FAMHXKR"],
            "frequent_stair_climbing": numeric["P02PA1"],
            "frequent_kneeling": numeric["P02PA2"],
            "frequent_deep_knee_bending": numeric["P02PA3"],
            "womac_total_worse_knee": row_max(numeric["V00WOMTSL"], numeric["V00WOMTSR"]),
            "koos_pain_worse_knee": pd.concat(
                [numeric["V00KOOSKPR"], numeric["V00KOOSKPL"]], axis=1
            ).min(axis=1, skipna=True),
            "patellofemoral_crepitus_any": row_max(
                numeric["V00RKPFCRE"], numeric["V00LKPFCRE"]
            ),
            "knee_effusion_any": row_max(numeric["V00RKEFFB"], numeric["V00LKEFFB"]),
            "flexion_contracture_worse_deg": flexion_contracture,
            "chair_stand_time_sec_mean": row_mean(
                numeric["V00CSTIME1"], numeric["V00CSTIME2"]
            ),
            "walk_400m_time_sec": numeric["V00400MTIM"],
            "knee_oa_confirmed": outcome_label,
        }
    )
    prepared = prepared.loc[valid_outcome].reset_index(drop=True)

    positive_rows = int((prepared["knee_oa_confirmed"] == "yes").sum())
    report = {
        "source_rows": int(len(data)),
        "rows_with_valid_radiographic_outcome": int(len(prepared)),
        "excluded_missing_or_nonstandard_outcome_rows": int((~valid_outcome).sum()),
        "outcome": {
            "source_column": OUTCOME_COLUMN,
            "definition": (
                "Any baseline radiographic knee OA. OAI code 0 is neither knee; "
                "codes 1, 2, and 3 indicate right, left, or both knees."
            ),
            "positive_rows": positive_rows,
            "negative_rows": int(len(prepared) - positive_rows),
        },
        "features": FEATURE_DEFINITIONS,
        "safety": (
            "No radiographic source field is retained as a predictor. This data is suitable "
            "for research screening model development, not clinical deployment."
        ),
    }
    return prepared, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a knee-OA training table directly from OAI AllClinical00 inside a ZIP archive."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--entry", default=DEFAULT_ENTRY)
    parser.add_argument("--out", type=Path, default=Path("oai_allclinical00_training.csv"))
    parser.add_argument(
        "--report-out", type=Path, default=Path("oai_allclinical00_preparation_report.json")
    )
    args = parser.parse_args()

    raw_data = load_oai_columns(args.archive, args.entry)
    training_data, report = prepare_oai_training_table(raw_data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(args.out, index=False)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved {len(training_data)} training rows: {args.out.resolve()}")
    print(f"Saved preparation report: {args.report_out.resolve()}")
    print(
        "Radiographic OA outcome: "
        f"{report['outcome']['positive_rows']} positive / {report['outcome']['negative_rows']} negative"
    )


if __name__ == "__main__":
    main()
