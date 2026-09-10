# Knee-OA XGBoost Screening Model

This folder trains a calibrated XGBoost model from patient questionnaire records and returns a knee-OA screening percentage for each patient. It is a research-screening component, not a diagnosis.

## Prepare labeled data

Use `patient_questionnaire_template.csv` as the header for your dataset. Each row is one patient assessment. The required target is `knee_oa_confirmed`:

- `1` / `yes` means knee OA confirmed by an independent clinical reference standard.
- `0` / `no` means knee OA not confirmed.

Do not derive this label from the same questionnaire rules used as model inputs. Do not include names, IDs, imaging results, grades, or a doctor diagnosis as feature columns. The training code excludes common identifiers and obvious diagnosis/imaging columns, but the dataset owner remains responsible for preventing leakage and protecting patient privacy.

## Train

From this folder, run:

```powershell
python train_oa_model.py --data patient_questionnaire_labeled.csv --target knee_oa_confirmed --model-out knee_oa_xgboost.joblib
```

The command saves the trained model, cross-validated metrics, a feature-importance CSV, and a model card. It uses shallow, regularized trees plus sigmoid calibration because the percentage output should be treated as a probability estimate, not just a ranking score.

## Train from the supplied OAI archive

The OAI archive can be prepared directly without extracting its 92 MB source text file:

```powershell
python prepare_oai_allclinical00.py --archive "C:\Users\HARSH\Downloads\OAICompleteData_ASCII.zip" --out oai_allclinical00_training.csv
python train_oa_model.py --data oai_allclinical00_training.csv --target knee_oa_confirmed --model-out oai_knee_oa_xgboost.joblib
```

`P01XRKOA` is used only to create the radiographic OA outcome. The preparation script removes radiographic fields from the feature table so the model does not learn the answer from an x-ray result. See `OAI_FEATURE_MAP.md` for the 15 selected patient, physical-exam, and functional-test features.

## Predict percentages

Use a CSV/XLSX with the same feature columns used for training. It may omit `knee_oa_confirmed`.

```powershell
python predict_oa_risk.py --model knee_oa_xgboost.joblib --data new_patients.csv --out oa_risk_predictions.csv
```

The result includes `oa_risk_percent` and `screen_positive`. The 50% screen-positive threshold is only a starting configuration; choose the operating threshold with a clinician after reviewing sensitivity, specificity, and the consequences of false positives and false negatives.

## Data-size guardrail

The code accepts at least 20 labeled records and at least 5 patients per class, but it places a warning in the model card for datasets below 100 rows. In particular, a 27-patient cohort can support a pilot demonstration only; it cannot establish a clinically dependable probability model without independent, representative validation.
