# Knee-OA Screening Model Card

## Intended use

Research-only knee osteoarthritis screening support from patient questionnaire data. This output is not a diagnosis and must not replace clinical assessment, examination, or appropriate imaging.

## Training data

- Source: `oai_allclinical00_training.csv`
- Records: 4795
- OA-positive records: 2679
- OA-negative records: 2116
- Validation: 5-fold stratified cross-validation

## Cross-validated performance

- ROC-AUC: 0.728
- Average precision: 0.761
- Brier score: 0.209
- Sensitivity at 50% threshold: 0.735
- Specificity at 50% threshold: 0.597

## Important limitation

Independent external validation is still required before clinical use.
