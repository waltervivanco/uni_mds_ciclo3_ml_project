# Phase 2 Summary (Data Prep + Baseline Training)

## Dataset

- Source: `data/raw/TL-20240723-202614 WA1200 #05.csv`
- Prepared file: `data/processed/wa1200_eda_ready.csv`
- Rows: 1245

## Target

- Variable: `High_Consumption`
- Definition: top 10% fuel rate (P90 threshold) from instant fuel index.

## Leakage Control

- Fuel variable used to construct target is removed from training features.

## Baseline Models

- Logistic Regression (`class_weight=balanced`)
- Random Forest (`class_weight=balanced`)

## Latest Baseline Result

- Best model: Logistic Regression
- Accuracy: 0.7671
- Precision (class 1): 0.2609
- Recall (class 1): 0.7200
- F1 (class 1): 0.3830
- ROC AUC: 0.8518

## Generated Artifacts

- `artifacts/models/baseline_model.joblib`
- `artifacts/meta/feature_columns.joblib`
- `artifacts/meta/baseline_metrics.json`

## Next Step

- Phase 3: threshold tuning and model optimization for class 1 precision/recall tradeoff.
