# Resumen Fase 2 (Preparacion de datos + entrenamiento base)

## Dataset

- Fuente: `data/raw/TL-20240723-202614 WA1200 #05.csv`
- Archivo preparado: `data/processed/wa1200_eda_ready.csv`
- Registros: 1245

## Variable objetivo

- Variable: `High_Consumption`
- Definicion: 10% superior del consumo de combustible (umbral P90) a partir del indice de combustible instantaneo.

## Control de leakage

- La variable de combustible usada para construir el target se excluye de las features de entrenamiento.

## Modelos base evaluados

- Logistic Regression (`class_weight=balanced`)
- Random Forest (`class_weight=balanced`)

## Ultimo resultado base

- Mejor modelo: Logistic Regression
- Accuracy: 0.7671
- Precision (clase 1): 0.2609
- Recall (clase 1): 0.7200
- F1 (clase 1): 0.3830
- ROC AUC: 0.8518

## Artefactos generados

- `artifacts/models/baseline_model.joblib`
- `artifacts/meta/feature_columns.joblib`
- `artifacts/meta/baseline_metrics.json`

## Siguiente paso

- Fase 3: ajuste de umbral y optimizacion del modelo para mejorar el balance precision/recall en clase 1.
