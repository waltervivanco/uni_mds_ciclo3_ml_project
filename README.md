# UNI MDS MLOps Final Project - WA1200

This repository contains the end-to-end workflow for a final MLOps project using WA1200 engine telemetry data.

## 1. Problem Definition

The use case is binary classification:

- Target: `High_Consumption`
- Goal: predict high fuel consumption events from engine telemetry.
- Business value: early detection of inefficient operating conditions for monitoring and maintenance decisions.

`High_Consumption` is defined from fuel rate percentile (P90) during experimentation.

## 2. Project Structure

```text
.
|- data/
|  |- raw/
|  |- processed/
|- notebooks/
|  |- 01_EDA_WA1200.ipynb
|- src/
|  |- data_preparation.py
|  |- train.py
|  |- serving.py
|- reports/
|  |- phase2_summary.md
|- requirements.txt
|- README.md
```

## 3. Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Data Preparation

Run:

```powershell
python src/data_preparation.py
```

Input:

- `data/raw/TL-20240723-202614 WA1200 #05.csv`

Output:

- `data/processed/wa1200_eda_ready.csv`

## 5. Train Baseline Models

Run:

```powershell
python src/train.py
```

Outputs:

- `artifacts/models/baseline_model.joblib`
- `artifacts/meta/feature_columns.joblib`
- `artifacts/meta/baseline_metrics.json`

## 6. Model Serving (FastAPI)

Run API:

```powershell
uvicorn src.serving:app --host 0.0.0.0 --port 8000
```

Open docs:

- `http://localhost:8000/docs`

Prediction endpoint:

- `POST /predict`

## 7. Notebook Experiments

Primary experimentation notebook:

- `notebooks/01_EDA_WA1200.ipynb`

Contains:

- EDA
- preprocessing
- feature selection
- baseline model training
- metrics and artifact export

## 8. Delivery Notes

- Use this repository URL for final submission.
- Keep work in main branch for final delivery.
- Optional branches can be used for development and merged by PR.
