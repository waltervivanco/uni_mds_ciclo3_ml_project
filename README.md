# Proyecto Final MLOps UNI MDS - WA1200

Este repositorio contiene el flujo end-to-end del proyecto final de MLOps usando telemetria del motor WA1200.

## 1. Definicion del problema

El caso de uso es de clasificacion binaria:

- Objetivo (`target`): `High_Consumption`
- Meta: predecir eventos de alto consumo de combustible a partir de telemetria del motor.
- Valor de negocio: detectar de forma temprana condiciones ineficientes de operacion para apoyar monitoreo y mantenimiento.

`High_Consumption` se define con el percentil 90 (P90) del indice de combustible instantaneo durante la etapa de experimentacion.

## 2. Estructura del proyecto

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

## 3. Configuracion del entorno

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Preparacion de datos

Ejecutar:

```powershell
python src/data_preparation.py
```

Entrada:

- `data/raw/TL-20240723-202614 WA1200 #05.csv`

Salida:

- `data/processed/wa1200_eda_ready.csv`

## 5. Entrenamiento de modelos base

Ejecutar:

```powershell
python src/train.py
```

Artefactos generados:

- `artifacts/models/baseline_model.joblib`
- `artifacts/meta/feature_columns.joblib`
- `artifacts/meta/baseline_metrics.json`

## 6. Despliegue del modelo (FastAPI)

Levantar API:

```powershell
uvicorn src.serving:app --host 0.0.0.0 --port 8000
```

Documentacion interactiva:

- `http://localhost:8000/docs`

Endpoint de prediccion:

- `POST /predict`

## 7. Notebook de experimentacion

Notebook principal:

- `notebooks/01_EDA_WA1200.ipynb`

Incluye:

- Analisis exploratorio (EDA)
- Preprocesamiento
- Seleccion de variables
- Entrenamiento de modelos base
- Metricas y exportacion de artefactos


