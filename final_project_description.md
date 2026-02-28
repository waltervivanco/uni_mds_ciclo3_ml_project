## Descripcion del Proyecto Final

### Titulo
Prediccion de eventos de alto consumo en telemetria WA1200

### Problema
Se busca identificar condiciones operativas asociadas a alto consumo de combustible en motores WA1200 usando datos de telemetria.

### Objetivo de ML
Entrenar un clasificador binario para predecir la variable `High_Consumption`.

### Datos
- Dataset crudo: `data/raw/TL-20240723-202614 WA1200 #05.csv`
- Dataset procesado: `data/processed/wa1200_eda_ready.csv`

### Flujo de trabajo implementado
1. EDA y analisis en notebook (`notebooks/01_EDA_WA1200.ipynb`).
2. Preparacion de datos en script modular (`src/data_preparation.py`).
3. Entrenamiento de modelos base (`src/train.py`).
4. Serializacion de artefactos en `artifacts/`.
5. Serving del modelo con FastAPI (`src/serving.py`).

### Artefactos
- Modelo: `artifacts/models/baseline_model.joblib`
- Columnas de entrada: `artifacts/meta/feature_columns.joblib`
- Metricas: `artifacts/meta/baseline_metrics.json`

### Resultado base
Modelo seleccionado: Logistic Regression con enfoque en deteccion de eventos de alto consumo.

### Despliegue
API REST con FastAPI y endpoint `POST /predict` para inferencia.
