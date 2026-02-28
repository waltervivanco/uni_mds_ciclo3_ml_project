# Proyecto Final MLOps UNI MDS - WA1200

Este repositorio implementa un flujo MLOps end-to-end para detectar eventos de alto consumo en telemetria del motor WA1200.

## 1. Definicion del problema

El caso de uso es de clasificacion binaria:

- Objetivo (`target`): `High_Consumption`
- Meta: predecir eventos de alto consumo de combustible a partir de telemetria del motor.
- Valor de negocio: deteccion temprana de condiciones ineficientes para apoyar monitoreo y mantenimiento.

La variable `High_Consumption` se define con el percentil 90 (P90) del indice de combustible instantaneo.

### Contexto operacional WA1200

El WA1200 es un equipo principal de carguio en mina. Su funcion es cargar material y depositarlo en los camiones 730E.
Por su rol critico en el ciclo de acarreo, cualquier sobreconsumo de combustible afecta costo por tonelada, productividad y disponibilidad.

### Posibles causas de sobreconsumo consideradas en el analisis

- Inyector con fuga o atomizacion deficiente.
- Filtro de aire o combustible obstruido.
- Presion de admision/turbo fuera de rango.
- Presion de riel de combustible inestable.
- Operacion prolongada en regimen ineficiente (rpm altas con baja carga util o ralenti excesivo).
- Temperaturas elevadas (refrigerante, aceite, aire de admision).
- Desgaste de componentes del motor o calibracion no optima.

## 2. Estructura del proyecto

```text
.
|- data/
|  |- raw/
|  |- processed/
|  |- features/
|- notebooks/
|  |- 01_EDA_WA1200.ipynb
|- src/
|  |- data_preparation.py
|  |- train.py
|  |- serving.py
|- artifacts/
|  |- models/
|  |- meta/
|- models/
|- reports/
|  |- phase2_summary.md
|- tests/
|- experiments/
|- resources/
|  |- images/
|- requirements.txt
|- final_project_description.md
|- CHANGELOG.md
|- README.md
```

## 3. Configuracion del entorno

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea `Activate.ps1`, usar:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Proceso end-to-end

1. Ingesta de datos crudos (`data/raw`).
2. Limpieza, correccion de texto y normalizacion de columnas (`src/data_preparation.py`).
3. Construccion de dataset procesado (`data/processed`) y dataset final (`data/features`).
4. Entrenamiento y comparacion de modelos base (`src/train.py`).
5. Guardado de artefactos (modelo, columnas, metricas).
6. Despliegue local del modelo como API (`src/serving.py`).

## 5. Preparacion de datos

Ejecutar:

```powershell
python src/data_preparation.py
```

Entrada:

- `data/raw/TL-20240723-202614 WA1200 #05.csv`

Salidas:

- `data/processed/wa1200_eda_ready.csv`
- `data/features/wa1200_features.csv`

Alcance temporal actual:

- Los datos usados en este proyecto corresponden a una extraccion historica puntual del archivo `TL-20240723-202614 WA1200 #05.csv` (fecha de captura: 23/07/2024).
- Este modelo es una primera version baseline basada en ese corte historico.

## 6. Dataset y variables

- Registros: `1245`
- Features de entrenamiento final: `49`
- Target: `High_Consumption`

Variables relevantes (ejemplos):

- `Velocidad del Motor [rpm]`
- `Por Ciento de Pedal o Palanca del Acelerador [Percent]`
- `Temperatura del Refrigerante del Motor [F]`
- `Presion de Aceite del Mot [psi]`

### Variables clave explicadas para gerencia

| Variable | Que representa en operacion | Alerta de negocio |
| --- | --- | --- |
| `Indice de Combustible Instantaneo [gph]` | Consumo instantaneo de combustible | Si sube sin mayor produccion, aumenta costo por tonelada |
| `Velocidad del Motor [rpm]` | Regimen del motor | RPM altas sostenidas sin carga productiva sugieren ineficiencia |
| `Por Ciento de Pedal o Palanca del Acelerador [Percent]` | Demanda de potencia del operador | Aceleracion alta con bajo avance puede indicar mala condicion operativa |
| `Carga Neta Porcentual [Percent]` | Nivel de carga efectiva del motor | Baja carga con alto consumo es patron de desperdicio |
| `Temperatura del Refrigerante del Motor [F]` | Estado termico del sistema de enfriamiento | Temperatura alta sostenida puede elevar consumo y riesgo de falla |
| `Temperatura de Aceite del Motor [F]` | Estado termico de lubricacion | Sobretemperatura puede degradar eficiencia y vida util |
| `Presion de Aceite del Mot [psi]` | Salud de lubricacion del motor | Fuera de rango puede indicar desgaste o riesgo mecanico |
| `Presion del Riel de Combustible Medi [psi]` | Calidad de alimentacion de combustible | Inestabilidad de presion puede relacionarse con inyectores/sistema de combustible |
| `Temperatura del Aire de Admision del Compresor del Turbocargad [F]` | Eficiencia de admision y turbo | Aire de admision caliente reduce eficiencia de combustion |

### Como se calcula el consumo alto (target)

El modelo usa como base el `Indice de Combustible Instantaneo [gph]` y define el target asi:

1. Se toma la serie de consumo instantaneo `fuel_rate_gph` del dataset.
2. Se calcula el umbral `P90` (percentil 90) de esa serie.
3. Para cada registro:
   - `High_Consumption = 1` si `fuel_rate_gph > P90`
   - `High_Consumption = 0` en caso contrario

En palabras de negocio: el modelo marca como "alto consumo" el 10% de eventos con mayor consumo instantaneo del periodo analizado.

## 7. Entrenamiento de modelos base

Ejecutar:

```powershell
python src/train.py
```

Modelos evaluados:

- Logistic Regression (`class_weight=balanced`)
- Random Forest (`class_weight=balanced`)

Artefactos generados:

- `artifacts/models/baseline_model.joblib`
- `artifacts/meta/feature_columns.joblib`
- `artifacts/meta/baseline_metrics.json`

## 8. Experimentos y seleccion de modelo

Modelo campeon seleccionado: `logreg`.

Metricas del modelo campeon:

| Metrica | Valor |
| --- | ---: |
| Accuracy | 0.7671 |
| Precision (clase 1) | 0.2609 |
| Recall (clase 1) | 0.7200 |
| F1 (clase 1) | 0.3830 |
| ROC AUC | 0.8518 |

Criterio de seleccion: mejor balance para deteccion de clase positiva (`High_Consumption`) en baseline.

## 9. Despliegue del modelo (FastAPI)

Levantar API:

```powershell
uvicorn src.serving:app --host 0.0.0.0 --port 8000
```

Alternativa sin activar entorno:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.serving:app --host 127.0.0.1 --port 8000
```

Documentacion interactiva:

- `http://127.0.0.1:8000/docs`

Endpoints:

- `GET /health`
- `POST /predict`

## 10. Prueba rapida de API

En PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body '{"features":{}}'
```

## 11. Ejemplo de prediccion

Request:

```json
{
  "features": {}
}
```

Response de ejemplo:

```json
{
  "prediction": 0,
  "probability_high_consumption": 0.3213
}
```

Interpretacion:

- `prediction = 0`: no alto consumo.
- `prediction = 1`: alto consumo.
- `probability_high_consumption`: probabilidad estimada de alto consumo.

## 12. Notebook de experimentacion

Notebook principal:

- `notebooks/01_EDA_WA1200.ipynb`

Incluye:

- Analisis exploratorio (EDA)
- Preprocesamiento
- Seleccion de variables
- Entrenamiento de modelos base
- Metricas y exportacion de artefactos

## 13. Conclusiones

- Se implemento un flujo reproducible de punta a punta para clasificacion binaria.
- El modelo baseline permite detectar una parte importante de eventos de alto consumo.
- La API local valida la etapa de serving exigida en el curso.

## 14. Limitaciones

- Dataset limitado a una fuente y periodo especifico.
- Baseline inicial sin ajuste fino de hiperparametros.
- Desbalance de clases en el target.

## 15. Mejoras futuras

- Ajuste de umbral de decision segun objetivo de negocio.
- Mejor ingenieria de variables y seleccion automatica de features.
- Validacion temporal y monitoreo de drift.
- Pipeline CI/CD para entrenamiento y despliegue.

## 16. Escalamiento a tiempo real (Databricks)

Esta version trabaja con un corte historico. Como siguiente etapa, se propone:

1. Conectar telemetria online a Databricks (ingesta continua).
2. Aplicar transformaciones en streaming para generar features en tiempo real.
3. Ejecutar scoring continuo y enviar alertas de alto consumo casi en linea.
4. Monitorear drift, performance del modelo y retraining programado.

Con este enfoque, la solucion pasaria de analisis historico a soporte operativo en tiempo real.

## 17. Lecciones aprendidas

- Separar notebook y scripts mejora reproducibilidad.
- Versionar datos/proceso/modelo en Git facilita trazabilidad.
- Exponer el modelo como API permite uso real por otros sistemas.


