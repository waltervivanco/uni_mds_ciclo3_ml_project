from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "wa1200_eda_ready.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
META_DIR = ARTIFACTS_DIR / "meta"

RANDOM_STATE = 42


def normalize_for_match(value: str) -> str:
    text = str(value).lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def to_num(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
    )
    return pd.to_numeric(s, errors="coerce")


def detect_fuel_col(columns: list[str]) -> str:
    for c in columns:
        norm = normalize_for_match(c)
        if "combustible" in norm and ("instant" in norm or "indice" in norm):
            return c
    for c in columns:
        norm = normalize_for_match(c)
        if "fuel" in norm:
            return c
    raise ValueError("Fuel column not found.")


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    model_df = pd.read_csv(path)

    # Build target if missing (P90 fuel threshold)
    if "High_Consumption" not in model_df.columns:
        fuel_col = detect_fuel_col(list(model_df.columns))
        fuel_num = to_num(model_df[fuel_col])
        if fuel_num.notna().sum() == 0:
            raise ValueError(f"{fuel_col} has no numeric values.")
        model_df[fuel_col] = fuel_num
        threshold = model_df[fuel_col].quantile(0.90)
        model_df["High_Consumption"] = np.where(model_df[fuel_col] > threshold, 1, 0)

    # Convert numeric-like columns
    keep_text_patterns = ["evento", "estado del motor", "tiempo real"]
    text_keep = [
        c for c in model_df.columns if any(p in str(c).lower() for p in keep_text_patterns)
    ]

    for c in model_df.columns:
        if c == "High_Consumption" or c in text_keep:
            continue
        if str(model_df[c].dtype) in ("object", "str"):
            num = to_num(model_df[c])
            if num.notna().mean() >= 0.80:
                model_df[c] = num

    num_df = model_df.select_dtypes(include=np.number).copy()
    y = num_df["High_Consumption"].astype(int)
    X = num_df.drop(columns=["High_Consumption"])

    # Avoid leakage by dropping fuel-like columns
    leak_cols = []
    for c in X.columns:
        norm = normalize_for_match(c)
        if ("combustible" in norm and ("instant" in norm or "indice" in norm)) or "fuel" in norm:
            leak_cols.append(c)
    if leak_cols:
        X = X.drop(columns=sorted(set(leak_cols)), errors="ignore")

    # Drop high-missing columns
    keep_cols = X.columns[X.isna().mean() <= 0.40].tolist()
    X = X[keep_cols]

    # Drop zero-variance columns
    zero_var_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    return X, y


def train_baselines(X: pd.DataFrame, y: pd.Series) -> tuple[str, Pipeline, dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models: dict[str, Pipeline] = {
        "logreg": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    best_name = ""
    best_model: Pipeline | None = None
    best_metrics: dict[str, Any] = {}
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        else:
            auc = float("nan")

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": auc,
        }
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model trained.")

    return best_name, best_model, best_metrics


def save_artifacts(best_name: str, best_model: Pipeline, metrics: dict[str, Any], X: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "baseline_model.joblib")
    joblib.dump(list(X.columns), META_DIR / "feature_columns.joblib")

    meta = {
        "best_model": best_name,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target": "High_Consumption",
        "metrics": metrics,
    }
    with open(META_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    X, y = load_and_prepare_data(DATA_FILE)
    best_name, best_model, best_metrics = train_baselines(X, y)
    save_artifacts(best_name, best_model, best_metrics, X)

    print("Training finished")
    print("Best model:", best_name)
    print("Metrics:", best_metrics)
    print("Saved model:", MODELS_DIR / "baseline_model.joblib")


if __name__ == "__main__":
    main()
