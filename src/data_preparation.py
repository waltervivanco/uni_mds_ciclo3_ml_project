from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "TL-20240723-202614 WA1200 #05.csv"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "wa1200_eda_ready.csv"


def fix_text(value: str) -> str:
    text = str(value)
    replacements = {
        "Mi¿½ltiple": "Multiple",
        "Mï¿½ltiple": "Multiple",
        "M�ltiple": "Multiple",
        "Operaci�n": "Operacion",
        "Reconstrucci�n": "Reconstruccion",
        "Presi�n": "Presion",
        "C�rter": "Carter",
        "Barom�trica": "Barometrica",
        "Admisi�n": "Admision",
        "�ndice": "Indice",
        "Bater�": "Bateria",
        "Sincronizaci": "Sincronizacion",
        "�F": "F",
        "Â°F": "F",
        "ï¿½": "",
        "�": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(value: str) -> str:
    text = fix_text(value).lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def merge_name_unit(name: str, unit: str) -> str:
    name = fix_text(name)
    unit = fix_text(unit)

    if unit == "" or unit.lower() == "unidades":
        col = name
    elif name == "":
        col = f"[{unit}]"
    else:
        col = f"{name} [{unit}]"

    if "temperatura" in normalize_for_match(col):
        if re.search(r"\[[^\]]*\]", col):
            col = re.sub(r"\[[^\]]*\]", "[F]", col)
        else:
            col = f"{col} [F]"

    return col


def prepare_dataframe(raw_file: Path) -> pd.DataFrame:
    with open(raw_file, "r", encoding="utf-8", errors="replace") as f:
        rows = [line.rstrip("\n\r").split(";") for line in f]

    if len(rows) < 21:
        raise ValueError("Unexpected CSV structure. Expected at least 21 rows.")

    vars_row = [str(x).strip() for x in rows[18]]
    units_row = [str(x).strip() for x in rows[19]]

    n = max(len(vars_row), len(units_row))
    vars_row += [""] * (n - len(vars_row))
    units_row += [""] * (n - len(units_row))

    cols = [merge_name_unit(v, u) for v, u in zip(vars_row, units_row)]
    data_rows = []
    for row in rows[20:]:
        row = [str(x).strip() for x in row]
        row = (row + [""] * n)[:n]
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=cols)

    keep_cols: list[str] = []
    for col in df.columns:
        name_empty = str(col).strip() == ""
        col_empty = df[col].astype(str).str.strip().eq("").all()
        if not (name_empty and col_empty):
            keep_cols.append(col)
    df = df[keep_cols]

    df = df[df.astype(str).apply(lambda r: r.str.strip().ne("").any(), axis=1)].reset_index(drop=True)

    if not df.empty and df.iloc[0].astype(str).str.lower().str.contains("unidades").any():
        df = df.iloc[1:].reset_index(drop=True)

    return df


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    df = prepare_dataframe(RAW_FILE)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved: {PROCESSED_FILE}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
