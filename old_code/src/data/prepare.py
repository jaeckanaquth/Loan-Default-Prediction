# src/data/prepare.py
"""
Prepare pipeline for Loan Default dataset.

Expectations:
- Raw CSV(s) are already placed under: src/data/raw/
  (e.g. src/data/raw/Dataset.csv and src/data/raw/Data_Dictionary.csv)
- This script creates: data/processed/loan.csv

What it does:
1. Finds the largest CSV in src/data/raw and loads it.
2. Detects a target column (common names or text statuses) and maps it to binary 0/1.
3. Selects a sensible set of features:
   - numeric columns (excluding obvious ID-like cols)
   - top categorical columns (factorized) if numeric features are few
4. Imputes numeric cols with median.
5. Drops rows without a target value.
6. Writes out data/processed/loan.csv and prints a short summary.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import sys

RAW_DIR = Path("src/data/raw")
PROC_DIR = Path("src/data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Config
MAX_NUM_FEATURES = 12
MAX_CAT_FEATURES = 6
DROP_MISSING_TARGET_THRESHOLD = 0.5  # if >50% missing target, warn but still proceed


def find_largest_csv(folder: Path) -> Path:
    csvs = list(folder.rglob("*.csv")) + list(folder.rglob("*.CSV"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder.resolve()}")
    csvs = sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0]


def detect_target_column(df: pd.DataFrame):
    # obvious candidate names
    candidates = [c for c in df.columns if c.lower() in (
        "target", "default", "loan_default", "is_default", "default_ind", "loan_status", "status", "loan_status_desc")]
    if candidates:
        return candidates[0]

    # look for binary-like columns
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) <= 3:
            sval = set([str(v).strip().lower() for v in vals])
            if sval.issubset({"0", "1", "yes", "no", "y", "n", "true", "false"}) or sval.issubset({"paid", "charged off", "current", "fully paid", "default", "late"}):
                return c

    # fallback None
    return None


def map_target_to_binary(series: pd.Series):
    # If numeric 0/1 already
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()
    # text mapping rules
    mapping = {}
    def mapper(x):
        if pd.isna(x):
            return np.nan
        sx = str(x).strip().lower()
        if sx in ("1", "yes", "y", "true", "t"):
            return 1
        if sx in ("0", "no", "n", "false", "f"):
            return 0
        if any(substr in sx for substr in ("charge", "charged off", "default", "delinquent", "late")):
            return 1
        if any(substr in sx for substr in ("fully paid", "paid", "current", "good", "completed")):
            return 0
        return np.nan

    return s.map(mapper)


def sensible_feature_selection(df: pd.DataFrame, target_col: str):
    # remove obvious id columns: those containing 'id' or 'account' or 'no.' etc.
    id_like = [c for c in df.columns if re.search(r"\b(id|account|acc|no\.?|accountno)\b", c.lower())]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != target_col and c not in id_like]

    # drop columns with too many missing values
    numeric = [c for c in numeric if df[c].isna().mean() < 0.8]

    selected = numeric[:MAX_NUM_FEATURES]

    if len(selected) < 5:
        # add categorical factorized columns by frequency
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cats = [c for c in cats if c != target_col and c not in id_like]
        # sort by number of non-null values / unique ratio
        cats = sorted(cats, key=lambda x: (df[x].notna().sum(), -df[x].nunique()), reverse=True)
        for c in cats[:MAX_CAT_FEATURES]:
            # create factorized column
            newc = f"{c}_enc"
            df[newc] = pd.factorize(df[c].fillna("##NA##"))[0]
            selected.append(newc)

    return df, selected


def prepare(save_path: Path = PROC_DIR / "loan.csv"):
    print("Raw dir:", RAW_DIR.resolve())
    csv_path = find_largest_csv(RAW_DIR)
    print("Using CSV:", csv_path.name)
    # try multiple encodings
    encodings = ["utf-8", "latin1", "cp1252"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            print(f"Loaded with encoding={enc}; shape={df.shape}")
            break
        except Exception as e:
            print(f"Failed encoding {enc}: {e}")
    if df is None:
        raise RuntimeError("Unable to read CSV with common encodings.")

    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    target_col = detect_target_column(df)
    if target_col:
        print("Detected target column:", target_col)
        df["target"] = map_target_to_binary(df[target_col])
    else:
        print("No obvious target column found. Creating 'target' as NaN (you must supply target).")
        df["target"] = np.nan

    missing_target_pct = df["target"].isna().mean()
    print(f"Target missing fraction: {missing_target_pct:.3f}")
    if missing_target_pct > DROP_MISSING_TARGET_THRESHOLD:
        print("WARNING: large fraction of target is missing. You may need to provide the labeled file or map a different column.")

    # feature selection
    df, selected_features = sensible_feature_selection(df, target_col="target")
    print("Selected feature columns:", selected_features)

    # impute numeric features
    for c in selected_features:
        if pd.api.types.is_numeric_dtype(df[c]):
            median = df[c].median()
            df[c] = df[c].fillna(median)
        else:
            # if not numeric, ensure numeric encoding exists (attempt factorize)
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            except Exception:
                df[c] = pd.factorize(df[c].fillna("##NA##"))[0]

    processed = df[selected_features + ["target"]].copy()
    before = processed.shape[0]
    processed = processed.dropna(subset=["target"])
    after = processed.shape[0]
    dropped = before - after
    print(f"Dropped {dropped} rows with missing target. Remaining rows: {after}")

    # Save processed CSV
    save_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(save_path, index=False)
    print("Saved processed data to:", save_path.resolve())

    # Save metadata (columns used) for reproducibility
    meta = {
        "raw_csv": str(csv_path.resolve()),
        "n_rows_raw": int(df.shape[0]),
        "n_rows_processed": int(after),
        "selected_features": selected_features,
        "target_column_used": target_col,
    }
    with open(save_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote metadata:", save_path.with_suffix(".meta.json").resolve())

    return processed


if __name__ == "__main__":
    out = prepare()
    # print small preview if run interactively
    print("\nPreview of processed data:")
    print(out.head().to_string(index=False))
