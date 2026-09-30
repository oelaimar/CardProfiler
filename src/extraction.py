"""
extraction.py — Data loading module.

Usage:
    from src.extraction import load_raw
    df = load_raw()
"""

from pathlib import Path
import pandas as pd

# paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


DEFAULT_RAW_FILE = RAW_DATA_DIR / "dataset.csv"

# expected schema
ID_COLUMN = "CUST_ID"

FEATURE_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
]

EXPECTED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS

# function
def load_raw(path : str | Path = DEFAULT_RAW_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"put the raw dataset in {RAW_DATA_DIR}"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def save_processed(df : pd.DataFrame, filename: str) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path

def load_processed(filename: str) -> pd.DataFrame:
    path = Path(PROCESSED_DATA_DIR / filename)
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)

if __name__ == "__main__" :
    pass