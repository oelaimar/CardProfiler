"""
preprocessing.py — Data cleaning module.

Usage:
    from src.preprocessing import clean_data
    df_clean = clean_data(df_raw)

"""

from src.extraction import ID_COLUMN, FEATURE_COLUMNS, save_processed
from pathlib import Path
import pandas as pd
from typing import Literal

CLEAN_FILENAME = "df_clean.csv"

# Steps:
#         1. Keep only the expected columns (ID + 7 features).
#         2. Remove exact duplicate rows.
#         3. Remove duplicated client IDs (keep first occurrence).
#         4. Drop the ID column (useless for modelling).
#         5. Handle remaining missing values.
#         6. Save to data/processed/df_clean.csv.


def clean_data(df_raw: pd.DataFrame, nan_strategy: Literal["drop", "median", "mean"] = "drop", save : bool = True) -> pd.DataFrame:
    # Work on a copy:
    df = df_raw.copy()
    n_start = len(df)

    # 1. Keep only expected columns
    df = df[[ID_COLUMN] + FEATURE_COLUMNS]

    # 2. Remove exact duplicate rows.
    df = df.drop_duplicates()

    # 3. Same ID appearing several times with different values
    df = df.drop_duplicates(subset=ID_COLUMN, keep="first")

    # 4. Drop the identifier
    df = df.drop(columns=ID_COLUMN)

    # 5. Missing values
    if nan_strategy == "drop":
        df = df.dropna()
    elif nan_strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif nan_strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("nan_strategy must be 'drop', 'median' or 'mean'")

    # Clean index
    df = df.reset_index(drop=True)

    # 6. Save before any transformation
    if save:
        save_processed(df, CLEAN_FILENAME)

    return df

if __name__ == "__main__":
    pass