"""
FR-6: nflverse backfill — weekly and seasonal player stats 2012-2024.

Pulls data, prints schema + sample for owner approval, then writes Parquet.
Script pauses before writing — do not proceed until schema is reviewed.
"""

import sys
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2012, 2025))
POSITIONS = ["QB", "RB", "WR", "TE"]

SEASONAL_DIR = Path("data/stats/seasonal")
WEEKLY_DIR = Path("data/stats/weekly")

SEASONAL_DIR.mkdir(parents=True, exist_ok=True)
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)


def print_checkpoint(label: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {label}")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nSchema:\n{df.dtypes.to_string()}")
    print(f"\nSample (5 rows):\n{df.head(5).to_string()}")
    print(f"\nRow counts by season:")
    if "season" in df.columns:
        print(df.groupby("season").size().to_string())
    print(f"\nnfl_verse player_id present: {'player_id' in df.columns}")
    if "player_id" in df.columns:
        null_ids = df["player_id"].isna().sum()
        print(f"Null player_ids: {null_ids} ({null_ids/len(df)*100:.1f}%)")


def confirm(prompt: str) -> None:
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted. Review the schema above and re-run when ready.")
        sys.exit(0)


def collect_seasonal() -> pd.DataFrame:
    print("\nPulling seasonal stats 2012-2024...")
    df = nfl.import_seasonal_data(years=YEARS)
    df = df[df["position"].isin(POSITIONS)].reset_index(drop=True)
    return df


def collect_weekly() -> pd.DataFrame:
    print("\nPulling weekly stats 2012-2024 (this takes a minute)...")
    df = nfl.import_weekly_data(years=YEARS)
    df = df[df["position"].isin(POSITIONS)].reset_index(drop=True)
    return df


def write_seasonal(df: pd.DataFrame) -> None:
    for year, group in df.groupby("season"):
        path = SEASONAL_DIR / f"{year}.parquet"
        group.reset_index(drop=True).to_parquet(path, index=False)
    print(f"\nWrote {df['season'].nunique()} seasonal Parquet files to {SEASONAL_DIR}/")


def write_weekly(df: pd.DataFrame) -> None:
    for year, group in df.groupby("season"):
        path = WEEKLY_DIR / f"{year}.parquet"
        group.reset_index(drop=True).to_parquet(path, index=False)
    print(f"Wrote {df['season'].nunique()} weekly Parquet files to {WEEKLY_DIR}/")


def main() -> None:
    # --- Seasonal ---
    seasonal = collect_seasonal()
    print_checkpoint("SEASONAL STATS", seasonal)
    confirm("Schema looks good — write seasonal Parquet files?")
    write_seasonal(seasonal)

    # --- Weekly ---
    weekly = collect_weekly()
    print_checkpoint("WEEKLY STATS", weekly)
    confirm("Schema looks good — write weekly Parquet files?")
    write_weekly(weekly)

    print("\nDone. Review data/stats/ and close FR-6 once satisfied.")


if __name__ == "__main__":
    main()
