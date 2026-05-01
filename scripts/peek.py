"""
Usage:
  python3 scripts/inspect.py data/stats/weekly/2024.parquet
  python3 scripts/inspect.py data/stats/weekly/
  python3 scripts/inspect.py data/stats/weekly/ --search player_display_name
  python3 scripts/inspect.py data/stats/weekly/2024.parquet --player "Tyreek Hill"
"""

import sys
import argparse
from pathlib import Path
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            print(f"No Parquet files found in {path}")
            sys.exit(1)
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def inspect(df: pd.DataFrame, path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"PATH:  {path}")
    print(f"SHAPE: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"{'='*60}")

    print("\nSCHEMA:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isna().sum()
        null_pct = f"{nulls / len(df) * 100:.1f}%" if len(df) else "n/a"
        print(f"  {col:<45} {str(dtype):<12} nulls: {null_pct}")

    if "season" in df.columns:
        print(f"\nROW COUNTS BY SEASON:")
        print(df.groupby("season").size().to_string())

    if "position" in df.columns:
        print(f"\nROW COUNTS BY POSITION:")
        print(df.groupby("position").size().to_string())

    print(f"\nSAMPLE (3 rows):")
    print(df.head(3).to_string())


def search_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        close = [c for c in df.columns if col.lower() in c.lower()]
        print(f"Column '{col}' not found.")
        if close:
            print(f"Did you mean: {close}")
        return
    print(f"\nColumn '{col}' — sample values:")
    print(df[col].dropna().unique()[:20])


def filter_player(df: pd.DataFrame, name: str) -> None:
    name_cols = [c for c in df.columns if "name" in c.lower()]
    if not name_cols:
        print("No name columns found.")
        return
    col = name_cols[0]
    matches = df[df[col].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f"No rows found for '{name}'")
        return
    print(f"\nRows matching '{name}' ({len(matches)} found):")
    print(matches.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--search", help="Check if a column exists and show sample values")
    parser.add_argument("--player", help="Filter rows by player name")
    args = parser.parse_args()

    df = load(args.path)

    if args.search:
        search_column(df, args.search)
    elif args.player:
        filter_player(df, args.player)
    else:
        inspect(df, args.path)


if __name__ == "__main__":
    main()
