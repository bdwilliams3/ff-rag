"""
FR-21: Player injury history — weekly designations via nfl_data_py.
Pulls injury reports (practice + game status) for skill positions 2012-2024.
Run: .venv/bin/python3 scripts/collect_injuries.py [--auto-confirm]
"""

import sys
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2012, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]
OUT_DIR = Path("data/injuries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CONFIRM = "--auto-confirm" in sys.argv


def fetch_years() -> pd.DataFrame:
    frames = []
    for yr in YEARS:
        try:
            df = nfl.import_injuries(years=[yr])
            frames.append(df)
            print(f"  {yr}: {len(df):,} rows")
        except Exception as e:
            print(f"  {yr} skipped ({e})")
    if not frames:
        raise RuntimeError("No injury data fetched")
    return pd.concat(frames, ignore_index=True)


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def main() -> None:
    print("Pulling injury data 2012-2025 (year-by-year)...")
    df = fetch_years()

    df = df[df["position"].isin(POSITIONS)].copy()
    df = df.rename(columns={"gsis_id": "player_id"})

    print(f"\n{'='*60}")
    print("CHECKPOINT: INJURY DATA")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nSchema:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isna().sum()
        print(f"  {col:<35} {str(dtype):<12} nulls: {nulls/len(df)*100:.1f}%")
    print(f"\nRow counts by season:")
    print(df.groupby("season").size().to_string())
    print(f"\nSample (5 rows):")
    print(df[["player_id", "full_name", "position", "season", "week",
              "report_primary_injury", "report_status"]].head(5).to_string())

    confirm("Schema looks good — write Parquet files?")

    for year, group in df.groupby("season"):
        path = OUT_DIR / f"{int(year)}.parquet"
        group.reset_index(drop=True).to_parquet(path, index=False)

    print(f"\nWrote {df['season'].nunique()} files to {OUT_DIR}/")
    print("Done. Review data/injuries/ and close FR-21 once satisfied.")


if __name__ == "__main__":
    main()
