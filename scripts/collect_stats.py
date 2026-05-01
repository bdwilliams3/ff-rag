"""
FR-6: nflverse backfill — weekly and seasonal player stats 2012-2024.

Pulls data, prints schema + sample for owner approval, then writes Parquet.
Run with --auto-confirm to skip interactive prompts (used for unattended runs).
"""

import sys
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2012, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]

SEASONAL_DIR = Path("data/stats/seasonal")
WEEKLY_DIR = Path("data/stats/weekly")

SEASONAL_DIR.mkdir(parents=True, exist_ok=True)
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CONFIRM = "--auto-confirm" in sys.argv


def print_checkpoint(label: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {label}")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nSchema:\n{df.dtypes.to_string()}")
    print(f"\nSample (5 rows):\n{df.head(5).to_string()}")
    if "season" in df.columns:
        print(f"\nRow counts by season:\n{df.groupby('season').size().to_string()}")
    print(f"\nplayer_id present: {'player_id' in df.columns}")
    if "player_id" in df.columns:
        null_ids = df["player_id"].isna().sum()
        print(f"Null player_ids: {null_ids} ({null_ids / len(df) * 100:.1f}%)")


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"\n[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted. Review the schema above and re-run when ready.")
        sys.exit(0)


def fetch_years(fn, label: str) -> pd.DataFrame:
    """Pull year-by-year, skip any year nflverse hasn't published yet."""
    frames = []
    for yr in YEARS:
        try:
            frames.append(fn([yr]))
            print(f"  {yr} OK")
        except Exception as e:
            print(f"  {yr} skipped ({e})")
    if not frames:
        raise RuntimeError(f"No data fetched for {label}")
    return pd.concat(frames, ignore_index=True)


def collect_weekly() -> pd.DataFrame:
    print("\nPulling weekly stats year-by-year (skips unpublished seasons)...")
    df = fetch_years(nfl.import_weekly_data, "weekly")
    df = df[df["position"].isin(POSITIONS)].reset_index(drop=True)
    return df


def collect_seasonal(weekly: pd.DataFrame) -> pd.DataFrame:
    # Seasonal data has no position column — derive skill position player_ids from weekly
    print("\nPulling seasonal stats year-by-year...")
    skill_ids = set(weekly["player_id"].unique())
    player_meta = (
        weekly[["player_id", "player_display_name", "position", "position_group"]]
        .drop_duplicates("player_id")
    )

    df = fetch_years(nfl.import_seasonal_data, "seasonal")
    df = df[df["player_id"].isin(skill_ids)].reset_index(drop=True)
    df = df.merge(player_meta, on="player_id", how="left")
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
    # Weekly first — used to derive position filter for seasonal
    weekly = collect_weekly()
    print_checkpoint("WEEKLY STATS", weekly)
    confirm("Schema looks good — write weekly Parquet files?")
    write_weekly(weekly)

    seasonal = collect_seasonal(weekly)
    print_checkpoint("SEASONAL STATS", seasonal)
    confirm("Schema looks good — write seasonal Parquet files?")
    write_seasonal(seasonal)

    print("\nDone. Review data/stats/ and close FR-6 once satisfied.")


if __name__ == "__main__":
    main()
