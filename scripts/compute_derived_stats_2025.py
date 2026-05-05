"""
Compute derivable metrics for 2025 weekly stats parquet.

Fields calculated:
  target_share    = player targets / team targets per week
  air_yards_share = player rec_air_yards / team rec_air_yards per week
  racr            = receiving_yards / receiving_air_yards
  wopr            = 1.5 * target_share + 0.7 * air_yards_share
  pacr            = passing_yards / passing_air_yards  (QBs only)

All others (EPA, first downs, fumbles, YAC, 2pt, dakota) require
play-by-play data and remain NaN.
"""

from pathlib import Path
import pandas as pd

WEEKLY_PATH = Path("data/stats/weekly/2025.parquet")


def main() -> None:
    df = pd.read_parquet(WEEKLY_PATH)
    print(f"Loaded {len(df):,} rows")

    # --- team targets per week (REG + POST) ---
    team_week = (
        df.groupby(["recent_team", "season", "week", "season_type"])
        .agg(
            team_targets=("targets", "sum"),
            team_air_yards=("receiving_air_yards", "sum"),
        )
        .reset_index()
    )

    df = df.merge(team_week, on=["recent_team", "season", "week", "season_type"], how="left")

    # target_share
    df["target_share"] = df["targets"] / df["team_targets"].replace(0, float("nan"))

    # air_yards_share
    df["air_yards_share"] = (
        df["receiving_air_yards"] / df["team_air_yards"].replace(0, float("nan"))
    )

    # racr
    df["racr"] = df["receiving_yards"] / df["receiving_air_yards"].replace(0, float("nan"))

    # wopr
    df["wopr"] = 1.5 * df["target_share"] + 0.7 * df["air_yards_share"]

    # pacr (QBs only — divide by zero → NaN naturally)
    df["pacr"] = df["passing_yards"] / df["passing_air_yards"].replace(0, float("nan"))

    df = df.drop(columns=["team_targets", "team_air_yards"])

    # Spot-check a couple players
    sample = df[df["passing_yards"] > 200].head(2)[
        ["player_display_name", "week", "passing_yards", "passing_air_yards", "pacr",
         "targets", "target_share", "wopr"]
    ]
    print("\nQB spot-check:")
    print(sample.to_string(index=False))

    sample2 = df[df["targets"] > 8].head(2)[
        ["player_display_name", "week", "targets", "target_share",
         "receiving_air_yards", "air_yards_share", "racr", "wopr"]
    ]
    print("\nWR/TE spot-check:")
    print(sample2.to_string(index=False))

    print(f"\nNaN rates for computed columns:")
    for col in ["target_share", "air_yards_share", "racr", "wopr", "pacr"]:
        pct = df[col].isna().mean() * 100
        print(f"  {col}: {pct:.1f}% NaN")

    df.to_parquet(WEEKLY_PATH, index=False)
    print(f"\nWrote {WEEKLY_PATH}")


if __name__ == "__main__":
    main()
