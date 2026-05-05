"""
FR-37: Collect the 2026 NFL regular season schedule.

Fetches via nfl_data_py (primary) or nflverse games.parquet (fallback),
expands to one row per team per game appearance, and writes:
    data/schedule/nfl_schedule_2026.parquet

Expanded rows give one RAG chunk per team-game so queries like
"BUF schedule Weeks 1-6" return direct hits rather than requiring
both home and away row inspection.

Run: .venv/bin/python scripts/collect_schedule.py [--auto-confirm]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests


SEASON = 2026
OUT_PATH = Path("data/schedule/nfl_schedule_2026.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NFLVERSE_FALLBACK = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
)

KEEP_COLS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "div_game",
    "roof",
    "surface",
    "stadium",
    "location",
    "spread_line",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
]

AUTO_CONFIRM = "--auto-confirm" in sys.argv


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def fetch_via_nfl_data_py() -> pd.DataFrame:
    print("Trying nfl_data_py.import_schedules …")
    df = nfl.import_schedules([SEASON])
    return df[df["game_type"] == "REG"].copy()


def fetch_via_nflverse_cdn() -> pd.DataFrame:
    print("Trying nflverse CDN (games.parquet) …")
    resp = requests.get(NFLVERSE_FALLBACK, timeout=60)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    return df[(df["season"] == SEASON) & (df["game_type"] == "REG")].copy()


def load_schedule() -> pd.DataFrame:
    for fetcher in (fetch_via_nfl_data_py, fetch_via_nflverse_cdn):
        try:
            df = fetcher()
            if len(df) > 0:
                print(f"  Got {len(df):,} rows")
                return df
            print("  No rows returned — trying next source")
        except Exception as e:
            print(f"  Failed: {e}")

    print(
        "\n"
        "ERROR: 2026 schedule not yet available from any source.\n"
        "\n"
        "The NFL typically publishes the full season schedule in early May.\n"
        "Re-run this script once nfl_data_py / nflverse updates their data:\n"
        "\n"
        "  .venv/bin/pip install --upgrade nfl-data-py\n"
        "  .venv/bin/python scripts/collect_schedule.py\n"
    )
    sys.exit(1)


def expand_to_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game becomes two rows — one from each team's POV.
    This lets embed.py chunk by team so "BUF schedule week 1" is a direct hit.
    """
    cols = [c for c in KEEP_COLS if c in games.columns]
    games = games[cols].copy()

    home = games.copy()
    home["team"] = home["home_team"]
    home["opponent_team"] = home["away_team"]
    home["home_away"] = "home"

    away = games.copy()
    away["team"] = away["away_team"]
    away["opponent_team"] = away["home_team"]
    away["home_away"] = "away"

    df = pd.concat([home, away], ignore_index=True)
    df = df.sort_values(["week", "team"]).reset_index(drop=True)
    df["season"] = SEASON

    return df[[
        "game_id",
        "season",
        "week",
        "gameday",
        "team",
        "opponent_team",
        "home_away",
        "div_game",
        "roof",
        "surface",
        "stadium",
        "location",
        "spread_line",
        "away_qb_name",
        "home_qb_name",
        "away_coach",
        "home_coach",
    ]]


def print_checkpoint(df: pd.DataFrame, games: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: NFL SCHEDULE {SEASON}")
    print(f"{'='*60}")
    print(f"\nUnique games:  {len(games):,}")
    print(f"Expanded rows: {len(df):,}  (one per team per game)")
    print(f"Teams:         {df['team'].nunique()}")
    print(f"Weeks:         {sorted(df['week'].unique())}")
    print(f"\nColumns:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print(f"\nSample (Week 1, first 8 rows):")
    sample = df[df["week"] == df["week"].min()].head(8)
    print(sample[["week", "team", "opponent_team", "home_away", "div_game", "gameday", "stadium"]].to_string(index=False))
    print(f"\nSample spread lines (non-null):")
    spreads = df[df["spread_line"].notna()][["team", "opponent_team", "week", "spread_line"]].head(6)
    if len(spreads):
        print(spreads.to_string(index=False))
    else:
        print("  (none yet — pre-season, spreads appear closer to game day)")


def validate(df: pd.DataFrame, games: pd.DataFrame) -> None:
    errors = []

    if len(games) < 250:
        errors.append(f"Too few games: {len(games)} (expected ~272 for 18-week regular season)")

    team_count = df["team"].nunique()
    if team_count != 32:
        errors.append(f"Expected 32 teams, got {team_count}")

    max_week = df["week"].max()
    if max_week < 18:
        errors.append(f"Expected 18 weeks, only found up to week {max_week}")

    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        confirm("Validation issues found — write anyway?")
    else:
        print("\nValidation passed.")


def main() -> None:
    games = load_schedule()
    df = expand_to_team_rows(games)
    print_checkpoint(df, games)
    validate(df, games)
    confirm(f"Write {len(df):,} rows to {OUT_PATH}?")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows to {OUT_PATH}")
    print(f"\nNext step: re-run scripts/embed.py to include data_source=nfl_schedule_2026")
    print("  Expects ~{} new points in ff-rag-v1".format(len(df)))


if __name__ == "__main__":
    main()
