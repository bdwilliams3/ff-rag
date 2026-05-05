"""
FR-25: 2025 NFL player stats from Sleeper API.

nflverse has not published 2025 data. This script pulls from Sleeper's
free stats API (no key required) for weeks 1-18 (REG) and 1-4 (POST).

Outputs match the existing weekly/seasonal parquet schema from collect_stats.py.
Derived metrics (EPA, RACR, WOPR, target_share, etc.) are left as NaN — to be
calculated in a follow-up pass.

Usage:
    .venv/bin/python3 scripts/collect_stats_2025.py
    .venv/bin/python3 scripts/collect_stats_2025.py --auto-confirm
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.player_resolver import PlayerResolver

YEAR = 2025
REG_WEEKS = range(1, 19)   # weeks 1-18
POST_WEEKS = range(1, 5)   # wild card through super bowl
POSITIONS = {"QB", "RB", "WR", "TE"}

WEEKLY_DIR = Path("data/stats/weekly")
SEASONAL_DIR = Path("data/stats/seasonal")
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
SEASONAL_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CONFIRM = "--auto-confirm" in sys.argv

SLEEPER_BASE = "https://api.sleeper.app/v1"

# Sleeper stat key → nflverse column name
STAT_MAP = {
    "pass_cmp":      "completions",
    "pass_att":      "attempts",
    "pass_yd":       "passing_yards",
    "pass_td":       "passing_tds",
    "pass_int":      "interceptions",
    "pass_sack":     "sacks",
    "pass_sack_yds": "sack_yards",
    "pass_air_yd":   "passing_air_yards",
    "rush_att":      "carries",
    "rush_yd":       "rushing_yards",
    "rush_td":       "rushing_tds",
    "rec":           "receptions",
    "rec_tgt":       "targets",
    "rec_yd":        "receiving_yards",
    "rec_td":        "receiving_tds",
    "rec_air_yd":    "receiving_air_yards",
    "pts_std":       "fantasy_points",
    "pts_ppr":       "fantasy_points_ppr",
}

# Columns present in nflverse output but not derivable from Sleeper
NAN_COLS = [
    "sack_fumbles", "sack_fumbles_lost",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa",
    "passing_2pt_conversions", "pacr", "dakota",
    "rushing_fumbles", "rushing_fumbles_lost", "rushing_first_downs",
    "rushing_epa", "rushing_2pt_conversions",
    "receiving_fumbles", "receiving_fumbles_lost",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share", "wopr",
    "special_teams_tds", "opponent_team",
]

# Final column order matching existing parquet schema
OUTPUT_COLS = [
    "player_id", "player_name", "player_display_name", "position", "position_group",
    "headshot_url", "recent_team", "season", "week", "season_type", "opponent_team",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
    "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
    "passing_epa", "passing_2pt_conversions", "pacr", "dakota",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share", "wopr",
    "special_teams_tds", "fantasy_points", "fantasy_points_ppr",
]


def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def load_sleeper_players() -> dict:
    """Returns {sleeper_id: {full_name, position, team, gsis_id}}."""
    print("Fetching Sleeper player roster...", end=" ", flush=True)
    raw = fetch_json(f"{SLEEPER_BASE}/players/nfl")
    players = {}
    for pid, p in raw.items():
        if p.get("position") not in POSITIONS:
            continue
        players[pid] = {
            "full_name": p.get("full_name") or "",
            "position":  p.get("position") or "",
            "team":      p.get("team") or "",
            "gsis_id":   p.get("gsis_id") or None,
        }
    print(f"{len(players)} skill players loaded")
    return players


def fetch_week(season_type: str, week: int) -> dict:
    """Returns {sleeper_id: stats_dict} for one week."""
    url = f"{SLEEPER_BASE}/stats/nfl/{season_type}/{YEAR}/{week}"
    try:
        return fetch_json(url)
    except Exception as e:
        print(f"  week {week} ({season_type}) skipped: {e}")
        return {}


def build_weekly_rows(
    sleeper_players: dict,
    resolver: PlayerResolver,
) -> list[dict]:
    weeks_to_fetch = (
        [("regular", w) for w in REG_WEEKS] +
        [("post", w) for w in POST_WEEKS]
    )

    all_rows = []
    for season_type, week in weeks_to_fetch:
        raw = fetch_week(season_type, week)
        if not raw:
            continue

        season_label = "REG" if season_type == "regular" else "POST"
        week_rows = []

        for sleeper_id, stats in raw.items():
            if sleeper_id.startswith("TEAM"):
                continue

            meta = sleeper_players.get(sleeper_id)
            if not meta:
                continue

            # Skip rows with no offensive activity (practice squad, inactive)
            has_activity = any(stats.get(k, 0) for k in [
                "pass_yd", "rush_yd", "rec_yd", "pass_td", "rush_td", "rec_td",
                "rec_tgt", "pass_att", "rush_att",
            ])
            if not has_activity:
                continue

            # Resolve to nflverse player_id
            gsis_id = meta["gsis_id"]
            if not gsis_id:
                gsis_id, _ = resolver.resolve({
                    "name":     meta["full_name"],
                    "position": meta["position"],
                })

            if not gsis_id:
                continue

            row = {
                "player_id":          gsis_id,
                "player_name":        _short_name(meta["full_name"]),
                "player_display_name": meta["full_name"],
                "position":           meta["position"],
                "position_group":     meta["position"],
                "headshot_url":       None,
                "recent_team":        meta["team"] or None,
                "season":             YEAR,
                "week":               week,
                "season_type":        season_label,
            }

            for sleeper_key, nfl_col in STAT_MAP.items():
                row[nfl_col] = stats.get(sleeper_key, None)

            for col in NAN_COLS:
                row.setdefault(col, None)

            week_rows.append(row)

        has_stats = sum(
            1 for r in week_rows
            if any(r.get(c) for c in ["passing_yards", "rushing_yards", "receiving_yards"])
        )
        print(f"  {season_label} week {week:>2}: {len(week_rows)} players resolved, {has_stats} with offensive stats")
        all_rows.extend(week_rows)
        time.sleep(0.2)

    return all_rows


def _short_name(full_name: str) -> str:
    parts = full_name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}.{parts[-1]}"
    return full_name


def build_seasonal(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly to seasonal (REG only, matching nflverse convention)."""
    reg = weekly[weekly["season_type"] == "REG"].copy()

    # Columns that sum cleanly
    sum_cols = [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost", "passing_air_yards",
        "passing_yards_after_catch", "passing_first_downs", "passing_2pt_conversions",
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
        "rushing_fumbles_lost", "rushing_first_downs", "rushing_2pt_conversions",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
        "receiving_yards_after_catch", "receiving_first_downs", "receiving_2pt_conversions",
        "special_teams_tds", "fantasy_points", "fantasy_points_ppr",
    ]
    # Only sum columns that actually exist
    sum_cols = [c for c in sum_cols if c in reg.columns]

    agg = reg.groupby("player_id")[sum_cols].sum().reset_index()

    meta = (
        reg.groupby("player_id")
        .agg(
            player_name=("player_name", "last"),
            player_display_name=("player_display_name", "last"),
            position=("position", "last"),
            position_group=("position_group", "last"),
            recent_team=("recent_team", "last"),
            season=("season", "last"),
            games=("week", "count"),
        )
        .reset_index()
    )

    seasonal = meta.merge(agg, on="player_id", how="left")
    return seasonal


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def main() -> None:
    sleeper_players = load_sleeper_players()
    resolver = PlayerResolver(min_confidence=0.35)

    print(f"\nFetching {YEAR} weekly stats...")
    rows = build_weekly_rows(sleeper_players, resolver)

    if not rows:
        print("No data collected. Exiting.")
        sys.exit(1)

    weekly = pd.DataFrame(rows)
    # Ensure column order matches existing parquet schema
    for col in OUTPUT_COLS:
        if col not in weekly.columns:
            weekly[col] = None
    weekly = weekly[OUTPUT_COLS]

    print(f"\nWeekly: {len(weekly):,} rows, {weekly['player_id'].notna().sum():,} with player_id")
    print(f"REG rows: {(weekly['season_type']=='REG').sum()}")
    print(f"POST rows: {(weekly['season_type']=='POST').sum()}")
    print(f"Positions: {weekly['position'].value_counts().to_dict()}")

    unmatched = resolver.flush_unmatched()
    if unmatched:
        print(f"Unmatched players written to data/unmatched/")

    confirm(f"Write data/stats/weekly/{YEAR}.parquet?")
    weekly.to_parquet(WEEKLY_DIR / f"{YEAR}.parquet", index=False)
    print(f"Wrote {WEEKLY_DIR}/{YEAR}.parquet")

    seasonal = build_seasonal(weekly)
    print(f"\nSeasonal: {len(seasonal):,} players")

    confirm(f"Write data/stats/seasonal/{YEAR}.parquet?")
    seasonal.to_parquet(SEASONAL_DIR / f"{YEAR}.parquet", index=False)
    print(f"Wrote {SEASONAL_DIR}/{YEAR}.parquet")

    print("\nDone. NaN columns to backfill: EPA, RACR, WOPR, target_share, air_yards_share.")


if __name__ == "__main__":
    main()
