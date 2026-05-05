"""
FR-27: Historical NFL player salaries and salary cap context.

Pulls nflverse historical contract data sourced from OverTheCap, flattens the
nested annual cap/cash rows, aggregates to one row per player-season, and writes:

data/salaries/player_salaries.parquet
data/salaries/salary_cap_by_season.parquet

Run:
  .venv/bin/python3 scripts/collect_salaries.py [--auto-confirm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import nfl_data_py as nfl
import numpy as np
import pandas as pd


YEARS = list(range(2012, 2026))
OUT_DIR = Path("data/salaries")
PLAYER_OUT_PATH = OUT_DIR / "player_salaries.parquet"
CAP_OUT_PATH = OUT_DIR / "salary_cap_by_season.parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base NFL salary cap by league year, in millions of dollars. These are useful
# for normalizing contract value across eras; OTC also carries per-row cap_percent.
OFFICIAL_BASE_CAP_MILLIONS = {
    2012: 120.600,
    2013: 123.000,
    2014: 133.000,
    2015: 143.280,
    2016: 155.270,
    2017: 167.000,
    2018: 177.200,
    2019: 188.200,
    2020: 198.200,
    2021: 182.500,
    2022: 208.200,
    2023: 224.800,
    2024: 255.400,
    2025: 279.200,
}

MONEY_COLS = [
    "base_salary",
    "prorated_bonus",
    "roster_bonus",
    "guaranteed_salary",
    "cap_number",
    "cash_paid",
    "workout_bonus",
    "other_bonus",
    "per_game_roster_bonus",
    "option_bonus",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--auto-confirm", action="store_true")
    p.add_argument("--start-year", type=int, default=min(YEARS))
    p.add_argument("--end-year", type=int, default=max(YEARS))
    return p.parse_args()


def confirm(prompt: str, auto_confirm: bool) -> None:
    if auto_confirm:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def iter_annual_rows(raw: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in raw.itertuples(index=False):
        annual = getattr(row, "cols")
        if not isinstance(annual, np.ndarray):
            continue
        for item in annual.tolist():
            if not isinstance(item, dict) or not str(item.get("year", "")).isdigit():
                continue
            rows.append({
                "player_id": row.gsis_id,
                "player_name": row.player,
                "position": row.position,
                "otc_id": row.otc_id,
                "source_url": row.player_page,
                **item,
            })
    return rows


def clean_annual_rows(raw: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    annual = pd.DataFrame(iter_annual_rows(raw))
    if annual.empty:
        raise RuntimeError("No annual salary rows found in nflverse contract data.")

    annual = annual.rename(columns={"year": "season"})
    annual["season"] = annual["season"].astype(int)
    annual = annual[annual["season"].between(start_year, end_year)].copy()

    for col in MONEY_COLS + ["cap_percent"]:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

    annual["source"] = "nflverse_overthecap"
    annual["salary_units"] = "millions_usd"
    annual["source_grain"] = "player_season_team"

    # The raw contract table has one nested annual cap table per contract-history
    # row, so the same annual row appears many times. Keep one copy first.
    dedupe_cols = [
        "player_id", "player_name", "position", "otc_id", "season", "team",
        "source_url", *MONEY_COLS, "cap_percent",
    ]
    annual = annual.drop_duplicates(dedupe_cols)

    # If OTC still has multiple lines for the same player-season-team, prefer the
    # row carrying the largest cap/cash signal. Traded-player multi-team seasons
    # are kept and aggregated later.
    annual["_rank_cap"] = annual["cap_number"].fillna(-1)
    annual["_rank_cash"] = annual["cash_paid"].fillna(-1)
    annual = (
        annual.sort_values(["otc_id", "season", "team", "_rank_cap", "_rank_cash"])
        .drop_duplicates(["otc_id", "season", "team"], keep="last")
        .drop(columns=["_rank_cap", "_rank_cash"])
    )
    return annual


def collapse_teams(values: pd.Series) -> str | None:
    teams = [str(v) for v in values.dropna().unique() if str(v)]
    return "/".join(sorted(teams)) if teams else None


def first_non_null(values: pd.Series) -> Any:
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def aggregate_player_seasons(annual: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        annual.groupby(["otc_id", "season"], as_index=False)
        .agg({
            "player_id": first_non_null,
            "player_name": first_non_null,
            "position": first_non_null,
            "team": collapse_teams,
            "source_url": first_non_null,
            "source": first_non_null,
            "salary_units": first_non_null,
            **{col: "sum" for col in MONEY_COLS},
            "cap_percent": "sum",
        })
    )
    grouped = grouped.rename(columns={
        "base_salary": "base_salary_millions",
        "prorated_bonus": "prorated_bonus_millions",
        "roster_bonus": "roster_bonus_millions",
        "guaranteed_salary": "guaranteed_salary_millions",
        "cap_number": "cap_number_millions",
        "cash_paid": "cash_paid_millions",
        "workout_bonus": "workout_bonus_millions",
        "other_bonus": "other_bonus_millions",
        "per_game_roster_bonus": "per_game_roster_bonus_millions",
        "option_bonus": "option_bonus_millions",
        "cap_percent": "cap_percent_of_league_cap",
    })
    grouped["source_grain"] = "player_season"
    grouped["official_base_salary_cap_millions"] = grouped["season"].map(OFFICIAL_BASE_CAP_MILLIONS)
    grouped["cap_number_pct_official_base_cap"] = (
        grouped["cap_number_millions"] / grouped["official_base_salary_cap_millions"]
    )
    grouped["cash_paid_pct_official_base_cap"] = (
        grouped["cash_paid_millions"] / grouped["official_base_salary_cap_millions"]
    )

    ordered = [
        "player_id", "player_name", "position", "season", "team", "otc_id",
        "base_salary_millions", "guaranteed_salary_millions", "cap_number_millions",
        "cash_paid_millions", "cap_percent_of_league_cap",
        "cap_number_pct_official_base_cap", "cash_paid_pct_official_base_cap",
        "prorated_bonus_millions", "roster_bonus_millions", "workout_bonus_millions",
        "other_bonus_millions", "per_game_roster_bonus_millions", "option_bonus_millions",
        "official_base_salary_cap_millions", "source", "source_grain", "salary_units",
        "source_url",
    ]
    return grouped[ordered].sort_values(["season", "player_name", "otc_id"]).reset_index(drop=True)


def build_salary_cap_table(player_seasons: pd.DataFrame) -> pd.DataFrame:
    cap = (
        player_seasons.groupby("season", as_index=False)
        .agg(
            official_base_salary_cap_millions=("official_base_salary_cap_millions", "first"),
            player_rows=("otc_id", "count"),
            total_player_cap_number_millions=("cap_number_millions", "sum"),
            total_player_cash_paid_millions=("cash_paid_millions", "sum"),
            max_player_cap_number_millions=("cap_number_millions", "max"),
            max_player_cash_paid_millions=("cash_paid_millions", "max"),
        )
    )
    cap["source"] = "nfl_official_base_cap_plus_nflverse_overthecap_player_totals"
    cap["salary_units"] = "millions_usd"
    return cap


def print_checkpoint(player_seasons: pd.DataFrame, salary_cap: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print("CHECKPOINT: PLAYER SALARIES")
    print(f"{'=' * 60}")
    print(f"\nPlayer salary rows: {len(player_seasons):,}")
    print(f"Salary cap rows:    {len(salary_cap):,}")
    print(f"\nSchema:")
    for col, dtype in player_seasons.dtypes.items():
        nulls = player_seasons[col].isna().sum()
        print(f"  {col:<42} {str(dtype):<12} nulls: {nulls / len(player_seasons) * 100:.1f}%")

    print("\nRows by season:")
    print(player_seasons.groupby("season").size().to_string())

    print("\nTop 20 player-season cap numbers:")
    cols = [
        "season", "player_name", "position", "team", "cap_number_millions",
        "cash_paid_millions", "cap_number_pct_official_base_cap",
    ]
    print(
        player_seasons.sort_values("cap_number_millions", ascending=False)
        .head(20)[cols]
        .to_string(index=False)
    )

    print("\nSalary cap table:")
    print(salary_cap.to_string(index=False))


def main() -> None:
    args = parse_args()
    print("Pulling nflverse historical contracts from OverTheCap...")
    raw = nfl.import_contracts()
    annual = clean_annual_rows(raw, args.start_year, args.end_year)
    player_seasons = aggregate_player_seasons(annual)
    salary_cap = build_salary_cap_table(player_seasons)

    print_checkpoint(player_seasons, salary_cap)
    confirm(f"Write {len(player_seasons):,} salary rows and {len(salary_cap):,} cap rows?", args.auto_confirm)

    player_seasons.to_parquet(PLAYER_OUT_PATH, index=False)
    salary_cap.to_parquet(CAP_OUT_PATH, index=False)
    print(f"\nWrote {len(player_seasons):,} rows to {PLAYER_OUT_PATH}")
    print(f"Wrote {len(salary_cap):,} rows to {CAP_OUT_PATH}")


if __name__ == "__main__":
    main()
