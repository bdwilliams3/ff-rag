"""
FR-15: Offensive and defensive coordinator tendency profiles.

Builds two tables from coaching_weekly + weekly player stats:

  data/coaching/offensive_coordinator_profiles.parquet
    One row per OC/team-season, with team offensive production during the
    coordinator's active weeks.

  data/coaching/defensive_coordinator_profiles.parquet
    One row per DC/team-season, with opponent production allowed during the
    coordinator's active weeks.

Run: .venv/bin/python3 scripts/build_coordinator_profiles.py [--auto-confirm]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


COACHING_PATH = Path("data/coaching/coaching_weekly.parquet")
WEEKLY_STATS_DIR = Path("data/stats/weekly")
OFFENSE_OUT_PATH = Path("data/coaching/offensive_coordinator_profiles.parquet")
DEFENSE_OUT_PATH = Path("data/coaching/defensive_coordinator_profiles.parquet")
OFFENSE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
AUTO_CONFIRM = "--auto-confirm" in sys.argv


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def load_weekly_stats() -> pd.DataFrame:
    files = sorted(WEEKLY_STATS_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No weekly stats found under {WEEKLY_STATS_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prep_stats(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats[stats["season_type"] == "REG"].copy()
    numeric_cols = [
        "attempts", "passing_yards", "carries", "rushing_yards",
        "targets", "receptions", "receiving_yards",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def team_week_offense(stats: pd.DataFrame) -> pd.DataFrame:
    base = (
        stats.groupby(["season", "week", "recent_team"], as_index=False)
        .agg(
            pass_attempts=("attempts", "sum"),
            pass_yards=("passing_yards", "sum"),
            rush_attempts=("carries", "sum"),
            rush_yards=("rushing_yards", "sum"),
            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            receiving_yards=("receiving_yards", "sum"),
        )
        .rename(columns={"recent_team": "team"})
    )

    pos = (
        stats.groupby(["season", "week", "recent_team", "position"], as_index=False)
        .agg(targets=("targets", "sum"), receiving_yards=("receiving_yards", "sum"), carries=("carries", "sum"))
        .pivot_table(
            index=["season", "week", "recent_team"],
            columns="position",
            values=["targets", "receiving_yards", "carries"],
            fill_value=0,
        )
    )
    pos.columns = [f"{position.lower()}_{metric}" for metric, position in pos.columns]
    pos = pos.reset_index().rename(columns={"recent_team": "team"})

    wr = stats[stats["position"] == "WR"].copy()
    wr = (
        wr.groupby(["season", "week", "recent_team", "player_id"], as_index=False)["targets"].sum()
        .sort_values(["season", "week", "recent_team", "targets"], ascending=[True, True, True, False])
    )
    wr["wr_rank"] = wr.groupby(["season", "week", "recent_team"]).cumcount() + 1
    wr_ranks = (
        wr[wr["wr_rank"].isin([1, 2])]
        .pivot_table(index=["season", "week", "recent_team"], columns="wr_rank", values="targets", fill_value=0)
        .reset_index()
        .rename(columns={"recent_team": "team", 1: "wr1_targets", 2: "wr2_targets"})
    )

    metrics = base.merge(pos, on=["season", "week", "team"], how="left")
    metrics = metrics.merge(wr_ranks, on=["season", "week", "team"], how="left")
    for col in [
        "rb_targets", "wr_targets", "te_targets",
        "rb_receiving_yards", "wr_receiving_yards", "te_receiving_yards",
        "rb_carries", "wr1_targets", "wr2_targets",
    ]:
        if col not in metrics.columns:
            metrics[col] = 0
    return metrics.fillna(0)


def team_week_defense_allowed(stats: pd.DataFrame) -> pd.DataFrame:
    base = (
        stats.groupby(["season", "week", "opponent_team"], as_index=False)
        .agg(
            pass_attempts_against=("attempts", "sum"),
            pass_yards_against=("passing_yards", "sum"),
            rush_attempts_against=("carries", "sum"),
            rush_yards_against=("rushing_yards", "sum"),
            targets_allowed=("targets", "sum"),
            receptions_allowed=("receptions", "sum"),
            receiving_yards_allowed=("receiving_yards", "sum"),
        )
        .rename(columns={"opponent_team": "team"})
    )

    pos = (
        stats.groupby(["season", "week", "opponent_team", "position"], as_index=False)
        .agg(targets=("targets", "sum"), receiving_yards=("receiving_yards", "sum"), carries=("carries", "sum"))
        .pivot_table(
            index=["season", "week", "opponent_team"],
            columns="position",
            values=["targets", "receiving_yards", "carries"],
            fill_value=0,
        )
    )
    pos.columns = [f"{position.lower()}_{metric}_allowed" for metric, position in pos.columns]
    pos = pos.reset_index().rename(columns={"opponent_team": "team"})

    metrics = base.merge(pos, on=["season", "week", "team"], how="left")
    for col in [
        "rb_targets_allowed", "wr_targets_allowed", "te_targets_allowed",
        "rb_receiving_yards_allowed", "wr_receiving_yards_allowed", "te_receiving_yards_allowed",
        "rb_carries_allowed",
    ]:
        if col not in metrics.columns:
            metrics[col] = 0
    return metrics.fillna(0)


def build_offensive_profiles(coaching: pd.DataFrame, offense: pd.DataFrame) -> pd.DataFrame:
    joined = coaching.merge(offense, on=["season", "week", "team"], how="inner")
    grouped = (
        joined.groupby(["season", "team", "team_name", "hc_name", "oc_name"], as_index=False)
        .agg(
            games_coached=("had_game", "sum"),
            weeks_on_staff=("week", "nunique"),
            pass_attempts=("pass_attempts", "sum"),
            pass_yards=("pass_yards", "sum"),
            rush_attempts=("rush_attempts", "sum"),
            rush_yards=("rush_yards", "sum"),
            total_targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            wr_targets=("wr_targets", "sum"),
            wr1_targets=("wr1_targets", "sum"),
            wr2_targets=("wr2_targets", "sum"),
            te_targets=("te_targets", "sum"),
            rb_targets=("rb_targets", "sum"),
            wr_receiving_yards=("wr_receiving_yards", "sum"),
            te_receiving_yards=("te_receiving_yards", "sum"),
            rb_receiving_yards=("rb_receiving_yards", "sum"),
            rb_carries=("rb_carries", "sum"),
        )
    )
    total_plays = grouped["pass_attempts"] + grouped["rush_attempts"]
    grouped["pass_rate"] = grouped["pass_attempts"] / total_plays.where(total_plays != 0)
    grouped["run_rate"] = grouped["rush_attempts"] / total_plays.where(total_plays != 0)
    for col, source in [
        ("wr_target_share", "wr_targets"),
        ("wr1_target_share", "wr1_targets"),
        ("wr2_target_share", "wr2_targets"),
        ("te_target_share", "te_targets"),
        ("rb_target_share", "rb_targets"),
    ]:
        grouped[col] = grouped[source] / grouped["total_targets"].where(grouped["total_targets"] != 0)
    return grouped.sort_values(["season", "team", "oc_name"]).reset_index(drop=True)


def build_defensive_profiles(coaching: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    joined = coaching.merge(defense, on=["season", "week", "team"], how="inner")
    grouped = (
        joined.groupby(["season", "team", "team_name", "hc_name", "dc_name"], as_index=False)
        .agg(
            games_coached=("had_game", "sum"),
            weeks_on_staff=("week", "nunique"),
            pass_attempts_against=("pass_attempts_against", "sum"),
            pass_yards_against=("pass_yards_against", "sum"),
            rush_attempts_against=("rush_attempts_against", "sum"),
            rush_yards_against=("rush_yards_against", "sum"),
            targets_allowed=("targets_allowed", "sum"),
            receptions_allowed=("receptions_allowed", "sum"),
            receiving_yards_allowed=("receiving_yards_allowed", "sum"),
            wr_targets_allowed=("wr_targets_allowed", "sum"),
            te_targets_allowed=("te_targets_allowed", "sum"),
            rb_targets_allowed=("rb_targets_allowed", "sum"),
            wr_receiving_yards_allowed=("wr_receiving_yards_allowed", "sum"),
            te_receiving_yards_allowed=("te_receiving_yards_allowed", "sum"),
            rb_receiving_yards_allowed=("rb_receiving_yards_allowed", "sum"),
            rb_carries_allowed=("rb_carries_allowed", "sum"),
        )
    )
    total_plays = grouped["pass_attempts_against"] + grouped["rush_attempts_against"]
    grouped["pass_rate_allowed"] = grouped["pass_attempts_against"] / total_plays.where(total_plays != 0)
    grouped["run_rate_allowed"] = grouped["rush_attempts_against"] / total_plays.where(total_plays != 0)
    for col, source in [
        ("wr_target_share_allowed", "wr_targets_allowed"),
        ("te_target_share_allowed", "te_targets_allowed"),
        ("rb_target_share_allowed", "rb_targets_allowed"),
    ]:
        grouped[col] = grouped[source] / grouped["targets_allowed"].where(grouped["targets_allowed"] != 0)
    return grouped.sort_values(["season", "team", "dc_name"]).reset_index(drop=True)


def print_checkpoint(offense: pd.DataFrame, defense: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("CHECKPOINT: OFFENSIVE COORDINATOR PROFILES")
    print(f"{'='*60}")
    print(f"\nShape: {offense.shape[0]:,} rows x {offense.shape[1]} columns")
    print("\nTop pass-rate OC seasons (min 8 games):")
    offense_cols = ["season", "team", "hc_name", "oc_name", "games_coached", "pass_rate", "wr_target_share", "te_target_share", "rb_target_share"]
    print(offense[offense["games_coached"] >= 8].sort_values("pass_rate", ascending=False)[offense_cols].head(12).to_string(index=False))

    print(f"\n{'='*60}")
    print("CHECKPOINT: DEFENSIVE COORDINATOR PROFILES")
    print(f"{'='*60}")
    print(f"\nShape: {defense.shape[0]:,} rows x {defense.shape[1]} columns")
    print("\nMost pass yards allowed by DC season (min 8 games):")
    defense_cols = ["season", "team", "hc_name", "dc_name", "games_coached", "pass_yards_against", "te_receiving_yards_allowed", "rb_carries_allowed"]
    print(defense[defense["games_coached"] >= 8].sort_values("pass_yards_against", ascending=False)[defense_cols].head(12).to_string(index=False))


def main() -> None:
    if not COACHING_PATH.exists():
        raise RuntimeError(f"Missing {COACHING_PATH}; run scripts/collect_coaching.py first")

    coaching = pd.read_parquet(COACHING_PATH)
    stats = prep_stats(load_weekly_stats())
    offense = team_week_offense(stats)
    defense = team_week_defense_allowed(stats)

    offensive_profiles = build_offensive_profiles(coaching, offense)
    defensive_profiles = build_defensive_profiles(coaching, defense)

    print_checkpoint(offensive_profiles, defensive_profiles)
    confirm("Schema looks good — write offensive and defensive coordinator profile Parquet files?")

    offensive_profiles.to_parquet(OFFENSE_OUT_PATH, index=False)
    defensive_profiles.to_parquet(DEFENSE_OUT_PATH, index=False)

    print(f"\nWrote {len(offensive_profiles):,} rows to {OFFENSE_OUT_PATH}")
    print(f"Wrote {len(defensive_profiles):,} rows to {DEFENSE_OUT_PATH}")
    print("Done. Review data/coaching/ and close FR-15 once satisfied.")


if __name__ == "__main__":
    main()
