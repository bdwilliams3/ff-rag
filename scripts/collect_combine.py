"""
FR-9: Combine measurables + draft status via nfl_data_py.
Uses import_players() as the authoritative base for full draft coverage
(includes combine-skippers like Waddle), then left-joins combine measurables.
Bridges to canonical nflverse player_id via gsis_id.
"""

import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, ".")
import nfl_data_py as nfl
import pandas as pd
from typing import Optional
from utils.player_resolver import TEAM_MAP

YEARS = list(range(2000, 2026))  # nfl_data_py combine data available from 2000
POSITIONS = ["QB", "RB", "WR", "TE"]
DRAFT_START = 2000  # reference table covers draftees from this year forward

OUT_PATH = Path("data/athletic/combine_draft.parquet")
UNMATCHED_PATH = Path(f"data/unmatched/combine_unmatched_{date.today()}.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
UNMATCHED_PATH.parent.mkdir(parents=True, exist_ok=True)

AUTO_CONFIRM = "--auto-confirm" in sys.argv


def normalize_team(val) -> Optional[str]:
    if not val or (hasattr(val, '__float__') and pd.isna(val)):
        return None
    key = str(val).upper().strip()
    if key in TEAM_MAP:
        return TEAM_MAP[key]
    # already a full name — normalize via reverse lookup
    t = str(val).strip().lower()
    for canonical in set(TEAM_MAP.values()):
        if t == canonical.lower():
            return canonical
    return str(val).strip()


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def main() -> None:
    print("Pulling player identity table...")
    players = nfl.import_players()
    players_skill = players[
        players["position"].isin(POSITIONS) &
        players["gsis_id"].notna()
    ].copy()

    print(f"Pulling combine data {DRAFT_START}-2025...")
    combine = nfl.import_combine_data(years=YEARS, positions=POSITIONS)

    # Base: three buckets unioned together, deduped on gsis_id
    #   1. Drafted skill players 2000-2025
    #   2. UDFA combine attendees (no draft year, but attended combine)
    #   3. Modern-era UDFAs who never attended the combine but made the league
    #      (born after 1980, ensures active in a fantasy-relevant era)
    combine_pfr = set(combine["pfr_id"].dropna())
    drafted = players_skill[players_skill["draft_year"].between(DRAFT_START, 2025)]
    udfa_combine = players_skill[
        players_skill["draft_year"].isna() &
        players_skill["pfr_id"].isin(combine_pfr)
    ]
    udfa_no_combine = players_skill[
        players_skill["draft_year"].isna() &
        ~players_skill["pfr_id"].isin(combine_pfr) &
        (players_skill["birth_date"] > "1980-01-01")
    ]
    base = pd.concat([drafted, udfa_combine, udfa_no_combine]).drop_duplicates("gsis_id").copy()
    print(f"  Drafted skill {DRAFT_START}-2025:   {len(drafted):,}")
    print(f"  UDFA combine attendees:     {len(udfa_combine):,}")
    print(f"  UDFA no-combine (modern):   {len(udfa_no_combine):,}")
    print(f"  Total base:                 {len(base):,}")

    # Combine measurables — one row per pfr_id
    measurables = (
        combine[["pfr_id", "ht", "wt", "forty", "bench", "vertical",
                 "broad_jump", "cone", "shuttle", "cfb_id"]]
        .drop_duplicates("pfr_id")
    )

    print("Joining combine measurables onto player base (left join)...")
    merged = base.merge(measurables, on="pfr_id", how="left")

    merged = merged.rename(columns={
        "gsis_id":      "player_id",
        "display_name": "player_name",
        "college_name": "college",
    })
    merged["draft_team"] = merged["draft_team"].apply(normalize_team)

    final_cols = [
        "player_id", "player_name", "position", "college",
        "draft_year", "draft_round", "draft_pick", "draft_team",
        "birth_date", "pfr_id", "cfb_id",
        "ht", "wt", "forty", "bench", "vertical",
        "broad_jump", "cone", "shuttle",
    ]
    final_cols = [c for c in final_cols if c in merged.columns]
    merged = merged[final_cols].reset_index(drop=True)

    with_measurables = merged["forty"].notna().sum()
    print(f"\nCoverage:")
    print(f"  Total players:            {len(merged):,}")
    print(f"  With combine measurables: {with_measurables:,} ({with_measurables/len(merged)*100:.1f}%)")

    print(f"\n{'='*60}")
    print("CHECKPOINT: COMBINE + DRAFT DATA")
    print(f"{'='*60}")
    print(f"\nShape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")
    print(f"\nSchema:")
    for col, dtype in merged.dtypes.items():
        nulls = merged[col].isna().sum()
        print(f"  {col:<30} {str(dtype):<12} nulls: {nulls/len(merged)*100:.1f}%")
    print(f"\nSample (3 rows):")
    print(merged[["player_id", "player_name", "position", "college",
                   "draft_round", "draft_pick", "draft_team",
                   "forty", "wt", "birth_date"]].head(3).to_string())

    # Confirm Waddle made it through
    waddle = merged[merged["player_name"].str.contains("Waddle", case=False, na=False)]
    print(f"\nWaddle check: {len(waddle)} row(s)")
    if not waddle.empty:
        print(waddle[["player_id", "player_name", "draft_round", "draft_pick",
                       "draft_team", "forty"]].to_string())

    confirm("Schema looks good — write Parquet?")

    merged.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(merged):,} rows to {OUT_PATH}")
    print("\nDone. Review output and close FR-9 once satisfied.")


if __name__ == "__main__":
    main()
