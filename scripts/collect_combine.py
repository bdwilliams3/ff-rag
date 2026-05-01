"""
FR-9: Combine measurables + draft status via nfl_data_py.
Bridges combine data to canonical nflverse player_id via pfr_id.
Unmatched players written to data/unmatched/ — never dropped silently.
"""

import sys
from pathlib import Path
from datetime import date
import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2012, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]

OUT_PATH = Path("data/athletic/combine_draft.parquet")
UNMATCHED_PATH = Path(f"data/unmatched/combine_unmatched_{date.today()}.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
UNMATCHED_PATH.parent.mkdir(parents=True, exist_ok=True)

AUTO_CONFIRM = "--auto-confirm" in sys.argv


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def main() -> None:
    print("Pulling combine data 2012-2024...")
    combine = nfl.import_combine_data(years=YEARS, positions=POSITIONS)

    print("Pulling player identity table...")
    players = nfl.import_players()
    players_slim = (
        players[["gsis_id", "pfr_id", "birth_date"]]
        .dropna(subset=["pfr_id"])
        .drop_duplicates("pfr_id")
    )

    print("Joining on pfr_id...")
    merged = combine.merge(players_slim, on="pfr_id", how="left")
    merged = merged.rename(columns={
        "gsis_id": "player_id",
        "pos":     "position",
        "school":  "college",
        "draft_ovr": "draft_pick",
    })

    matched   = merged[merged["player_id"].notna()].reset_index(drop=True)
    unmatched = merged[merged["player_id"].isna()].reset_index(drop=True)

    total = len(merged)
    print(f"\nMatch results:")
    print(f"  Matched:   {len(matched):,} ({len(matched)/total*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched):,} ({len(unmatched)/total*100:.1f}%)")
    print(f"    └── UDFAs (no draft round): {unmatched['draft_round'].isna().sum()}")
    print(f"    └── Drafted but unmatched:  {unmatched['draft_round'].notna().sum()}")

    # Schema checkpoint
    print(f"\n{'='*60}")
    print("CHECKPOINT: COMBINE + DRAFT DATA")
    print(f"{'='*60}")
    print(f"\nShape: {matched.shape[0]:,} rows x {matched.shape[1]} columns")
    print(f"\nSchema:")
    for col, dtype in matched.dtypes.items():
        nulls = matched[col].isna().sum()
        print(f"  {col:<30} {str(dtype):<12} nulls: {nulls/len(matched)*100:.1f}%")
    print(f"\nSample (3 rows):")
    print(matched[["player_id", "player_name", "position", "college",
                   "draft_round", "draft_pick", "draft_team",
                   "forty", "vertical", "wt", "birth_date"]].head(3).to_string())

    confirm("Schema looks good — write Parquet?")

    matched.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(matched):,} rows to {OUT_PATH}")

    if len(unmatched):
        unmatched.to_csv(UNMATCHED_PATH, index=False)
        print(f"Wrote {len(unmatched)} unmatched rows to {UNMATCHED_PATH}")

    print("\nDone. Review output and close FR-9 once satisfied.")


if __name__ == "__main__":
    main()
