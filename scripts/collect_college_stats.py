"""
FR-10: CFBD college season stats — passing + rushing + receiving.

Pulls by year (~29 years × 3 categories = ~87 requests). For each year, accumulates
raw stat rows keyed by CFBD playerId. After all years are fetched, resolves each
unique CFBD player to an nflverse player_id in two phases:

  Phase A — PlayerResolver (FR-13):
    Tries (team × draft_year_offset) combinations. Tier 3 (name + college, 0.75)
    fires when CFBD's team string matches nflverse's college string exactly. When
    it doesn't (multi-school transfers like "USC; Pittsburgh", "Wyoming; Reedley"),
    Tier 4 (name + draft_year, 0.50) takes over. draft_year is estimated as
    last_college_season + {1, 2, 3} to handle players with a gap between their
    last CFBD-tracked season and the NFL draft (e.g. transfers to D2 schools that
    CFBD doesn't cover, like Tyreek Hill: Oklahoma State 2014 → West Alabama 2015
    → drafted 2016). Position from the CFBD stat row enables Tier 5 fallback.

  Phase B — slug recovery for residual unmatched:
    Some combine rows store nflverse-formatted nicknames ("Cam Ward") while CFBD
    returns the formal first name ("Cameron Ward") — PlayerResolver's fuzzy first-
    name comparison can't bridge that gap. The combine table's `cfb_id` field
    *does* encode the canonical CFBD slug ("cameron-ward-1"), so we strip the
    trailing counter and match it against `slugify(CFBD_player_name)`. This is an
    exact key match against the existing nflverse↔CFBD link, not fuzzy matching.

Progress is checkpointed after each year (raw rows, pre-resolution).

Usage:
    .venv/bin/python3 scripts/collect_college_stats.py
    .venv/bin/python3 scripts/collect_college_stats.py --auto-confirm
    .venv/bin/python3 scripts/collect_college_stats.py --resume
"""

import sys
import os
import time
import json
import re
import unicodedata
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
import requests
from utils.player_resolver import PlayerResolver

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

COMBINE_PATH    = Path("data/athletic/combine_draft.parquet")
OUT_PATH        = Path("data/athletic/college_stats.parquet")
CHECKPOINT_PATH = Path("data/athletic/college_stats_checkpoint.parquet")
YEARS_DONE_PATH = Path("data/athletic/college_stats_years_done.json")

YEARS           = list(range(1997, 2026))
STAT_CATEGORIES = ["passing", "rushing", "receiving"]
REQUEST_DELAY   = 0.5
BASE_URL        = "https://api.collegefootballdata.com"

AUTO_CONFIRM = "--auto-confirm" in sys.argv
RESUME       = "--resume"       in sys.argv

# CFBD stat type → output column, per category. Names verified against live API.
STAT_FIELDS = {
    "passing": {
        "ATT":         "pass_att",
        "COMPLETIONS": "pass_comp",
        "YDS":         "pass_yds",
        "TD":          "pass_td",
        "INT":         "pass_int",
    },
    "rushing": {
        "CAR": "rush_att",
        "YDS": "rush_yds",
        "TD":  "rush_td",
    },
    "receiving": {
        "REC": "rec",
        "YDS": "rec_yds",
        "TD":  "rec_td",
    },
}

ALL_STAT_COLS = [c for fields in STAT_FIELDS.values() for c in fields.values()]


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    val = os.environ.get("CFBD_API_KEY")
    if val:
        return val
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CFBD_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("CFBD_API_KEY not set. Add to .env: CFBD_API_KEY=your_key_here")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_NONALNUM = re.compile(r"[^a-z0-9]+")

def to_slug(name: str) -> str:
    """Replicates nflverse cfb_id slug format: lowercase, accents stripped,
    apostrophes/periods removed, all other non-alphanumerics collapsed to '-'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", "").replace(".", "")
    s = _SLUG_NONALNUM.sub("-", s)
    return s.strip("-")


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def fetch_year_stats(year: int, category: str, headers: dict) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/stats/player/season",
        headers=headers,
        params={"year": year, "category": category},
        timeout=30,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json()


def empty_stat_row(cfbd_id, name, team, position, season) -> dict:
    row = {
        "cfbd_player_id":  str(cfbd_id),
        "player_name":     name,
        "college":         team,
        "cfbd_position":   position if position and position != "?" else None,
        "season":          season,
    }
    for col in ALL_STAT_COLS:
        row[col] = None
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = _load_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    combine           = pd.read_parquet(COMBINE_PATH)
    target_player_ids = set(combine["player_id"].dropna())
    cfb_player_ids    = set(combine[combine["cfb_id"].notna()]["player_id"].dropna())

    # Slug index for Phase B recovery. Two keys per combine row to bridge name
    # variants:
    #   1. cfb_id stem ("cameron-ward-1" → "cameron-ward")  — catches Cam Ward
    #      where nflverse player_name uses the nickname but cfb_id encodes the
    #      formal first name CFBD returns in stats.
    #   2. slugified player_name ("JuJu Smith-Schuster" → "juju-smith-schuster") —
    #      catches hyphenated last names that nflverse truncates in cfb_id
    #      ("juju-smith-1" only stores the first half).
    # Collisions on either key are stored as a list and disambiguated by
    # draft_year proximity at lookup time.
    slug_index: dict[str, list[tuple]] = {}
    for _, row in combine[combine["cfb_id"].notna()].iterrows():
        entry    = (row["player_id"], row.get("draft_year"))
        cfb_stem = str(row["cfb_id"]).rsplit("-", 1)[0]
        slug_index.setdefault(cfb_stem, []).append(entry)

        name_slug = to_slug(str(row["player_name"]))
        if name_slug and name_slug != cfb_stem:
            slug_index.setdefault(name_slug, []).append(entry)

    print(f"Combine table: {len(combine)} players total, {len(cfb_player_ids)} with cfb_id, "
          f"{len(slug_index)} unique slug stems")

    # Load checkpoint if resuming
    years_done: set[int] = set()
    raw_rows:   list[dict] = []
    if RESUME:
        if YEARS_DONE_PATH.exists():
            years_done = set(json.loads(YEARS_DONE_PATH.read_text()))
        if CHECKPOINT_PATH.exists() and years_done:
            ckpt_df = pd.read_parquet(CHECKPOINT_PATH)
            required = {"cfbd_player_id", "cfbd_position", "rush_att", "pass_comp"}
            if not required.issubset(ckpt_df.columns):
                print("Existing checkpoint is in an old format — discarding.")
                years_done = set()
            else:
                raw_rows = ckpt_df.to_dict("records")
                print(f"Resuming: {len(years_done)} years done, {len(raw_rows)} raw rows in checkpoint")

    years_to_fetch = [y for y in YEARS if y not in years_done]
    if years_to_fetch:
        total_requests = len(years_to_fetch) * len(STAT_CATEGORIES)
        eta            = total_requests * REQUEST_DELAY
        print(
            f"Years to fetch: {years_to_fetch[0]}–{years_to_fetch[-1]} "
            f"({len(years_to_fetch)} years × {len(STAT_CATEGORIES)} categories "
            f"= {total_requests} requests, ~{eta:.0f}s at {REQUEST_DELAY}s delay)"
        )
        confirm("Proceed?")
    else:
        print("All years already fetched — re-running resolution only.")

    # ---------------------------------------------------------------------------
    # Phase 1: fetch raw stats by year × category
    # ---------------------------------------------------------------------------

    for year in years_to_fetch:
        year_rows: dict[str, dict] = {}    # cfbd_player_id → row dict

        for category in STAT_CATEGORIES:
            print(f"  {year} / {category:<9} ...", end="  ", flush=True)
            try:
                raw = fetch_year_stats(year, category, headers)
            except requests.HTTPError as e:
                print(f"skip ({e.response.status_code})")
                time.sleep(REQUEST_DELAY)
                continue
            except Exception as e:
                print(f"error: {e}")
                time.sleep(REQUEST_DELAY)
                continue

            field_map = STAT_FIELDS[category]
            for entry in raw:
                cfbd_id = entry.get("playerId")
                if cfbd_id is None:
                    continue
                cfbd_id = str(cfbd_id)
                if cfbd_id not in year_rows:
                    year_rows[cfbd_id] = empty_stat_row(
                        cfbd_id,
                        entry.get("player", ""),
                        entry.get("team", ""),
                        entry.get("position", ""),
                        year,
                    )
                # Upgrade position if previously '?' and now we see a real one
                pos = entry.get("position", "")
                if pos and pos != "?" and not year_rows[cfbd_id]["cfbd_position"]:
                    year_rows[cfbd_id]["cfbd_position"] = pos

                stat_type = (entry.get("statType") or "").upper()
                if stat_type in field_map:
                    year_rows[cfbd_id][field_map[stat_type]] = entry.get("stat")

            print(f"{len(raw):>5} stat rows → {len(year_rows):>4} players so far")
            time.sleep(REQUEST_DELAY)

        raw_rows.extend(year_rows.values())

        years_done.add(year)
        if raw_rows:
            pd.DataFrame(raw_rows).to_parquet(CHECKPOINT_PATH, index=False)
        YEARS_DONE_PATH.write_text(json.dumps(sorted(years_done)))

    if not raw_rows:
        print("\nNo rows collected — nothing to resolve or write.")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Phase 2: resolve player_id once per unique CFBD player
    # ---------------------------------------------------------------------------

    df_raw = pd.DataFrame(raw_rows)
    n_unique = df_raw["cfbd_player_id"].nunique()
    print(f"\nResolving {n_unique:,} unique CFBD players via PlayerResolver + slug recovery...")

    resolver = PlayerResolver()

    resolutions: dict[str, tuple[str, float, str]] = {}    # cfbd_id → (player_id, conf, source)
    slug_recoveries = 0

    for cfbd_id, group in df_raw.groupby("cfbd_player_id"):
        sorted_g    = group.sort_values("season")
        name        = sorted_g["player_name"].iloc[-1]
        teams       = [t for t in group["college"].dropna().unique() if t]
        last_season = int(group["season"].max())

        positions = [p for p in group["cfbd_position"].dropna().unique() if p]
        position  = positions[0] if positions else None

        best_pid:  str | None = None
        best_conf: float       = 0.0

        # Phase A: PlayerResolver with team × draft_year sweeps.
        # Stop at the first Tier-3 hit (0.75) — no point pinging more combos.
        for offset in (1, 2, 3):
            for team in teams or [""]:
                record = {
                    "name":       name,
                    "college":    team,
                    "draft_year": last_season + offset,
                }
                if position:
                    record["position"] = position
                pid, conf = resolver.resolve(record)
                if pid and conf > best_conf:
                    best_pid, best_conf = pid, conf
                    if best_conf >= 0.75:
                        break
            if best_conf >= 0.75:
                break

        source = "resolver" if best_pid else None

        # Phase B: slug-based recovery (uses combine's cfb_id field as join key)
        if best_pid is None:
            slug = to_slug(name)
            if slug in slug_index:
                candidates = slug_index[slug]
                target_year = last_season + 1
                # Pick the candidate with closest draft_year to our estimate
                scored = sorted(
                    candidates,
                    key=lambda c: abs(c[1] - target_year) if pd.notna(c[1]) else 99,
                )
                best_pid  = scored[0][0]
                best_conf = 0.50    # treat as Tier-4-equivalent confidence
                source    = "slug"
                slug_recoveries += 1

        if best_pid:
            resolutions[str(cfbd_id)] = (best_pid, best_conf, source)

    resolver.flush_unmatched()

    df_raw["player_id"]           = df_raw["cfbd_player_id"].map(lambda c: resolutions.get(c, (None, 0.0, None))[0])
    df_raw["resolver_confidence"] = df_raw["cfbd_player_id"].map(lambda c: resolutions.get(c, (None, 0.0, None))[1])
    df_raw["match_source"]        = df_raw["cfbd_player_id"].map(lambda c: resolutions.get(c, (None, 0.0, None))[2])

    df = df_raw[df_raw["player_id"].isin(target_player_ids)].copy()

    # ---------------------------------------------------------------------------
    # Match-rate report (acceptance: >95% of cfb_id players matched)
    # ---------------------------------------------------------------------------

    resolved_ids = set(df["player_id"].dropna())
    matched_cfb  = len(resolved_ids & cfb_player_ids)
    rate         = matched_cfb / len(cfb_player_ids) * 100

    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Raw CFBD rows fetched:    {len(df_raw):,}")
    print(f"Resolved & in combine:    {len(df):,}")
    print(f"Unique players resolved:  {df['player_id'].nunique():,}")
    print(f"Target (cfb_id set):      {len(cfb_player_ids):,}")
    print(f"Matched from cfb_id:      {matched_cfb} / {len(cfb_player_ids)} ({rate:.1f}%)")
    print(f"Slug-recovery saves:      {slug_recoveries}")
    if rate < 95.0:
        print(f"  WARNING: below 95% target — review data/unmatched/ before closing FR-10")

    print(f"\nMatch source breakdown (unique players):")
    src_counts = df.drop_duplicates("player_id")["match_source"].value_counts()
    for src, n in src_counts.items():
        print(f"  {src:<10} {n:,}")

    print(f"\nConfidence breakdown (unique players):")
    conf_counts = df.drop_duplicates("player_id")["resolver_confidence"].value_counts().sort_index(ascending=False)
    for conf, n in conf_counts.items():
        tier = {1.0: "Tier 1", 0.9: "Tier 2", 0.75: "Tier 3", 0.5: "Tier 4 / slug", 0.35: "Tier 5"}.get(conf, "?")
        print(f"  {conf:.2f} ({tier}):  {n:,}")

    print(f"\nSchema:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isna().sum()
        print(f"  {col:<24} {str(dtype):<12} nulls: {nulls/len(df)*100:.1f}%")

    print(f"\nSpot checks (Tyreek Hill, Cam Ward):")
    spot = df[df["player_id"].isin({"00-0033040", "00-0040676"})]
    if not spot.empty:
        print(spot[["player_id", "player_name", "college", "season",
                   "pass_yds", "rush_att", "rush_yds", "rec", "rec_yds",
                   "match_source"]].to_string())
    else:
        print("  none found — investigate before closing FR-10")

    confirm(f"Write {len(df):,} rows to {OUT_PATH}?")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows → {OUT_PATH}")

    for p in (CHECKPOINT_PATH, YEARS_DONE_PATH):
        if p.exists():
            p.unlink()

    print(f"Done. Review {OUT_PATH} and advance FR-10 once match rate and schema are confirmed.")


if __name__ == "__main__":
    main()
