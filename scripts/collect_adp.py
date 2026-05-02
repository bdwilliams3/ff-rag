"""
FR-16: FantasyPros historical ADP — standard, PPR, half-PPR, 2017-present.
Scrapes overall ADP pages, resolves player_id via PlayerResolver (Tier 5).
Run: .venv/bin/python3 scripts/collect_adp.py [--auto-confirm]
"""

import sys
import time
from pathlib import Path
from typing import Optional
sys.path.insert(0, ".")
import requests
from bs4 import BeautifulSoup
import pandas as pd
from utils.player_resolver import PlayerResolver

YEARS = list(range(2017, 2026))
FORMATS = {
    "std":      "https://www.fantasypros.com/nfl/adp/overall.php",
    "ppr":      "https://www.fantasypros.com/nfl/adp/ppr-overall.php",
    "half_ppr": "https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php",
}
POSITIONS = {"QB", "RB", "WR", "TE"}
NFL_TEAM_CODES = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LA", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}
TEAM_NORMALIZE = {"LAR": "LA", "JAC": "JAX"}

OUT_PATH = Path("data/adp/adp_historical.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
AUTO_CONFIRM = "--auto-confirm" in sys.argv


def _strip_pos(pos: str) -> str:
    """'WR1' → 'WR', 'RB3' → 'RB'"""
    return pos.rstrip("0123456789")


def _normalize_team(team: Optional[str]) -> Optional[str]:
    if not team:
        return None
    team = team.strip().upper()
    if team not in NFL_TEAM_CODES:
        return None
    return TEAM_NORMALIZE.get(team, team)


def scrape_page(url: str, year: int, fmt: str) -> list[dict]:
    params = {} if year == 2025 else {"year": year}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"    {year} {fmt} failed: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return []

    t = tables[0]
    col_headers = [th.get_text(strip=True) for th in t.find_all("th")]
    avg_idx = next((i for i, h in enumerate(col_headers) if h == "AVG"), None)
    if avg_idx is None:
        return []

    rows = []
    for tr in t.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # Historical FantasyPros team labels are sparse and can be current-team
        # values on old seasons, so the scraper intentionally ignores them.
        player_td = tds[1]
        name_tag = player_td.find("a", class_="player-name")
        if not name_tag:
            continue
        player_name = name_tag.get_text(strip=True)

        pos_raw = tds[2].get_text(strip=True) if len(tds) > 2 else ""
        position = _strip_pos(pos_raw)
        if position not in POSITIONS:
            continue

        try:
            avg_adp = float(tds[avg_idx].get_text(strip=True))
        except (ValueError, IndexError):
            continue

        rows.append({
            "player_name": player_name,
            "position":    position,
            "avg_adp":     avg_adp,
            "format":      fmt,
            "season":      year,
        })

    return rows


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def _first_team_by_player_season(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    df = df.dropna(subset=["player_id", team_col]).copy()
    df[team_col] = df[team_col].map(_normalize_team)
    df = df.dropna(subset=[team_col])
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(99)
        df = df.sort_values(["player_id", "season", "week"])
    return (
        df.groupby(["player_id", "season"], as_index=False)[team_col]
        .first()
        .rename(columns={team_col: "team"})
    )


def infer_teams_from_local_data(df: pd.DataFrame) -> pd.DataFrame:
    """Infer the historical ADP-season team from local nflverse data."""
    inferred = []
    weekly_dir = Path("data/stats/weekly")
    weekly_files = sorted(weekly_dir.glob("*.parquet"))
    if weekly_files:
        weekly = pd.concat([
            pd.read_parquet(f, columns=["player_id", "season", "week", "recent_team"])
            for f in weekly_files
        ], ignore_index=True)
        weekly["season"] = weekly["season"].astype(int)
        weekly_teams = _first_team_by_player_season(weekly, "recent_team")
        weekly_teams["team_source"] = "nflverse_weekly"
        inferred.append(weekly_teams)

    injury_dir = Path("data/injuries")
    injury_files = sorted(injury_dir.glob("*.parquet"))
    if injury_files:
        injuries = pd.concat([
            pd.read_parquet(f, columns=["player_id", "season", "week", "team"])
            for f in injury_files
        ], ignore_index=True)
        injuries["season"] = injuries["season"].astype(int)
        injury_teams = _first_team_by_player_season(injuries, "team")
        injury_teams["team_source"] = "nflverse_injuries"
        inferred.append(injury_teams)

    if not inferred:
        df["team"] = None
        df["team_source"] = "missing"
        return df

    teams = pd.concat(inferred, ignore_index=True)
    source_rank = {"nflverse_weekly": 0, "nflverse_injuries": 1}
    teams["source_rank"] = teams["team_source"].map(source_rank)
    teams = (
        teams.sort_values(["player_id", "season", "source_rank"])
        .drop_duplicates(["player_id", "season"], keep="first")
        .drop(columns=["source_rank"])
    )

    out = df.merge(teams, on=["player_id", "season"], how="left")
    out["team_source"] = out["team_source"].fillna("missing")
    return out


def main() -> None:
    all_rows = []
    for fmt, url in FORMATS.items():
        for year in YEARS:
            rows = scrape_page(url, year, fmt)
            print(f"  {year} {fmt:<9} {len(rows):>3} rows")
            all_rows.extend(rows)
            time.sleep(0.4)  # polite crawl rate

    df = pd.DataFrame(all_rows)
    print(f"\nTotal scraped: {len(df):,} rows")

    print("\nResolving player IDs via PlayerResolver (Tier 5 — name + position)...")
    resolver = PlayerResolver(min_confidence=0.0)
    resolution_cache = {}
    ids, confs = [], []
    for _, row in df.iterrows():
        cache_key = (row["player_name"], row["position"])
        if cache_key not in resolution_cache:
            resolution_cache[cache_key] = resolver.resolve({
                "name":     row["player_name"],
                "position": row["position"],
            })
        pid, conf = resolution_cache[cache_key]
        ids.append(pid)
        confs.append(conf)
    df["player_id"]  = ids
    df["confidence"] = confs

    print("\nInferring historical team from local nflverse weekly/injury data...")
    df = infer_teams_from_local_data(df)

    matched = df["player_id"].notna().sum()
    print(f"  Matched: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    print(f"  Unique name/position resolutions: {len(resolution_cache):,}")
    resolver.flush_unmatched()

    print(f"\n{'='*60}")
    print("CHECKPOINT: ADP DATA")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nSchema:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isna().sum()
        print(f"  {col:<20} {str(dtype):<12} nulls: {nulls/len(df)*100:.1f}%")
    print(f"\nTeam source breakdown:")
    print(df["team_source"].value_counts(dropna=False).to_string())
    print(f"\nRows by season and format:")
    print(df.groupby(["season", "format"]).size().unstack(fill_value=0).to_string())
    print(f"\nSample (5 rows):")
    print(df[["player_name", "position", "team", "team_source", "avg_adp",
              "format", "season", "player_id", "confidence"]].head(5).to_string())

    confirm("Schema looks good — write Parquet?")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows to {OUT_PATH}")
    print("Done. Review data/adp/ and close FR-16 once satisfied.")


if __name__ == "__main__":
    main()
