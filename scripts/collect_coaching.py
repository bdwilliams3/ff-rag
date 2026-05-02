"""
FR-14: Coaching staff history by team/week.

Scrapes Pro Football History franchise pages for HC/OC/DC staff assignments,
expands them to one row per team per regular-season week, and writes:
data/coaching/coaching_weekly.parquet

Run: .venv/bin/python3 scripts/collect_coaching.py [--auto-confirm]
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


YEARS = list(range(2012, 2026))
OUT_PATH = Path("data/coaching/coaching_weekly.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://pro-football-history.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CRAWL_DELAY_SECONDS = 1.2
AUTO_CONFIRM = "--auto-confirm" in sys.argv

TEAM_CODES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

ROLE_MAP = {
    "Head Coach": "hc",
    "Interim Head Coach": "hc_interim",
    "Offensive Coordinator": "oc",
    "Interim Offensive Coordinator": "oc_interim",
    "Defensive Coordinator": "dc",
    "Interim Defensive Coordinator": "dc_interim",
}


def get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.text


def confirm(prompt: str) -> None:
    if AUTO_CONFIRM:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def parse_franchises() -> list[dict]:
    soup = BeautifulSoup(get(f"{BASE_URL}/franchises"), "html.parser")
    franchises = []
    seen = set()
    for a in soup.find_all("a", href=True):
        name = a.get_text(" ", strip=True)
        href = a["href"]
        if name not in TEAM_CODES or "/franchise/" not in href or not href.endswith("-coaches"):
            continue
        if name in seen:
            continue
        seen.add(name)
        franchises.append({"team_name": name, "team": TEAM_CODES[name], "url": f"{BASE_URL}{href}"})
    return sorted(franchises, key=lambda x: x["team"])


def names_after_role(strong: Tag) -> list[str]:
    names = []
    for sibling in strong.next_siblings:
        if isinstance(sibling, Tag) and sibling.name == "strong":
            break
        if isinstance(sibling, Tag):
            for a in sibling.find_all("a"):
                name = a.get_text(" ", strip=True)
                if name:
                    names.append(name)
            if sibling.name == "a":
                name = sibling.get_text(" ", strip=True)
                if name:
                    names.append(name)
    return names


def parse_staff_cell(cell: Tag) -> dict[str, str | None]:
    roles = {v: None for v in ROLE_MAP.values()}
    for strong in cell.find_all("strong"):
        role = strong.get_text(" ", strip=True).rstrip(":")
        key = ROLE_MAP.get(role)
        if not key:
            continue
        names = names_after_role(strong)
        if names:
            roles[key] = " / ".join(dict.fromkeys(names))
    return roles


def parse_franchise_page(franchise: dict) -> list[dict]:
    soup = BeautifulSoup(get(franchise["url"]), "html.parser")
    rows = []
    table = next((t for t in soup.find_all("table") if t.find(id=re.compile(r"franchise_history_"))), None)
    if table is None:
        raise RuntimeError(f"No franchise history table found for {franchise['team_name']}")

    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        year_text = cells[0].get_text(" ", strip=True)
        m = re.match(r"(\d{4})\s+(.+)", year_text)
        if not m:
            continue
        season = int(m.group(1))
        if season not in YEARS:
            continue
        staff = parse_staff_cell(cells[2])
        season_link = cells[0].find("a", href=True)
        rows.append({
            "season": season,
            "team": franchise["team"],
            "team_name": franchise["team_name"],
            "record": cells[1].get_text(" ", strip=True),
            "season_url": f"{BASE_URL}{season_link['href']}" if season_link else None,
            **staff,
        })
    return rows


def parse_firing_date(season_url: str | None) -> pd.Timestamp | None:
    if not season_url:
        return None
    html = get(season_url)
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    m = re.search(r"\b(?:was|were) fired .*? on [A-Za-z]+, ([A-Za-z]+ \d{1,2}, \d{4})", text)
    if not m:
        return None
    try:
        return pd.Timestamp(datetime.strptime(m.group(1), "%B %d, %Y").date())
    except ValueError:
        return None


def regular_season_weeks() -> pd.DataFrame:
    schedules = nfl.import_schedules(YEARS)
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    weeks = []
    for season, g in schedules.groupby("season"):
        max_week = int(g["week"].max())
        teams = sorted(set(g["home_team"]) | set(g["away_team"]))
        for team in teams:
            team_games = g[(g["home_team"] == team) | (g["away_team"] == team)]
            week_dates = dict(zip(team_games["week"], pd.to_datetime(team_games["gameday"])))
            for week in range(1, max_week + 1):
                weeks.append({
                    "season": int(season),
                    "week": int(week),
                    "team": team,
                    "gameday": week_dates.get(week),
                    "had_game": week in week_dates,
                })
    return pd.DataFrame(weeks)


def expand_weekly(season_staff: pd.DataFrame) -> pd.DataFrame:
    weeks = regular_season_weeks()
    df = weeks.merge(season_staff, on=["season", "team"], how="left")

    needs_dates = df[
        (df["hc_interim"].notna() | df["oc_interim"].notna() | df["dc_interim"].notna())
        & df["season_url"].notna()
    ][["season", "team", "season_url"]].drop_duplicates()

    switch_start_weeks = {}
    for row in needs_dates.itertuples(index=False):
        key = (row.season, row.team)
        print(f"  fetching firing date for {row.team} {row.season}")
        fired = parse_firing_date(row.season_url)
        if fired is not None:
            week_rows = weeks[
                (weeks["season"] == row.season)
                & (weeks["team"] == row.team)
                & (weeks["gameday"].notna())
                & (weeks["gameday"] > fired)
            ]
            if not week_rows.empty:
                switch_start_weeks[key] = int(week_rows["week"].min())
        time.sleep(CRAWL_DELAY_SECONDS)

    def after_firing(row: pd.Series) -> bool:
        start_week = switch_start_weeks.get((row["season"], row["team"]))
        return start_week is not None and row["week"] >= start_week

    switch_mask = df.apply(after_firing, axis=1)
    for role in ["hc", "oc", "dc"]:
        interim_col = f"{role}_interim"
        active = switch_mask & df[interim_col].notna()
        df.loc[active, role] = df.loc[active, interim_col]
        df[f"{role}_interim_flag"] = active

    for role in ["hc", "oc", "dc"]:
        df[role] = df[role].fillna(f"No official {role.upper()} listed")

    return df[[
        "season", "week", "team", "team_name", "gameday", "had_game",
        "hc", "oc", "dc",
        "hc_interim_flag", "oc_interim_flag", "dc_interim_flag",
        "record", "season_url",
    ]].rename(columns={
        "hc": "hc_name",
        "oc": "oc_name",
        "dc": "dc_name",
        "hc_interim_flag": "hc_interim",
        "oc_interim_flag": "oc_interim",
        "dc_interim_flag": "dc_interim",
    }).sort_values(["season", "team", "week"]).reset_index(drop=True)


def print_checkpoint(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("CHECKPOINT: COACHING WEEKLY")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nSeasons: {df['season'].min()}-{df['season'].max()}")
    print(f"Teams per season:\n{df.groupby('season')['team'].nunique().to_string()}")
    for col in ["hc_name", "oc_name", "dc_name"]:
        nulls = df[col].isna().sum()
        print(f"{col:<10} nulls: {nulls:,} ({nulls / len(df) * 100:.1f}%)")
    print("\nInterim rows:")
    print(df[["hc_interim", "oc_interim", "dc_interim"]].sum().to_string())
    print("\nKnown mid-season checks:")
    checks = df[
        ((df["season"] == 2023) & (df["team"].isin(["LV", "CAR", "LAC"])))
        | ((df["season"] == 2022) & (df["team"].isin(["CAR", "IND"])))
    ]
    print(checks[["season", "week", "team", "hc_name", "oc_name", "dc_name", "hc_interim", "oc_interim"]].to_string(index=False))


def main() -> None:
    franchises = parse_franchises()
    print(f"Found {len(franchises)} active franchises")

    rows = []
    for i, franchise in enumerate(franchises, 1):
        print(f"  [{i:02d}/{len(franchises)}] {franchise['team_name']}")
        rows.extend(parse_franchise_page(franchise))
        time.sleep(CRAWL_DELAY_SECONDS)

    season_staff = pd.DataFrame(rows)
    weekly = expand_weekly(season_staff)
    print_checkpoint(weekly)
    confirm("Schema looks good — write coaching_weekly.parquet?")

    weekly.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(weekly):,} rows to {OUT_PATH}")
    print("Done. Review data/coaching/ and close FR-14 once satisfied.")


if __name__ == "__main__":
    main()
