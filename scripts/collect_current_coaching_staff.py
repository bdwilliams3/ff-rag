"""
FR-39: Current coaching destinations and links to historical staff profiles.

Builds data/coaching/current_coaching_staff.parquet from current public staff
tables, then enriches each current HC/OC with local historical coordinator
tendency profile references. This keeps "who is currently on staff" separate
from "what that coach/team did historically".

Run:
    .venv/bin/python3 scripts/collect_current_coaching_staff.py
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


OUT_PATH = Path("data/coaching/current_coaching_staff.parquet")
OFFENSE_PROFILE_PATH = Path("data/coaching/offensive_coordinator_profiles.parquet")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

HEAD_COACH_URL = "https://en.wikipedia.org/wiki/List_of_current_NFL_head_coaches"
OFFENSIVE_COORDINATOR_URL = "https://en.wikipedia.org/wiki/List_of_current_NFL_offensive_coordinators"

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


def clean_text(value: str) -> str:
    value = re.sub(r"\[.*?\]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_head_coaches() -> pd.DataFrame:
    rows = []
    table = soup(HEAD_COACH_URL).find("table", class_="wikitable")
    if table is None:
        raise RuntimeError("Could not find current NFL head coach table")
    for tr in table.find_all("tr"):
        cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if not cells or cells[0] not in TEAM_CODES:
            continue
        rows.append({
            "team_name": cells[0],
            "team": TEAM_CODES[cells[0]],
            "current_hc_name": cells[2],
            "current_hc_since": cells[3],
            "current_hc_source_url": HEAD_COACH_URL,
        })
    return pd.DataFrame(rows)


def parse_offensive_coordinators() -> pd.DataFrame:
    rows = []
    page = soup(OFFENSIVE_COORDINATOR_URL)
    for table in page.find_all("table", class_="wikitable"):
        for tr in table.find_all("tr"):
            cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if not cells or cells[0] not in TEAM_CODES:
                continue
            rows.append({
                "team_name": cells[0],
                "team": TEAM_CODES[cells[0]],
                "current_oc_name": cells[1],
                "current_oc_since": cells[2],
                "current_oc_previous_position": cells[3] if len(cells) > 3 else None,
                "current_oc_source_url": OFFENSIVE_COORDINATOR_URL,
            })
    return pd.DataFrame(rows)


def historical_summary(profiles: pd.DataFrame, coach_column: str, prefix: str) -> pd.DataFrame:
    grouped = (
        profiles.groupby(coach_column, as_index=False)
        .agg(
            profile_count=("season", "count"),
            seasons=("season", lambda s: ",".join(str(int(v)) for v in sorted(s.dropna().unique()))),
            teams=("team", lambda s: ",".join(sorted(str(v) for v in s.dropna().unique()))),
            avg_pass_rate=("pass_rate", "mean"),
            avg_run_rate=("run_rate", "mean"),
            avg_wr_target_share=("wr_target_share", "mean"),
            avg_te_target_share=("te_target_share", "mean"),
            avg_rb_target_share=("rb_target_share", "mean"),
        )
    )
    return grouped.rename(columns={
        coach_column: f"{prefix}_historical_profile_name",
        "profile_count": f"{prefix}_historical_oc_profile_count",
        "seasons": f"{prefix}_historical_oc_profile_seasons",
        "teams": f"{prefix}_historical_oc_profile_teams",
        "avg_pass_rate": f"{prefix}_historical_avg_pass_rate",
        "avg_run_rate": f"{prefix}_historical_avg_run_rate",
        "avg_wr_target_share": f"{prefix}_historical_avg_wr_target_share",
        "avg_te_target_share": f"{prefix}_historical_avg_te_target_share",
        "avg_rb_target_share": f"{prefix}_historical_avg_rb_target_share",
    })


def add_historical_links(current: pd.DataFrame) -> pd.DataFrame:
    if not OFFENSE_PROFILE_PATH.exists():
        return current
    profiles = pd.read_parquet(OFFENSE_PROFILE_PATH)
    hc_hist = historical_summary(profiles, "oc_name", "hc")
    oc_hist = historical_summary(profiles, "oc_name", "oc")
    current = current.merge(
        hc_hist,
        left_on="current_hc_name",
        right_on="hc_historical_profile_name",
        how="left",
    )
    current = current.merge(
        oc_hist,
        left_on="current_oc_name",
        right_on="oc_historical_profile_name",
        how="left",
    )
    for prefix in ["hc", "oc"]:
        count_col = f"{prefix}_historical_oc_profile_count"
        if count_col in current:
            current[count_col] = current[count_col].fillna(0).astype(int)
    return current


def main() -> None:
    hc = parse_head_coaches()
    oc = parse_offensive_coordinators()
    if len(hc) != 32:
        raise RuntimeError(f"Expected 32 head coach rows, found {len(hc)}")
    if len(oc) != 32:
        raise RuntimeError(f"Expected 32 offensive coordinator rows, found {len(oc)}")

    current = hc.merge(
        oc.drop(columns=["team_name"]),
        on="team",
        how="inner",
        validate="one_to_one",
    )
    current["as_of_date"] = date.today().isoformat()
    current["source_name"] = "Wikipedia current NFL coach/coordinator lists"
    current["source_confidence"] = "secondary_current_index"
    current = add_historical_links(current)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current.sort_values("team").to_parquet(OUT_PATH, index=False)

    print(f"Wrote {len(current)} rows -> {OUT_PATH}")
    print(current[[
        "team",
        "team_name",
        "current_hc_name",
        "current_oc_name",
        "oc_historical_oc_profile_count",
        "oc_historical_oc_profile_seasons",
    ]].sort_values("team").to_string(index=False))


if __name__ == "__main__":
    main()
