"""
FR-17: Redraft league-winner roster frequency from public MFL leagues.

Collects public MyFantasyLeague redraft leagues, identifies the championship
matchup, counts every player rostered by the champion in that matchup
(starters + nonstarters), resolves to canonical nflverse player_id, and writes:

data/winners/league_winner_frequency.parquet

Run:
  .venv/bin/python3 scripts/collect_league_winners.py [--auto-confirm]
  .venv/bin/python3 scripts/collect_league_winners.py --max-leagues-per-season 50 --auto-confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Optional

import pandas as pd
import requests

sys.path.insert(0, ".")
from utils.player_resolver import PlayerResolver


YEARS = list(range(2012, 2026))
SEARCH_TERMS = ["redraft", "ppr", "half ppr", "standard"]
OUT_PATH = Path("data/winners/league_winner_frequency.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("data/cache/mfl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 2.0
POSITIONS = {"QB", "RB", "WR", "TE"}
EXCLUDE_NAME_RE = re.compile(
    r"best\s*ball|bestball|draft\s*master|draftmaster|dynasty|keeper|devy|empire|contract|salary|auction|playoff",
    re.IGNORECASE,
)
TEAM_NORMALIZE = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GBP": "GB",
    "GNB": "GB", "GB": "GB", "HOU": "HOU", "IND": "IND", "JAC": "JAX", "JAX": "JAX",
    "KCC": "KC", "KC": "KC", "LAC": "LAC", "LAR": "LA", "RAM": "LA", "LVR": "LV",
    "OAK": "LV", "MIA": "MIA", "MIN": "MIN", "NEP": "NE", "NE": "NE", "NOS": "NO",
    "NO": "NO", "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT",
    "SEA": "SEA", "SFO": "SF", "SF": "SF", "TBB": "TB", "TB": "TB", "TEN": "TEN",
    "WAS": "WAS", "WSH": "WAS",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--auto-confirm", action="store_true")
    p.add_argument("--start-year", type=int, default=min(YEARS))
    p.add_argument("--end-year", type=int, default=max(YEARS))
    p.add_argument("--max-leagues-per-season", type=int, default=100)
    return p.parse_args()


def confirm(prompt: str, auto_confirm: bool) -> None:
    if auto_confirm:
        print(f"[auto-confirm] {prompt}")
        return
    answer = input(f"\n{prompt} [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def load_mfl_api_key() -> Optional[str]:
    if os.environ.get("MFL_API_KEY"):
        return os.environ["MFL_API_KEY"].strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("MFL_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


MFL_API_KEY = load_mfl_api_key()


def cache_path(year: int, host: str, params: dict) -> Path:
    safe_params = {k: v for k, v in params.items() if k != "APIKEY"}
    key = urlencode(sorted(safe_params.items()))
    digest = hashlib.sha256(f"{year}|{host}|{key}".encode()).hexdigest()[:24]
    req_type = str(params.get("TYPE", "request"))
    return CACHE_DIR / str(year) / f"{req_type}_{digest}.json"


def mfl_get(year: int, params: dict, base_url: Optional[str] = None) -> dict:
    host = base_url or f"https://api.myfantasyleague.com/{year}"
    url = f"{host}/export"
    request_params = {**params, "JSON": "1"}
    if MFL_API_KEY:
        request_params["APIKEY"] = MFL_API_KEY
    path = cache_path(year, host, request_params)
    if path.exists():
        return json.loads(path.read_text())

    last_error = None
    for attempt in range(4):
        with requests.get(
            url,
            params=request_params,
            timeout=(5, 20),
            headers={"Connection": "close"},
        ) as r:
            if r.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"    rate limited; sleeping {wait}s", flush=True)
                time.sleep(wait)
                last_error = requests.HTTPError(f"429 for {r.url}", response=r)
                continue
            r.raise_for_status()
            data = r.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        time.sleep(REQUEST_DELAY)
        return data
    raise last_error or RuntimeError(f"MFL request failed: {url}")


def as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def discover_leagues(year: int) -> list[dict]:
    leagues: dict[str, dict] = {}
    for term in SEARCH_TERMS:
        data = mfl_get(year, {"TYPE": "leagueSearch", "SEARCH": term})
        items = as_list(data.get("leagues", {}).get("league"))
        for item in items:
            lid = str(item.get("id", "")).strip()
            if lid:
                leagues[lid] = item
    return sorted(leagues.values(), key=lambda x: x.get("id", ""))


def normalize_player_name(name: str) -> str:
    name = str(name or "").strip()
    if "," not in name:
        return name
    last, first = [p.strip() for p in name.split(",", 1)]
    return f"{first} {last}".strip()


def normalize_team(team: Optional[str]) -> Optional[str]:
    if not team:
        return None
    return TEAM_NORMALIZE.get(str(team).upper().strip(), str(team).upper().strip())


def passes_league_filter(search_row: dict, league: dict) -> tuple[bool, str]:
    name = str(league.get("name") or search_row.get("name") or "")
    if EXCLUDE_NAME_RE.search(name):
        return False, "name_excluded"
    if str(league.get("bestLineup", "No")).lower() not in {"no", "0", "false"}:
        return False, "best_lineup"
    if str(league.get("taxiSquad", "0")) not in {"0", "", "No", "no"}:
        return False, "taxi"
    if str(league.get("usesContractYear", "0")) not in {"0", "", "No", "no"}:
        return False, "contracts"
    if str(league.get("usesSalaries", "0")) not in {"0", "", "No", "no"}:
        return False, "salaries"
    if str(league.get("h2h", "YES")).upper() not in {"YES", "1", "TRUE"}:
        return False, "not_h2h"
    starters = league.get("starters", {})
    if not starters or int(str(starters.get("count", "0") or "0")) <= 0:
        return False, "no_starters"
    return True, "ok"


def cc_points(rule: dict) -> Optional[float]:
    event = (rule.get("event") or {}).get("$t")
    if event != "CC":
        return None
    points = str((rule.get("points") or {}).get("$t", "")).replace("*", "")
    try:
        return float(points)
    except ValueError:
        return None


def infer_scoring_format(rules: dict) -> tuple[str, bool]:
    position_rules = as_list(rules.get("positionRules"))
    base_points: list[float] = []
    te_points: list[float] = []
    for pos_rule in position_rules:
        positions = str(pos_rule.get("positions", ""))
        ruleset = as_list(pos_rule.get("rule"))
        points = [p for p in (cc_points(r) for r in ruleset) if p is not None]
        if not points:
            continue
        if positions == "TE":
            te_points.extend(points)
        if any(p in positions.split("|") for p in ["QB", "RB", "WR"]):
            base_points.extend(points)
    base = max(base_points) if base_points else 0.0
    te = max(te_points) if te_points else base
    if base >= 0.95:
        fmt = "ppr"
    elif base >= 0.45:
        fmt = "half_ppr"
    elif base == 0:
        fmt = "std"
    else:
        fmt = "custom"
    return fmt, te > base


def primary_bracket(playoff_brackets: dict) -> Optional[dict]:
    brackets = as_list(playoff_brackets.get("playoffBrackets", {}).get("playoffBracket"))
    if not brackets:
        return None
    for b in brackets:
        title = f"{b.get('name','')} {b.get('bracketWinnerTitle','')}".lower()
        if "champion" in title and "3rd" not in title and "consolation" not in title:
            return b
    return brackets[0]


def championship_game(playoff_bracket: dict) -> Optional[tuple[int, str, str]]:
    rounds = as_list(playoff_bracket.get("playoffBracket", {}).get("playoffRound"))
    if not rounds:
        return None
    final_round = max(rounds, key=lambda r: int(r.get("week", 0)))
    games = as_list(final_round.get("playoffGame"))
    if not games:
        return None
    game = games[0]
    home = game.get("home", {})
    away = game.get("away", {})
    try:
        home_pts = float(home.get("points", 0))
        away_pts = float(away.get("points", 0))
    except ValueError:
        return None
    winner = home.get("franchise_id") if home_pts >= away_pts else away.get("franchise_id")
    return int(final_round["week"]), str(game.get("game_id")), str(winner)


def rostered_players_from_weekly_results(weekly_results: dict, winner_id: str) -> list[str]:
    for matchup in as_list(weekly_results.get("weeklyResults", {}).get("matchup")):
        if str(matchup.get("regularSeason", "0")) not in {"0", "false", "False"}:
            continue
        franchises = as_list(matchup.get("franchise"))
        ids = {str(f.get("id")) for f in franchises}
        if winner_id not in ids:
            continue
        for franchise in franchises:
            if str(franchise.get("id")) != winner_id:
                continue
            starters = [p for p in str(franchise.get("starters", "")).split(",") if p]
            nonstarters = [p for p in str(franchise.get("nonstarters", "")).split(",") if p]
            if starters or nonstarters:
                return list(dict.fromkeys(starters + nonstarters))
            return [
                str(p.get("id"))
                for p in as_list(franchise.get("player"))
                if p.get("id") and p.get("status") in {"starter", "nonstarter"}
            ]
    return []


def load_mfl_players(year: int) -> dict[str, dict]:
    data = mfl_get(year, {"TYPE": "players"})
    return {str(p.get("id")): p for p in as_list(data.get("players", {}).get("player"))}


def resolve_mfl_players(rows: list[dict], players_by_year: dict[int, dict[str, dict]]) -> list[dict]:
    resolver = PlayerResolver(min_confidence=0.0)
    cache: dict[tuple[int, str], tuple[Optional[str], float, Optional[str], Optional[str], Optional[str]]] = {}
    out = []
    for row in rows:
        key = (row["season"], row["mfl_player_id"])
        if key not in cache:
            p = players_by_year[row["season"]].get(row["mfl_player_id"], {})
            name = normalize_player_name(p.get("name", ""))
            position = str(p.get("position", "") or "")
            team = normalize_team(p.get("team"))
            if position not in POSITIONS:
                cache[key] = (None, 0.0, name, position, team)
            else:
                player_id, conf = resolver.resolve({"name": name, "position": position})
                cache[key] = (player_id, conf, name, position, team)
        player_id, conf, name, position, team = cache[key]
        if player_id:
            out.append({**row, "player_id": player_id, "player_name": name, "position": position, "team": team, "resolver_confidence": conf})
    resolver.flush_unmatched()
    return out


def collect(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    league_rows = []
    players_by_year: dict[int, dict[str, dict]] = {}
    years = [y for y in YEARS if args.start_year <= y <= args.end_year]

    for year in years:
        candidates = discover_leagues(year)
        print(f"\n{year}: {len(candidates)} candidate public leagues", flush=True)
        players_by_year[year] = load_mfl_players(year)
        accepted = 0
        rejects = Counter()
        for search_row in candidates:
            if accepted >= args.max_leagues_per_season:
                break
            lid = str(search_row["id"])
            try:
                league = mfl_get(year, {"TYPE": "league", "L": lid})
                league = league.get("league", {})
                ok, reason = passes_league_filter(search_row, league)
                if not ok:
                    rejects[reason] += 1
                    continue
                base_url = league.get("baseURL")
                rules = mfl_get(year, {"TYPE": "rules", "L": lid}, base_url=base_url).get("rules", {})
                scoring_format, te_premium = infer_scoring_format(rules)
                brackets = mfl_get(year, {"TYPE": "playoffBrackets", "L": lid}, base_url=base_url)
                bracket = primary_bracket(brackets)
                if not bracket:
                    rejects["no_bracket"] += 1
                    continue
                bracket_detail = mfl_get(year, {"TYPE": "playoffBracket", "L": lid, "BRACKET_ID": bracket["id"]}, base_url=base_url)
                champ = championship_game(bracket_detail)
                if not champ:
                    rejects["no_championship_game"] += 1
                    continue
                championship_week, championship_game_id, winner_id = champ
                weekly = mfl_get(year, {"TYPE": "weeklyResults", "L": lid, "W": championship_week}, base_url=base_url)
                player_ids = rostered_players_from_weekly_results(weekly, winner_id)
                if not player_ids:
                    rejects["no_winner_roster"] += 1
                    continue

                accepted += 1
                print(f"  accepted {accepted}: {lid} {scoring_format}{' TE+' if te_premium else ''} ({len(player_ids)} players)", flush=True)
                league_rows.append({
                    "season": year,
                    "league_id": lid,
                    "league_name": league.get("name"),
                    "scoring_format": scoring_format,
                    "te_premium": te_premium,
                    "championship_week": championship_week,
                    "championship_game_id": championship_game_id,
                    "winner_franchise_id": winner_id,
                    "rostered_players_count": len(player_ids),
                })
                for mfl_player_id in player_ids:
                    raw_rows.append({
                        "season": year,
                        "league_id": lid,
                        "league_type": "redraft",
                        "scoring_format": scoring_format,
                        "te_premium": te_premium,
                        "mfl_player_id": str(mfl_player_id),
                    })
            except Exception as e:
                rejects[f"error:{type(e).__name__}"] += 1
                continue

        print(f"  accepted {accepted}; rejects {dict(rejects)}", flush=True)

    resolved_rows = resolve_mfl_players(raw_rows, players_by_year)
    if not resolved_rows:
        return pd.DataFrame(), pd.DataFrame(league_rows)

    raw = pd.DataFrame(resolved_rows)
    grouped_cols = ["player_id", "player_name", "position", "season", "league_type", "scoring_format", "te_premium"]
    appearances = (
        raw.drop_duplicates(["season", "league_id", "scoring_format", "te_premium", "player_id"])
        .groupby(grouped_cols, as_index=False)
        .agg(championship_roster_appearances=("league_id", "nunique"))
    )
    totals = (
        pd.DataFrame(league_rows)
        .groupby(["season", "league_type", "scoring_format", "te_premium"], as_index=False)
        .agg(total_leagues_sampled=("league_id", "nunique"))
    )
    out = appearances.merge(totals, on=["season", "league_type", "scoring_format", "te_premium"], how="left")
    out["roster_appearance_rate"] = out["championship_roster_appearances"] / out["total_leagues_sampled"]
    out["sample_source"] = "mfl"
    out = out.sort_values(["season", "scoring_format", "te_premium", "roster_appearance_rate", "championship_roster_appearances"], ascending=[True, True, True, False, False]).reset_index(drop=True)
    return out, pd.DataFrame(league_rows)


def print_checkpoint(df: pd.DataFrame, leagues: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("CHECKPOINT: LEAGUE WINNER FREQUENCY")
    print(f"{'='*60}")
    print(f"\nLeague samples: {len(leagues):,}")
    if not leagues.empty:
        print(leagues.groupby(["season", "scoring_format", "te_premium"]).size().to_string())
    print(f"\nOutput rows: {len(df):,}")
    if not df.empty:
        print("\nTop 20 appearance rates with at least 5 sampled leagues:")
        cols = ["season", "scoring_format", "te_premium", "player_name", "position", "championship_roster_appearances", "total_leagues_sampled", "roster_appearance_rate"]
        print(df[df["total_leagues_sampled"] >= 5][cols].head(20).to_string(index=False))


def main() -> None:
    args = parse_args()
    df, leagues = collect(args)
    print_checkpoint(df, leagues)
    confirm(f"Write {len(df):,} rows to {OUT_PATH}?", args.auto_confirm)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
