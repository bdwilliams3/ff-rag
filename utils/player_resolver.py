"""
FR-13: PlayerResolver — cross-source player identity matching.

Maps any partial player record from any external source to a canonical
nflverse player_id. Import and reuse across all data collection scripts.

Usage:
    from utils.player_resolver import PlayerResolver

    resolver = PlayerResolver()
    player_id, confidence = resolver.resolve({
        "name": "Dre Miller",
        "position": "TE",
        "draft_year": 2021,
        "draft_round": 4,
        "drafting_team": "CIN",
    })
"""

from pathlib import Path
from datetime import date
from typing import Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz

COMBINE_PATH = Path("data/athletic/combine_draft.parquet")
UNMATCHED_DIR = Path("data/unmatched")

# Position family groups — treat members as equivalent when comparing across sources
POSITION_FAMILIES = [
    {"WR", "TE", "FL", "SE"},
    {"RB", "HB", "FB"},
    {"QB"},
    {"CB", "S", "DB", "FS", "SS"},
    {"DE", "DT", "NT", "DL", "EDGE"},
    {"OT", "OG", "C", "OL", "T", "G"},
    {"LB", "OLB", "ILB", "MLB"},
    {"K", "P", "LS"},
]

# Team name/abbreviation normalization → canonical full name
TEAM_MAP = {
    # AFC East
    "BUF": "Buffalo Bills", "BUFFALO": "Buffalo Bills",
    "MIA": "Miami Dolphins", "MIAMI": "Miami Dolphins",
    "NE": "New England Patriots", "NEP": "New England Patriots", "NEW ENGLAND": "New England Patriots",
    "NYJ": "New York Jets", "JETS": "New York Jets",
    # AFC North
    "BAL": "Baltimore Ravens", "BALTIMORE": "Baltimore Ravens",
    "CIN": "Cincinnati Bengals", "CINCINNATI": "Cincinnati Bengals",
    "CLE": "Cleveland Browns", "CLEVELAND": "Cleveland Browns",
    "PIT": "Pittsburgh Steelers", "PITTSBURGH": "Pittsburgh Steelers",
    # AFC South
    "HOU": "Houston Texans", "HOUSTON": "Houston Texans",
    "IND": "Indianapolis Colts", "INDIANAPOLIS": "Indianapolis Colts",
    "JAC": "Jacksonville Jaguars", "JAX": "Jacksonville Jaguars", "JACKSONVILLE": "Jacksonville Jaguars",
    "TEN": "Tennessee Titans", "TENNESSEE": "Tennessee Titans",
    # AFC West
    "DEN": "Denver Broncos", "DENVER": "Denver Broncos",
    "KC": "Kansas City Chiefs", "KAN": "Kansas City Chiefs", "KANSAS CITY": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "OAK": "Las Vegas Raiders", "OAKLAND": "Las Vegas Raiders", "LAS VEGAS": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers", "SD": "Los Angeles Chargers", "SAN DIEGO": "San Diego Chargers", "LOS ANGELES CHARGERS": "Los Angeles Chargers",
    # NFC East
    "DAL": "Dallas Cowboys", "DALLAS": "Dallas Cowboys",
    "NYG": "New York Giants", "GIANTS": "New York Giants",
    "PHI": "Philadelphia Eagles", "PHILADELPHIA": "Philadelphia Eagles",
    "WAS": "Washington Commanders", "WSH": "Washington Commanders", "WASHINGTON": "Washington Commanders",
    # NFC North
    "CHI": "Chicago Bears", "CHICAGO": "Chicago Bears",
    "DET": "Detroit Lions", "DETROIT": "Detroit Lions",
    "GB": "Green Bay Packers", "GNB": "Green Bay Packers", "GREEN BAY": "Green Bay Packers",
    "MIN": "Minnesota Vikings", "MINNESOTA": "Minnesota Vikings",
    # NFC South
    "ATL": "Atlanta Falcons", "ATLANTA": "Atlanta Falcons",
    "CAR": "Carolina Panthers", "CAROLINA": "Carolina Panthers",
    "NO": "New Orleans Saints", "NOR": "New Orleans Saints", "NEW ORLEANS": "New Orleans Saints",
    "TB": "Tampa Bay Buccaneers", "TAM": "Tampa Bay Buccaneers", "TAMPA BAY": "Tampa Bay Buccaneers",
    # NFC West
    "ARI": "Arizona Cardinals", "ARIZONA": "Arizona Cardinals",
    "LAR": "Los Angeles Rams", "LA": "Los Angeles Rams", "STL": "Los Angeles Rams", "ST. LOUIS": "Los Angeles Rams",
    "SF": "San Francisco 49ers", "SFO": "San Francisco 49ers", "SAN FRANCISCO": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "SEATTLE": "Seattle Seahawks",
}


def _normalize_team(team: Optional[str]) -> Optional[str]:
    if not team:
        return None
    normalized = TEAM_MAP.get(str(team).upper().strip())
    if normalized:
        return normalized
    # Fall back to partial match against full names
    t = str(team).strip().lower()
    for full in set(TEAM_MAP.values()):
        if t in full.lower() or full.lower() in t:
            return full
    return str(team).strip()


def _same_position_family(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return True  # unknown position — don't penalize
    a, b = a.upper().strip(), b.upper().strip()
    if a == b:
        return True
    for family in POSITION_FAMILIES:
        if a in family and b in family:
            return True
    return False


import re as _re
_SUFFIX_RE = _re.compile(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)\.?$', _re.IGNORECASE)

def _strip_suffix(name: str) -> str:
    return _SUFFIX_RE.sub('', str(name).strip())


def _name_score(a: str, b: str) -> float:
    """0–100 fuzzy similarity using token sort ratio. Strips generational suffixes first."""
    return fuzz.token_sort_ratio(
        _strip_suffix(a).lower(),
        _strip_suffix(b).lower(),
    )


def _last_name(name: str) -> str:
    parts = _strip_suffix(str(name)).strip().split()
    return parts[-1].lower() if parts else ""


class PlayerResolver:

    def __init__(self, reference_path: Path = COMBINE_PATH, min_confidence: float = 0.50):
        if not Path(reference_path).exists():
            raise FileNotFoundError(
                f"Reference table not found: {reference_path}\n"
                "Run scripts/collect_combine.py first (FR-9)."
            )
        self._ref = pd.read_parquet(reference_path)
        self._min_confidence = min_confidence
        self._unmatched: list[dict] = []
        UNMATCHED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve(self, record: dict) -> Tuple[Optional[str], float]:
        """
        Map a partial player record to a canonical nflverse player_id.

        Args:
            record: dict with any of: name, position, college,
                    draft_year, draft_round, draft_pick, drafting_team

        Returns:
            (player_id, confidence) — player_id is None if no match above
            min_confidence threshold. Unmatched records are queued for export.
        """
        player_id, confidence = self._match(record)

        if confidence < self._min_confidence or player_id is None:
            self._unmatched.append({**record, "_confidence": confidence})
            return None, confidence

        return player_id, confidence

    def flush_unmatched(self) -> int:
        """Write queued unmatched records to CSV. Returns count written."""
        if not self._unmatched:
            return 0
        path = UNMATCHED_DIR / f"resolver_unmatched_{date.today()}.csv"
        pd.DataFrame(self._unmatched).to_csv(path, index=False, mode="a",
                                              header=not path.exists())
        count = len(self._unmatched)
        self._unmatched.clear()
        return count

    # ------------------------------------------------------------------
    # Matching tiers
    # ------------------------------------------------------------------

    def _match(self, r: dict) -> Tuple[Optional[str], float]:
        draft_year  = r.get("draft_year")
        draft_round = r.get("draft_round")
        draft_pick  = r.get("draft_pick")   # overall pick number
        team        = _normalize_team(r.get("drafting_team"))
        name        = r.get("name", "")
        position    = r.get("position", "")
        college     = r.get("college", "")

        ref = self._ref

        # ------ Tier 1: draft_year + draft_pick (overall) ------
        # Every overall pick in every year is unique — true primary key
        if draft_year and draft_pick:
            mask = (
                (ref["draft_year"] == float(draft_year)) &
                (ref["draft_pick"] == float(draft_pick))
            )
            hit = ref[mask]
            if len(hit) == 1:
                return hit.iloc[0]["player_id"], 1.0

        # ------ Tier 2: draft_year + draft_round + team + fuzzy last name ------
        if draft_year and draft_round and team and name:
            mask = (
                (ref["draft_year"] == float(draft_year)) &
                (ref["draft_round"] == float(draft_round)) &
                (ref["draft_team"] == team)
            )
            candidates = ref[mask]
            if not candidates.empty:
                candidates = candidates.copy()
                candidates["_score"] = candidates["player_name"].apply(
                    lambda n: _name_score(str(n).split()[-1], _last_name(name))
                )
                best = candidates.loc[candidates["_score"].idxmax()]
                if best["_score"] >= 85:
                    return best["player_id"], 0.90

        # ------ Tier 3: fuzzy full name + college + position family ------
        if name and college:
            mask = ref["college"].str.lower().str.strip() == college.lower().strip()
            candidates = ref[mask]
            if not candidates.empty:
                candidates = candidates.copy()
                candidates["_score"] = candidates["player_name"].apply(
                    lambda n: _name_score(n, name)
                )
                candidates["_pos_ok"] = candidates["position"].apply(
                    lambda p: _same_position_family(p, position)
                )
                valid = candidates[candidates["_pos_ok"]]
                if not valid.empty:
                    best = valid.loc[valid["_score"].idxmax()]
                    if best["_score"] >= 80:
                        return best["player_id"], 0.75

        # ------ Tier 4: fuzzy name + draft_year only ------
        if name and draft_year:
            mask = ref["draft_year"] == float(draft_year)
            candidates = ref[mask].copy()
            if not candidates.empty:
                candidates["_last_score"] = candidates["player_name"].apply(
                    lambda n: _name_score(str(n).split()[-1], _last_name(name))
                )
                candidates["_first_score"] = candidates["player_name"].apply(
                    lambda n: _name_score(
                        str(n).split()[0] if str(n).split() else "",
                        name.split()[0] if name.split() else ""
                    )
                )
                # Require exact last name match + fuzzy first
                exact_last = candidates[candidates["_last_score"] >= 100]
                if not exact_last.empty:
                    best = exact_last.loc[exact_last["_first_score"].idxmax()]
                    if best["_first_score"] >= 70:
                        return best["player_id"], 0.50

        # ------ Tier 5: fuzzy full name + position family only ------
        # Used for sources (e.g. ADP) with no draft info or college.
        # High name threshold (90) + position check limits false positives.
        if name and position:
            candidates = ref.copy()
            candidates["_score"] = candidates["player_name"].apply(
                lambda n: _name_score(n, name)
            )
            candidates["_pos_ok"] = candidates["position"].apply(
                lambda p: _same_position_family(p, position)
            )
            valid = candidates[candidates["_pos_ok"]]
            if not valid.empty:
                best = valid.loc[valid["_score"].idxmax()]
                if best["_score"] >= 88:
                    return best["player_id"], 0.35

        return None, 0.0
