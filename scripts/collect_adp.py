"""
FR-16: FantasyPros historical ADP — standard, PPR, half-PPR, 2017-present.
Scrapes overall ADP pages, resolves player_id via PlayerResolver (Tier 5).
Run: .venv/bin/python3 scripts/collect_adp.py [--auto-confirm]
"""

import sys
import time
from pathlib import Path
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

OUT_PATH = Path("data/adp/adp_historical.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
AUTO_CONFIRM = "--auto-confirm" in sys.argv


def _strip_pos(pos: str) -> str:
    """'WR1' → 'WR', 'RB3' → 'RB'"""
    return pos.rstrip("0123456789")


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

        # Player name from <a> tag; team from first <small>
        player_td = tds[1]
        name_tag = player_td.find("a", class_="player-name")
        if not name_tag:
            continue
        player_name = name_tag.get_text(strip=True)
        smalls = player_td.find_all("small")
        team = smalls[0].get_text(strip=True) if smalls else ""

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
            "team":        team,
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
    ids, confs = [], []
    for _, row in df.iterrows():
        pid, conf = resolver.resolve({
            "name":     row["player_name"],
            "position": row["position"],
        })
        ids.append(pid)
        confs.append(conf)
    df["player_id"]  = ids
    df["confidence"] = confs

    matched = df["player_id"].notna().sum()
    print(f"  Matched: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    resolver.flush_unmatched()

    print(f"\n{'='*60}")
    print("CHECKPOINT: ADP DATA")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nSchema:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isna().sum()
        print(f"  {col:<20} {str(dtype):<12} nulls: {nulls/len(df)*100:.1f}%")
    print(f"\nRows by season and format:")
    print(df.groupby(["season", "format"]).size().unstack(fill_value=0).to_string())
    print(f"\nSample (5 rows):")
    print(df[["player_name", "position", "team", "avg_adp",
              "format", "season", "player_id", "confidence"]].head(5).to_string())

    confirm("Schema looks good — write Parquet?")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows to {OUT_PATH}")
    print("Done. Review data/adp/ and close FR-16 once satisfied.")


if __name__ == "__main__":
    main()
