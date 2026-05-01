# FF RAG — Fantasy Football Retrieval-Augmented Generation

A local RAG system that provides an LLM with deep historical context for fantasy football draft decisions. Query the system during a live draft and get data-backed answers in under 3 seconds.

---

## How It Works

1. **Data collection scripts** pull historical NFL and fantasy data from public APIs and scrapers, storing everything as Parquet files.
2. **PlayerResolver** (`utils/player_resolver.py`) normalizes player identity across all sources using draft metadata and fuzzy name matching — every record joins to a canonical nflverse `player_id`.
3. **Embedding pipeline** chunks and embeds all Parquet data into a local Qdrant vector store running in Docker.
4. **Go retrieval script** (`cmd/query/main.go`) hits the vector store and returns ranked context chunks in <300ms.
5. **LLM integration** assembles retrieved context into a prompt and returns a draft recommendation via CLI.

---

## Architecture

```
Data Collection (Python)
  ├── nfl_data_py / nflverse      → stats, combine, injuries
  ├── CFBD API                    → college stats
  ├── PFR scraper                 → coaching history, 2025 stats
  ├── FantasyPros scraper         → ADP
  └── Sleeper API + BBM           → league winner frequency

PlayerResolver (Python)
  └── utils/player_resolver.py   → cross-source player identity normalization

Vector Store (Docker)
  └── Qdrant                     → persistent local container

Embedding Pipeline (Python)
  └── scripts/embed.py           → chunks Parquet → embeds → writes to Qdrant

Retrieval (Go)
  └── cmd/query/main.go          → query Qdrant, return JSON context chunks

LLM Integration
  └── cmd/draft/main.go          → retrieval + LLM call + response to CLI
```

---

## Data Epics

| Epic | Description | Key Tasks |
|------|-------------|-----------|
| [FR-1](https://aispm.atlassian.net/browse/FR-1) | Historical Stats | nflverse backfill, PFR enrichment, in-season webhook, injury history |
| [FR-2](https://aispm.atlassian.net/browse/FR-2) | Athletic Profile | Combine + draft data, CFBD college stats, prospect confidence score |
| [FR-3](https://aispm.atlassian.net/browse/FR-3) | Coaching History | HC/OC/DC per team per week, scheme tendency metrics |
| [FR-4](https://aispm.atlassian.net/browse/FR-4) | ADP | FantasyPros historical ADP by player/format/year |
| [FR-5](https://aispm.atlassian.net/browse/FR-5) | League Winners | Championship roster frequency via Sleeper + best ball tournaments |
| [FR-12](https://aispm.atlassian.net/browse/FR-12) | Embed Utilities | PlayerResolver, Docker vector DB, embedding pipeline |
| [FR-19](https://aispm.atlassian.net/browse/FR-19) | LLM Retrieval | Go query script, LLM hookup |

---

## Critical Path

```
FR-9 (combine + draft data)
  └── FR-13 (PlayerResolver)
        └── all cross-source data tasks
              └── FR-23 (Docker vector DB)
                    └── FR-18 (embed all data)   ← blocked by all 5 data epics
                          └── FR-20 (Go query script)
                                └── FR-22 (LLM hookup)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data collection | Python, nfl_data_py, requests, BeautifulSoup |
| Player identity | Python, rapidfuzz |
| Vector store | Qdrant (Docker) |
| Embedding | Python (model TBD at FR-18 kickoff) |
| Retrieval | Go |
| LLM | Anthropic Claude or OpenAI (TBD at FR-22 kickoff) |

---

## Local Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## PlayerResolver (`utils/player_resolver.py`)

Cross-source player identity matching utility. Maps any partial player record from any data source to a canonical nflverse `player_id`. Import and reuse in all collection scripts that need cross-source joins.

### Interface

```python
from utils.player_resolver import PlayerResolver

resolver = PlayerResolver()
player_id, confidence = resolver.resolve({
    "name":          "Dre Miller",
    "position":      "TE",
    "draft_year":    2022,
    "draft_round":   4,
    "drafting_team": "CIN",
})

# After a batch of records:
count = resolver.flush_unmatched()  # writes data/unmatched/resolver_unmatched_YYYY-MM-DD.csv
```

`resolve()` returns `(None, confidence)` for records below `min_confidence` (default 0.50) and queues them for `flush_unmatched()`. Pass `min_confidence=0.0` when all matches are wanted (e.g. ADP scraper).

### Matching tiers (highest to lowest)

| Tier | Inputs required | Confidence | Notes |
|------|----------------|------------|-------|
| 1 | `draft_year` + `draft_pick` (overall) | 1.00 | True primary key — every pick in every year is unique |
| 2 | `draft_year` + `draft_round` + `drafting_team` + `name` | 0.90 | Fuzzy last-name ≥ 85 score |
| 3 | `name` + `college` + `position` | 0.75 | Fuzzy full-name ≥ 80; position family match (WR/TE treated as same family) |
| 4 | `name` + `draft_year` | 0.50 | Exact last name + fuzzy first ≥ 70 |
| 5 | `name` + `position` | 0.35 | Fuzzy full-name ≥ 90; for sources with no draft info (ADP) |

### Reference table

`data/athletic/combine_draft.parquet` — 2,545 rows covering:
- All drafted QB/RB/WR/TE from 2000–2025 with a canonical gsis_id
- UDFA combine attendees (no draft year, but attended the combine and made the league)
- Combine measurables (forty, vertical, wt, etc.) where available — ~81% coverage
- Players who skipped the combine (e.g. Jaylen Waddle) are included with NaN measurables

Run `scripts/collect_combine.py` to regenerate. Combine data available from 2000 (nfl_data_py has no earlier data).

### Position families

Positions are grouped so that cross-source label differences don't cause mismatches:
- `{WR, TE, FL, SE}` — receiving positions
- `{RB, HB, FB}` — backfield
- `{QB}`
- `{CB, S, DB, FS, SS}` — secondary
- `{DE, DT, NT, DL, EDGE}` — D-line
- `{OT, OG, C, OL, T, G}` — O-line
- `{LB, OLB, ILB, MLB}` — linebackers
- `{K, P, LS}` — specialists

### Team normalization

`drafting_team` accepts any abbreviation or partial name — `"CIN"`, `"Cincinnati"`, `"Cincinnati Bengals"` all resolve to `"Cincinnati Bengals"` for comparison against the reference table.

---

## Scripts

### `scripts/collect_stats.py` — FR-6: Historical Stats Backfill

Pulls weekly and seasonal player stats (QB/RB/WR/TE) from 2012–2025 via nfl_data_py and writes Parquet files to `data/stats/`.

```bash
.venv/bin/python3 scripts/collect_stats.py
.venv/bin/python3 scripts/collect_stats.py --auto-confirm   # unattended
```

Output:
- `data/stats/weekly/YYYY.parquet` — game-level rows, one file per season
- `data/stats/seasonal/YYYY.parquet` — season-level rows, one file per season

Note: 2025 data pulls successfully if nflverse has published it; years without data are skipped automatically.

---

### `scripts/collect_combine.py` — FR-9: Combine & Draft Data

Builds the PlayerResolver reference table. Uses `import_players()` as the authoritative base (full draft coverage including combine-skippers), then left-joins combine measurables.

```bash
.venv/bin/python3 scripts/collect_combine.py
.venv/bin/python3 scripts/collect_combine.py --auto-confirm
```

Output: `data/athletic/combine_draft.parquet` — 2,545 rows, 19 columns.

Key design decisions:
- **Flipped join**: players table is the base, combine data is left-joined — ensures players who skipped the combine (e.g. Jaylen Waddle, pick 6 in 2021) are still in the reference table.
- **UDFA coverage**: skill position players with no draft year who attended the combine are included.
- **Extended range**: 2000–2025 (nfl_data_py combine data starts at 2000). Covers ADP-era veterans like Antonio Brown (drafted 2010).
- **Team normalization**: `draft_team` stored as full team name (e.g. `"Miami Dolphins"`) using `TEAM_MAP` from `player_resolver.py`.

---

### `scripts/collect_injuries.py` — FR-21: Player Injury History

Pulls weekly injury designations (practice status + game status) for QB/RB/WR/TE from 2012–2025 via nfl_data_py.

```bash
.venv/bin/python3 scripts/collect_injuries.py
.venv/bin/python3 scripts/collect_injuries.py --auto-confirm
```

Output: `data/injuries/YYYY.parquet` — one file per season, 23,712 total rows. `gsis_id` renamed to `player_id` for consistency. 0% null on player_id.

Key columns: `player_id`, `full_name`, `position`, `season`, `week`, `game_type`, `report_primary_injury`, `report_status`, `practice_primary_injury`, `practice_status`.

---

### `scripts/collect_adp.py` — FR-16: FantasyPros ADP

Scrapes historical ADP from FantasyPros for standard, PPR, and half-PPR formats, 2017–present. Resolves player IDs via PlayerResolver Tier 5 (name + position).

```bash
.venv/bin/python3 scripts/collect_adp.py
.venv/bin/python3 scripts/collect_adp.py --auto-confirm
```

Output: `data/adp/adp_historical.parquet` — 10,734 rows, 9 seasons × 3 formats.

Notes:
- 70.8% of rows resolve to a player_id via Tier 5 (confidence = 0.35). Unmatched rows are mostly pre-2000 draftees (veterans like Antonio Brown) not in the reference table.
- Half-PPR data not available for 2017 (FantasyPros didn't publish it that year).
- Scraper uses a 0.4s delay between requests to stay within polite crawl rate.
- The `confidence` column is retained in the output — downstream joins should filter or weight by it.

---

### `scripts/collect_adp.py` — FR-10: CFBD College Stats *(blocked — needs API key)*

**Status: blocked.** CFBD API requires a free key. Register at [collegefootballdata.com](https://collegefootballdata.com) and add to `.env`:

```
CFBD_API_KEY=your_key_here
```

The script will join to combine_draft via the `cfb_id` column (already populated from combine data). Planned output: `data/athletic/college_stats.parquet`.

---

### `scripts/viewer.py` — Local Data Viewer (localhost:8080)

Browse any collected dataset in your browser. Supports year filtering, player name search, and pagination.

```bash
.venv/bin/python3 scripts/viewer.py
open http://localhost:8080
pkill -f viewer.py   # stop
```

Datasets available once collected:
- Stats — Weekly / Seasonal
- Athletic — Combine & Draft
- Injuries
- ADP — Historical
- Coaching — Weekly / Coordinator Profiles *(not yet collected)*
- League Winners *(not yet collected)*

---

### `scripts/peek.py` — Parquet Inspection Utility

Inspect any Parquet file or directory.

```bash
.venv/bin/python3 scripts/peek.py data/stats/weekly/2024.parquet
.venv/bin/python3 scripts/peek.py data/stats/weekly/           # all years combined
.venv/bin/python3 scripts/peek.py data/stats/weekly/2024.parquet --search target_share
.venv/bin/python3 scripts/peek.py data/stats/weekly/2024.parquet --player "Tyreek Hill"
```

---

### `scripts/test_resolver.py` — PlayerResolver Manual Test Suite

Validates all 5 resolver tiers. Run after any change to `player_resolver.py` or after rebuilding `combine_draft.parquet`.

```bash
.venv/bin/python3 scripts/test_resolver.py
```

Expected output: 7/7 ✓. Test cases cover Tier 1 (exact pick), Tier 2 (team+round+name), Tier 3 (name+college, position mismatch), Tier 4 (name+year), and no-match.

---

## Data Storage

All Parquet files are gitignored — large and reproducible from collection scripts.

| Path | Contents | Status |
|------|----------|--------|
| `data/stats/weekly/YYYY.parquet` | Weekly player stats 2012–2025 | ✅ Collected |
| `data/stats/seasonal/YYYY.parquet` | Seasonal player stats 2012–2025 | ✅ Collected |
| `data/athletic/combine_draft.parquet` | Combine measurables + draft data 2000–2025 | ✅ Collected |
| `data/injuries/YYYY.parquet` | Weekly injury designations 2012–2025 | ✅ Collected |
| `data/adp/adp_historical.parquet` | FantasyPros ADP std/PPR/half-PPR 2017–2025 | ✅ Collected |
| `data/athletic/college_stats.parquet` | CFBD college stats | ⏳ Blocked (needs CFBD key) |
| `data/coaching/` | HC/OC/DC weekly + coordinator profiles | ⏳ Not started |
| `data/winners/` | League winner roster frequency | ⏳ Not started |
| `data/unmatched/` | PlayerResolver unmatched records for manual review | Auto-generated |

---

## Jira Ticket Reference

Project: `FR` at [aispm.atlassian.net](https://aispm.atlassian.net). Credentials in `/Users/ghost/Dev/jira/.suite`.

| Ticket | Summary | Status |
|--------|---------|--------|
| FR-6 | Historical stats backfill (nflverse) | ✅ Done |
| FR-9 | Combine & draft data | ✅ Done |
| FR-13 | PlayerResolver utility | 🔍 Review |
| FR-21 | Player injury history | 🔍 Review |
| FR-16 | FantasyPros ADP scraper | 🔍 Review |
| FR-10 | CFBD college stats | ⏳ Blocked (CFBD API key) |
| FR-14 | PFR coaching scraper (HC/OC/DC) | ⏳ Not started |
| FR-17 | League winner frequency (Sleeper + BBM) | ⏳ Not started |
| FR-25 | PFR 2025 season stats backfill | ⏳ Not started |
| FR-11 | Prospect confidence score formula | ⏳ Deferred |
| FR-23 | Docker vector DB (Qdrant) | ⏳ Not started |
| FR-18 | Embed all data | ⏳ Blocked by all data tasks |
| FR-20 | Go query script | ⏳ Not started |
| FR-22 | LLM hookup | ⏳ Not started |

---

## Transition IDs (Jira REST API)

```
2  → Review
11 → To Do
21 → In Progress
31 → Done
```

POST to `/rest/api/3/issue/{key}/transitions` with `{"transition": {"id": "2"}}`.
Search: POST to `/rest/api/3/search/jql` with `{"jql": "key = 'FR-13'", "fields": [...]}`.
