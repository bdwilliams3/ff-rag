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
  ├── nflverse / OverTheCap       → player salaries, cap numbers, cash paid
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
| [FR-26](https://aispm.atlassian.net/browse/FR-26) | Player Salaries | Historical salary/cap/cash data by player-season |
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
| 5 | `name` + `position` | 0.35 | Fuzzy full-name ≥ 88 (after suffix strip); for sources with no draft info (ADP) |

### Reference table

`data/athletic/combine_draft.parquet` — 4,436 rows covering three buckets:
- All drafted QB/RB/WR/TE from 2000–2025 with a canonical gsis_id
- UDFA combine attendees (no draft year, but attended the combine and made the league)
- Modern-era UDFAs who never attended the combine but made the league (born after 1980) — covers players like Austin Ekeler, Adam Thielen
- Combine measurables (forty, vertical, wt, etc.) where available — ~47% coverage across full table (higher for drafted players)
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

Output: `data/athletic/combine_draft.parquet` — 4,436 rows, 19 columns.

Key design decisions:
- **Flipped join**: players table is the base, combine data is left-joined — ensures players who skipped the combine (e.g. Jaylen Waddle, pick 6 in 2021) are still in the reference table.
- **Three-bucket base**: (1) drafted skill players 2000–2025, (2) UDFA combine attendees, (3) modern-era UDFAs who never attended the combine but made the league (born after 1980). This third bucket captures high-value UDFAs like Austin Ekeler and Adam Thielen who had no combine entry.
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

Scrapes historical ADP from FantasyPros for standard, PPR, and half-PPR formats, 2012–present. Resolves player IDs via PlayerResolver Tier 5 (name + position).

```bash
.venv/bin/python3 scripts/collect_adp.py
.venv/bin/python3 scripts/collect_adp.py --auto-confirm
```

Output: `data/adp/adp_historical.parquet` — 14,015 rows across 2012–2025.

Notes:
- 96.3% of rows resolve to a player_id via Tier 5 (confidence = 0.35). Remaining rows are nickname aliases (e.g. "Hollywood Brown" for Marquise Brown), pre-2000 era players not in the reference table, and some source naming differences.
- The canonical `team` column is inferred from local nflverse weekly stats by `player_id + season`, with 2025 partial fallback from injury reports. FantasyPros historical team labels are intentionally ignored because they are sparse and can show current/latest team rather than the historical ADP-season team.
- Standard and PPR are available back to 2012. Half-PPR is available from 2018 onward; FantasyPros returns no report for 2012–2017.
- Scraper uses a 0.4s delay between requests to stay within polite crawl rate.
- The `confidence` column is retained in the output — downstream joins should filter or weight by it.

---

### `scripts/collect_college_stats.py` — FR-10: CFBD College Stats

Pulls season-level passing, rushing, and receiving stats from the College Football Data API for all players in `combine_draft.parquet`. Resolves player IDs via PlayerResolver — Tier 3 (name + college, 0.75) when CFBD's team string matches nflverse's college string; Tier 4 (name + draft_year, 0.50) as fallback for transfer players whose nflverse college field carries multiple schools (e.g. `"USC; Pittsburgh"`, `"Wyoming; Reedley"`).

```bash
.venv/bin/python3 scripts/collect_college_stats.py
.venv/bin/python3 scripts/collect_college_stats.py --auto-confirm   # unattended
.venv/bin/python3 scripts/collect_college_stats.py --resume         # continue a previous run
```

Requires `CFBD_API_KEY` in `.env` (free key at [collegefootballdata.com](https://collegefootballdata.com)):

```
CFBD_API_KEY=your_key_here
```

Output: `data/athletic/college_stats.parquet` — one row per player per college season.

Key columns: `player_id`, `cfbd_player_id`, `player_name`, `college`, `season`, `pass_att`, `pass_comp`, `pass_yds`, `pass_td`, `pass_int`, `rush_att`, `rush_yds`, `rush_td`, `rec`, `rec_yds`, `rec_td`, `resolver_confidence`.

Notes:
- Pulls by year (one request per year per category ≈ 87 total) rather than per player — keeps API usage minimal on a free-tier key.
- Resolution runs once per unique CFBD player after all years are fetched, using `last_college_season + 1` as the draft-year estimate. This lets Tier 4 catch transfer players whose nflverse college string is multi-school.
- Progress is checkpointed after each year to `data/athletic/college_stats_checkpoint.parquet` — use `--resume` to continue across sessions.
- Targets are not tracked at the NCAA level and are not returned by the CFBD API.

---

### `scripts/collect_coaching.py` — FR-14: Coaching Staff History

Scrapes Pro Football History franchise/season pages for HC, OC, and DC roles from 2012–2025, expands staff assignments to one row per team per regular-season week, and preserves interim replacements from the first post-firing game week onward.

```bash
.venv/bin/python3 scripts/collect_coaching.py
.venv/bin/python3 scripts/collect_coaching.py --auto-confirm
```

Output: `data/coaching/coaching_weekly.parquet` — 7,776 rows.

Key columns: `season`, `week`, `team`, `team_name`, `gameday`, `had_game`, `hc_name`, `oc_name`, `dc_name`, `hc_interim`, `oc_interim`, `dc_interim`.

Notes:
- PFR was the original Jira source, but direct automated access is blocked by Cloudflare from this environment. Pro Football History provides the needed staff roles, interim labels, and firing-date context in scrapeable HTML.
- Bye weeks are retained with `had_game = False` so the table has complete team-week coverage.
- If a team has no official listed OC or DC, the value is stored as `No official OC listed` / `No official DC listed` rather than a null.

---

### `scripts/build_coordinator_profiles.py` — FR-15: Coaching Tendency Profiles

Joins `coaching_weekly.parquet` to weekly player stats and builds separate offensive and defensive tendency tables. Offensive rows aggregate only the team's own production under the active OC. Defensive rows aggregate opponent production allowed under the active DC. Both tables include `hc_name` so head coach context is available on each line item.

```bash
.venv/bin/python3 scripts/build_coordinator_profiles.py
.venv/bin/python3 scripts/build_coordinator_profiles.py --auto-confirm
```

Outputs:
- `data/coaching/offensive_coordinator_profiles.parquet` — 422 OC/team-season rows
- `data/coaching/defensive_coordinator_profiles.parquet` — 422 DC/team-season rows

Offensive metrics include pass rate, run rate, games coached, pass/rush yards, WR/TE/RB target share, WR1 target share, WR2 target share, TE receiving yards, and RB carries.

Defensive metrics include pass attempts/yards against, rush attempts/yards against, targets/receptions/receiving yards allowed, WR/TE/RB target share allowed, TE receiving yards allowed, and RB carries allowed.

---

### `scripts/collect_salaries.py` — FR-27: Historical Player Salaries

Pulls nflverse historical contracts sourced from OverTheCap, flattens the nested annual cap/cash rows, aggregates to one row per player-season, and writes salary-cap context for era normalization.

```bash
.venv/bin/python3 scripts/collect_salaries.py
.venv/bin/python3 scripts/collect_salaries.py --auto-confirm
```

Outputs:
- `data/salaries/player_salaries.parquet` — 33,460 player-season rows for 2012–2025
- `data/salaries/salary_cap_by_season.parquet` — official base cap plus aggregate player salary/cap totals by season

Key columns: `player_id`, `player_name`, `position`, `season`, `team`, `base_salary_millions`, `cap_number_millions`, `cash_paid_millions`, `cap_percent_of_league_cap`, `official_base_salary_cap_millions`, and source metadata.

Notes:
- Monetary values are stored in millions of USD, matching the nflverse/OverTheCap source.
- The output keeps broad NFL salary coverage across all positions, not only QB/RB/WR/TE.
- `player_id` comes from the source `gsis_id`; about 2% of salary rows lack a player ID in the source.
- The current source covers 2,677 of 4,510 local FF RAG player IDs. Missing rows are source gaps, mostly older or lower-salary historical players, not inferred salaries.

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
- Coaching — Weekly / Offensive Profiles / Defensive Profiles
- League Winners
- Salaries — Player Salaries / Salary Cap

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
| `data/adp/adp_historical.parquet` | FantasyPros ADP std/PPR/half-PPR 2012–2025 | ✅ Collected |
| `data/athletic/college_stats.parquet` | CFBD college stats by player × season | ⏳ Pending first run |
| `data/coaching/coaching_weekly.parquet` | HC/OC/DC per team-week, including interim changes | ✅ Collected |
| `data/coaching/offensive_coordinator_profiles.parquet` | OC offensive tendency profiles by team-season, with HC context | ✅ Collected |
| `data/coaching/defensive_coordinator_profiles.parquet` | DC allowed-production profiles by team-season, with HC context | ✅ Collected |
| `data/winners/league_winner_frequency.parquet` | MFL redraft champion roster frequency 2012–2025 | ✅ Script fixed; run full collection |
| `data/salaries/player_salaries.parquet` | nflverse/OverTheCap player salary/cap/cash by player-season 2012–2025 | ✅ Collected |
| `data/salaries/salary_cap_by_season.parquet` | official base salary cap and aggregate salary totals by season | ✅ Collected |
| `data/unmatched/` | PlayerResolver unmatched records for manual review | Auto-generated |

---

## Jira Ticket Reference

Project: `FR` at [aispm.atlassian.net](https://aispm.atlassian.net). Credentials in `/Users/ghost/Dev/jira/.suite`.

| Ticket | Summary | Status |
|--------|---------|--------|
| FR-6  | Historical stats backfill (nflverse)        | ✅ Done |
| FR-9  | Combine & draft data                        | ✅ Done |
| FR-10 | CFBD college stats                          | ✅ Done |
| FR-13 | PlayerResolver utility                      | ✅ Done |
| FR-14 | Coaching scraper (HC/OC/DC)                 | ✅ Done |
| FR-15 | Coaching impact formula                     | ✅ Done |
| FR-16 | FantasyPros ADP scraper                     | ✅ Done |
| FR-21 | Player injury history                       | ✅ Done |
| FR-27 | Historical player salaries                  | ✅ Done |
| FR-17 | League winner frequency (MFL redraft)        | 🔄 In Progress — script fixed, run full collection |
| FR-7  | PFR enrichment scraper                      | ⏳ Not started |
| FR-8  | In-season weekly ingest (SportRadar/Sleeper)| ⏳ Not started |
| FR-25 | PFR 2025 season stats backfill              | ⏳ Not started |
| FR-11 | Prospect confidence score formula           | ⏳ Deferred |
| FR-23 | Docker vector DB (Qdrant)                   | ⏳ Not started |
| FR-18 | Embed all data                              | ⏳ Blocked by FR-17 + FR-23 |
| FR-20 | Go query script                             | ⏳ Not started |
| FR-22 | LLM hookup                                  | ⏳ Not started |

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
