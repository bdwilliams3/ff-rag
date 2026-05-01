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
  ├── PFR scraper                 → coaching history
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

## Scripts

### `scripts/collect_stats.py` — FR-6: Historical Stats Backfill

Pulls weekly and seasonal player stats (QB/RB/WR/TE) from 2012–2025 via nfl_data_py and writes Parquet files to `data/stats/`.

```bash
# Interactive — pauses for schema approval before writing files
.venv/bin/python3 scripts/collect_stats.py

# Unattended — auto-confirms both checkpoints and writes immediately
.venv/bin/python3 scripts/collect_stats.py --auto-confirm
```

Output:
- `data/stats/weekly/YYYY.parquet` — one file per season, game-level rows
- `data/stats/seasonal/YYYY.parquet` — one file per season, season-level rows

---

### `scripts/peek.py` — Parquet Inspection Utility

Inspect any Parquet file or directory. Shows schema, dtypes, null rates, row counts by season/position, and a sample.

```bash
# Inspect a single year
.venv/bin/python3 scripts/peek.py data/stats/weekly/2024.parquet

# Inspect all years combined
.venv/bin/python3 scripts/peek.py data/stats/weekly/

# Check if a column exists and see sample values
.venv/bin/python3 scripts/peek.py data/stats/weekly/2024.parquet --search target_share

# Filter rows to a specific player
.venv/bin/python3 scripts/peek.py data/stats/weekly/2024.parquet --player "Tyreek Hill"
```

---

## Data Storage

All collected Parquet files are gitignored — they are large and reproducible from the collection scripts.

| Path | Contents |
|------|----------|
| `data/stats/` | Weekly and seasonal player stats (FR-1) |
| `data/injuries/` | Weekly injury designations (FR-1) |
| `data/athletic/` | Combine, draft, college stats (FR-2) |
| `data/coaching/` | Weekly coaching roster + scheme metrics (FR-3) |
| `data/adp/` | Historical ADP by format and year (FR-4) |
| `data/winners/` | League winner roster frequency (FR-5) |
| `data/unmatched/` | PlayerResolver unmatched records for manual review |
