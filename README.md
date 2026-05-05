# FF-RAG Getting Started Guide

A local fantasy football AI you can ask anything: who to draft, who to start, which waivers to target, rookie breakouts, scheme fits. It runs entirely on your machine — your data, your questions, no subscription.

---

## What This Does

It builds a searchable knowledge base from 13 years of NFL data (stats, ADP, coaching tendencies, injury history, college production, salaries, combine measurables), stores it in a local vector database, and lets you ask natural-language questions answered by Gemini with that data as context.

**Example questions it can answer:**
- *"Which WRs have the best early-season schedule and are still available in the 5th round?"*
- *"Which 2025 rookies have elite college production and land in a pass-heavy offense?"*
- *"Compare Davante Adams and Stefon Diggs — who should I start in Week 6?"*
- *"Which defenses give up the most points to RBs?"*

---

## Prerequisites

You need three things installed before starting:

### 1. Python 3.9 or newer
Check if you have it:
```bash
python3 --version
```
If not, download from [python.org](https://www.python.org/downloads/).

### 2. Docker Desktop
Qdrant (the vector database) runs in Docker. Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop). After installing, open Docker Desktop and leave it running.

Check it works:
```bash
docker --version
```

### 3. Git
Usually pre-installed on Mac/Linux. On Windows, download from [git-scm.com](https://git-scm.com).

---

## Step 1 — Get the Code

```bash
git clone <repository-url>
cd ff-rag
```

---

## Step 2 — Get Your API Keys

You need two free API keys. Get both before proceeding.

### CFBD API Key (College Football Data)

Used to pull college stats for every NFL player — crucial for evaluating rookies and prospects.

1. Go to [collegefootballdata.com/key](https://collegefootballdata.com/key)
2. Sign up for a free account
3. Your API key will appear on the dashboard — copy it

**Free tier is sufficient.** No credit card required.

### Gemini API Key (Google)

The AI model that reads the data and answers your questions.

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **"Get API key"** → **"Create API key"**
4. Copy the key (starts with `AIza...`)

**Free tier limits:** 15 requests/minute, 1,500/day — more than enough for personal use.

---

## Step 3 — Configure Environment

Create a `.env` file in the project root with your two keys:

```bash
# Run this, replacing the placeholder values with your actual keys
cat > .env << 'EOF'
CFBD_API_KEY=your_cfbd_key_here
GEMINI_API_KEY=your_gemini_key_here
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ff-rag-v1
EOF
```

Or create the file manually — it should look exactly like this:
```
CFBD_API_KEY=DoKxb...
GEMINI_API_KEY=AIzaSy...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ff-rag-v1
```

---

## Step 4 — Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** You'll see the model files download the first time you run embed.py (~130MB for BAAI/bge-small-en). This is a one-time download.

---

## Step 5 — Start the Vector Database

```bash
docker-compose up -d
```

This starts Qdrant in the background on port 6333. Your data persists in a Docker volume between restarts.

Verify it's running:
```bash
curl http://localhost:6333/healthz
```
You should see `{"title":"qdrant - vector search engine"}` or similar.

---

## Step 6 — Collect the Data

Run these scripts in order. Each one downloads NFL data and saves it locally as Parquet files. **The data directory is gitignored** — nothing large is committed to the repo.

> **Total time estimate:** 30–60 minutes, mostly due to the college stats script making many API calls. The other scripts finish in under 5 minutes each.

Load your API keys into the shell first:
```bash
export $(grep -v '^#' .env | xargs)
```

---

### 6a — Combine & Draft Data
*Builds the player identity reference table. Must run first — every other script depends on it.*
```bash
python scripts/collect_combine.py --auto-confirm
```
Output: `data/athletic/combine_draft.parquet` (~4,400 players, 2000–2025)

---

### 6b — Historical Stats (2012–2025)
*Weekly and seasonal stat lines for all QB/RB/WR/TE.*
```bash
python scripts/collect_stats.py --auto-confirm
```
Output: `data/stats/weekly/YYYY.parquet` and `data/stats/seasonal/YYYY.parquet`

---

### 6c — Injury History
*Weekly injury designations and practice reports, 2012–2025.*
```bash
python scripts/collect_injuries.py --auto-confirm
```
Output: `data/injuries/YYYY.parquet`

---

### 6d — College Stats
*Pulls production data for every NFL player from collegefootballdata.com. Slowest script — ~30 minutes. Supports `--resume` if interrupted.*
```bash
python scripts/collect_college_stats.py --auto-confirm
```
If it gets interrupted, resume where it left off:
```bash
python scripts/collect_college_stats.py --resume --auto-confirm
```
Output: `data/athletic/college_stats.parquet`

---

### 6e — ADP History (Draft Position)
*Historical average draft position by format (Standard, PPR, Half-PPR), 2012–2025.*
```bash
python scripts/collect_adp.py --auto-confirm
```
Output: `data/adp/adp_historical.parquet` (~14,000 rows)

---

### 6f — Coaching Staff History
*Head coach, offensive coordinator, and defensive coordinator by team and week, 2012–2025.*
```bash
python scripts/collect_coaching.py --auto-confirm
```
Output: `data/coaching/coaching_weekly.parquet` (~7,800 rows)

---

### 6g — Coordinator Tendency Profiles
*Aggregates coaching data into per-coordinator offensive and defensive tendency profiles (pass rate, target share splits, yards allowed, etc.).*
```bash
python scripts/build_coordinator_profiles.py --auto-confirm
```
Output:
- `data/coaching/offensive_coordinator_profiles.parquet`
- `data/coaching/defensive_coordinator_profiles.parquet`

---

### 6h — Player Salaries & Cap Data
*Contract values, guaranteed money, and cap hits, 2012–2025.*
```bash
python scripts/collect_salaries.py --auto-confirm
```
Output:
- `data/salaries/player_salaries.parquet` (~33,000 rows)
- `data/salaries/salary_cap_by_season.parquet`

---

### 6i — 2025 Season Data
*Specialized collection and derived metrics for the current season.*
```bash
python scripts/collect_stats_2025.py --auto-confirm
python scripts/compute_derived_stats_2025.py --auto-confirm
```

---

## Step 7 — Embed the Data into Qdrant

This reads every Parquet file, converts each row into a text chunk, embeds it with a local AI model (BAAI/bge-small-en), and stores everything in Qdrant. **Run once after all data is collected.** Re-run after adding new data.

```bash
python scripts/embed.py --provider fastembed-bge
```

This will process ~160,000 rows across all data sources. Expect 10–20 minutes. Progress prints to the console.

When it finishes, verify the collection exists:
```bash
curl http://localhost:6333/collections/ff-rag-v1
```
Look for `"vectors_count"` — it should be around 160,000+.

---

## Step 8 — Ask Questions

Load your API keys if you haven't already:
```bash
export $(grep -v '^#' .env | xargs)
```

Now ask anything:

```bash
python scripts/ask.py "Who are the best WRs to target in rounds 3–5 of my PPR draft?"
```

```bash
python scripts/ask.py "Which 2025 rookies should I target and why?"
```

```bash
python scripts/ask.py "Which RBs are in run-heavy offenses this year?"
```

```bash
python scripts/ask.py "Who are the best handcuffs worth rostering?"
```

---

## Useful Flags

Narrow the search to get more relevant answers:

```bash
# Filter by position
python scripts/ask.py --position WR "which veterans are undervalued this year?"

# Filter by recent seasons only
python scripts/ask.py --season-min 2022 "who has the most consistent target share?"

# Search a specific data source
python scripts/ask.py --source offensive_coordinator_profiles "which OCs run the most pass-heavy schemes?"
python scripts/ask.py --source defensive_coordinator_profiles "which defenses are easiest to stream against?"

# See what data fed the answer
python scripts/ask.py --show-context "best TE streaming options"

# Combine filters
python scripts/ask.py --position RB --season-min 2023 "highest carry volume backs"
```

Available `--source` values:
| Value | What it contains |
|---|---|
| `weekly_stats` | Game-by-game stat lines |
| `seasonal_stats` | Full-season totals |
| `combine_draft` | Combine measurables and draft position |
| `college_stats` | College production by season |
| `injuries` | Injury history and designations |
| `adp` | Historical average draft position |
| `coaching_weekly` | Weekly HC/OC/DC assignments |
| `offensive_coordinator_profiles` | OC pass rate, target share splits |
| `defensive_coordinator_profiles` | DC yards/TDs allowed by position |
| `player_salaries` | Contract values and cap hits |
| `league_winner_frequency` | Championship roster frequency |

---

## Troubleshooting

**`No module named 'nfl_data_py'` or similar**
Make sure your virtual environment is activated: `source .venv/bin/activate`

**`Connection refused` on port 6333**
Qdrant isn't running. Start it: `docker-compose up -d`

**`Set GEMINI_API_KEY environment variable`**
Run `export $(grep -v '^#' .env | xargs)` to load your keys into the shell.

**`CFBD_API_KEY` not found during college stats**
Same as above — load your `.env` before running collection scripts.

**College stats script hangs or errors mid-way**
Use `--resume` flag — it checkpoints after each year so you don't start over.

**Embedding is slow**
Normal — it's running a local AI model on ~160,000 rows. Let it run. The model download only happens once.

**`vectors_count` is 0 after embedding**
Check that Qdrant was running during `embed.py`. Re-run embed after starting Docker.

---

## Updating Data Mid-Season

When you want to pull fresh stats during the season, re-run only the scripts for the data that changed, then re-embed:

```bash
# Refresh current season stats
python scripts/collect_stats_2025.py --auto-confirm
python scripts/compute_derived_stats_2025.py --auto-confirm

# Re-embed (upserts are idempotent — safe to re-run)
python scripts/embed.py --provider fastembed-bge
```

---

## Data Sources

| Data | Source | API Key? |
|---|---|---|
| Stats (weekly/seasonal) | nflverse via nfl_data_py | No |
| Combine & draft history | nflverse via nfl_data_py | No |
| Injury reports | nflverse via nfl_data_py | No |
| College stats | collegefootballdata.com | **Yes (free)** |
| ADP history | FantasyPros (scraped) | No |
| Coaching staff history | Pro Football History (scraped) | No |
| Player salaries | nflverse/OverTheCap | No |
| LLM answers | Google Gemini | **Yes (free tier)** |
| Vector storage | Qdrant (local Docker) | No |
