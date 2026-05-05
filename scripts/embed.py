"""
FR-18: embed collected Parquet data into the local Qdrant RAG corpus.

Uses Ollama's nomic-embed-text model for 768-dimensional embeddings and Qdrant's
REST API for deterministic upserts into the ff-rag-v1 collection.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION = "ff-rag-v1"
MODEL = "nomic-embed-text"
FASTEMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
FASTEMBED_BGE_MODEL = "BAAI/bge-small-en"

VECTOR_SIZES = {
    "ollama": 768,
    "fastembed": 768,
    "fastembed-bge": 384,
}

OLLAMA_BATCH_SIZE = 16
FASTEMBED_BATCH_SIZE = 512
UPSERT_SIZE = 128


IMPORTANT_COLUMNS = {
    "weekly_stats": [
        "player_display_name",
        "player_name",
        "position",
        "recent_team",
        "opponent_team",
        "season",
        "week",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "target_share",
        "air_yards_share",
        "fantasy_points",
        "fantasy_points_ppr",
    ],
    "seasonal_stats": [
        "player_display_name",
        "player_name",
        "position",
        "season",
        "games",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "fantasy_points",
        "fantasy_points_ppr",
    ],
    "combine_draft": [
        "player_name",
        "position",
        "college",
        "draft_year",
        "draft_round",
        "draft_pick",
        "draft_team",
        "ht",
        "wt",
        "forty",
        "bench",
        "vertical",
        "broad_jump",
        "cone",
        "shuttle",
    ],
    "college_stats": [
        "player_name",
        "college",
        "cfbd_position",
        "season",
        "pass_att",
        "pass_comp",
        "pass_yds",
        "pass_td",
        "pass_int",
        "rush_att",
        "rush_yds",
        "rush_td",
        "rec",
        "rec_yds",
        "rec_td",
        "resolver_confidence",
        "match_source",
    ],
    "injuries": [
        "full_name",
        "position",
        "team",
        "season",
        "games_listed",
        "weeks",
        "report_statuses",
        "report_injuries",
        "practice_statuses",
        "practice_injuries",
    ],
    "adp": [
        "player_name",
        "position",
        "team",
        "season",
        "format",
        "avg_adp",
        "confidence",
        "team_source",
    ],
    "coaching_weekly": [
        "season",
        "week",
        "team",
        "team_name",
        "gameday",
        "had_game",
        "hc_name",
        "oc_name",
        "dc_name",
        "hc_interim",
        "oc_interim",
        "dc_interim",
        "record",
    ],
    "offensive_coordinator_profiles": [
        "season",
        "team",
        "team_name",
        "hc_name",
        "oc_name",
        "games_coached",
        "pass_rate",
        "run_rate",
        "wr_target_share",
        "wr1_target_share",
        "wr2_target_share",
        "te_target_share",
        "rb_target_share",
        "pass_yards",
        "rush_yards",
        "te_receiving_yards",
        "rb_carries",
    ],
    "defensive_coordinator_profiles": [
        "season",
        "team",
        "team_name",
        "hc_name",
        "dc_name",
        "games_coached",
        "pass_rate_allowed",
        "run_rate_allowed",
        "wr_target_share_allowed",
        "te_target_share_allowed",
        "rb_target_share_allowed",
        "pass_yards_against",
        "rush_yards_against",
        "te_receiving_yards_allowed",
        "rb_carries_allowed",
    ],
    "player_salaries": [
        "player_name",
        "position",
        "team",
        "season",
        "base_salary_millions",
        "guaranteed_salary_millions",
        "cap_number_millions",
        "cash_paid_millions",
        "cap_percent_of_league_cap",
        "official_base_salary_cap_millions",
        "source",
        "source_grain",
    ],
    "salary_cap": [
        "season",
        "official_base_salary_cap_millions",
        "player_rows",
        "total_player_cap_number_millions",
        "total_player_cash_paid_millions",
        "max_player_cap_number_millions",
        "max_player_cash_paid_millions",
        "source",
    ],
    "league_winner_frequency": [
        "player_name",
        "position",
        "season",
        "league_type",
        "scoring_format",
        "te_premium",
        "championship_roster_appearances",
        "total_leagues_sampled",
        "roster_appearance_rate",
        "sample_source",
    ],
    "fantasypros_championship_rosters": [
        "player_name",
        "source_position",
        "resolved_position",
        "source_team",
        "season",
        "appearance_rate",
        "source_label",
        "contest_type",
        "platform",
        "scoring_format",
        "sample_type",
        "confidence_weight",
        "resolver_confidence",
    ],
    "nfl_schedule_2026": [
        "season",
        "week",
        "gameday",
        "team",
        "opponent_team",
        "home_away",
        "div_game",
        "roof",
        "surface",
        "stadium",
        "location",
        "spread_line",
        "away_qb_name",
        "home_qb_name",
        "away_coach",
        "home_coach",
    ],
}


DATASETS = [
    ("weekly_stats", DATA_DIR / "stats" / "weekly", "player-week"),
    ("seasonal_stats", DATA_DIR / "stats" / "seasonal", "player-season"),
    ("combine_draft", DATA_DIR / "athletic" / "combine_draft.parquet", "player"),
    ("college_stats", DATA_DIR / "athletic" / "college_stats.parquet", "player-college-season"),
    ("injuries", DATA_DIR / "injuries", "player-season"),
    ("adp", DATA_DIR / "adp" / "adp_historical.parquet", "player-season-format"),
    ("coaching_weekly", DATA_DIR / "coaching" / "coaching_weekly.parquet", "team-week"),
    (
        "offensive_coordinator_profiles",
        DATA_DIR / "coaching" / "offensive_coordinator_profiles.parquet",
        "coordinator-team-season",
    ),
    (
        "defensive_coordinator_profiles",
        DATA_DIR / "coaching" / "defensive_coordinator_profiles.parquet",
        "coordinator-team-season",
    ),
    ("player_salaries", DATA_DIR / "salaries" / "player_salaries.parquet", "player-season"),
    ("salary_cap", DATA_DIR / "salaries" / "salary_cap_by_season.parquet", "season"),
    (
        "league_winner_frequency",
        DATA_DIR / "winners" / "league_winner_frequency.parquet",
        "player-season-league",
    ),
    (
        "fantasypros_championship_rosters",
        DATA_DIR / "winners" / "fantasypros_championship_rosters.parquet",
        "player-season-source",
    ),
    (
        "nfl_schedule_2026",
        DATA_DIR / "schedule" / "nfl_schedule_2026.parquet",
        "team-game",
    ),
]


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def display_value(value: Any) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def row_value(row: pd.Series, *columns: str) -> Any:
    for column in columns:
        if column in row:
            value = clean_value(row[column])
            if value is not None and display_value(value) != "":
                return value
    return None


def scalar_payload(value: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def row_payload(row: pd.Series, source: str, source_path: Path, grain: str, row_key: str) -> dict[str, Any]:
    payload = {
        "data_source": source,
        "source_path": str(source_path.relative_to(ROOT)),
        "grain": grain,
        "row_key": row_key,
    }

    field_map = {
        "player_id": ("player_id",),
        "player_name": ("player_display_name", "player_name", "full_name"),
        "position": ("position", "resolved_position", "source_position", "cfbd_position"),
        "season": ("season", "draft_year"),
        "week": ("week",),
        "team": ("recent_team", "team", "draft_team", "source_team"),
        "opponent_team": ("opponent_team",),
        "format": ("format", "scoring_format"),
        "league_type": ("league_type", "contest_type"),
        "coordinator": ("oc_name", "dc_name"),
        "head_coach": ("hc_name",),
    }
    for key, columns in field_map.items():
        value = row_value(row, *columns)
        if value is not None:
            payload[key] = scalar_payload(value)
    return payload


def row_key(row: pd.Series, source: str, index: int) -> str:
    columns_by_source = {
        "weekly_stats": ["player_id", "season", "week", "recent_team"],
        "seasonal_stats": ["player_id", "season"],
        "combine_draft": ["player_id"],
        "college_stats": ["player_id", "season", "college", "cfbd_player_id"],
        "injuries": ["player_id", "season", "team"],
        "adp": ["player_id", "season", "format"],
        "coaching_weekly": ["season", "week", "team"],
        "offensive_coordinator_profiles": ["season", "team", "oc_name"],
        "defensive_coordinator_profiles": ["season", "team", "dc_name"],
        "player_salaries": ["player_id", "season", "team", "otc_id"],
        "salary_cap": ["season"],
        "league_winner_frequency": ["player_id", "season", "league_type", "scoring_format"],
        "fantasypros_championship_rosters": ["player_name", "season", "source_label"],
        "nfl_schedule_2026": ["game_id", "team"],
    }
    parts = []
    for column in columns_by_source[source]:
        if column in row:
            parts.append(display_value(row[column]))
    parts = [part for part in parts if part]
    if not parts:
        parts = [str(index)]
    parts.append(f"row={index}")
    return "|".join(parts)


def point_id(source: str, key: str) -> str:
    digest = hashlib.sha1(f"{source}|{key}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def format_row(source: str, row: pd.Series) -> str:
    labels = {
        "weekly_stats": "Weekly NFL stat line",
        "seasonal_stats": "Seasonal NFL stat line",
        "combine_draft": "Combine and draft profile",
        "college_stats": "College production line",
        "injuries": "Injury history summary",
        "adp": "FantasyPros ADP",
        "coaching_weekly": "Weekly coaching staff",
        "offensive_coordinator_profiles": "Offensive coordinator tendency profile",
        "defensive_coordinator_profiles": "Defensive coordinator allowed-production profile",
        "player_salaries": "Player salary and cap context",
        "salary_cap": "NFL salary cap season context",
        "league_winner_frequency": "Fantasy league winner frequency",
        "fantasypros_championship_rosters": "FantasyPros championship roster note",
        "nfl_schedule_2026": "2026 NFL regular season schedule",
    }
    parts = []
    for column in IMPORTANT_COLUMNS[source]:
        if column in row:
            value = display_value(row[column])
            if value:
                parts.append(f"{column}: {value}")
    if source == "seasonal_stats":
        parts.extend(seasonal_share_parts(row))
    return f"{labels[source]}. " + "; ".join(parts)


def seasonal_share_parts(row: pd.Series) -> list[str]:
    """nflverse seasonal share fields are sums of weekly shares, not season rates."""
    games = clean_value(row.get("games"))
    if not games:
        return []

    parts = []
    for source_column, output_column in [
        ("target_share", "avg_weekly_target_share"),
        ("air_yards_share", "avg_weekly_air_yards_share"),
    ]:
        value = clean_value(row.get(source_column))
        if value is None:
            continue
        avg_value = float(value) / float(games)
        parts.append(f"{output_column}: {display_value(avg_value)}")
    return parts


def read_parquet(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files under {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    return pd.read_parquet(path)


def load_dataset(source: str, path: Path) -> pd.DataFrame:
    df = read_parquet(path)
    if source == "injuries":
        return aggregate_injuries(df)
    return df.reset_index(drop=True)


def aggregate_injuries(df: pd.DataFrame) -> pd.DataFrame:
    def unique_join(values: pd.Series) -> str:
        clean = sorted({display_value(value) for value in values if display_value(value)})
        return ", ".join(clean[:12])

    grouped = (
        df.groupby(["player_id", "season", "team"], dropna=False)
        .agg(
            full_name=("full_name", "first"),
            position=("position", "first"),
            games_listed=("week", "count"),
            weeks=("week", lambda values: ", ".join(str(int(v)) for v in sorted(set(values.dropna())))),
            report_statuses=("report_status", unique_join),
            report_injuries=("report_primary_injury", unique_join),
            practice_statuses=("practice_status", unique_join),
            practice_injuries=("practice_primary_injury", unique_join),
        )
        .reset_index()
    )
    return grouped


def chunks_for_dataset(source: str, path: Path, grain: str, limit: int | None) -> Iterator[dict[str, Any]]:
    df = load_dataset(source, path)
    if limit:
        df = df.head(limit)
    for index, row in df.iterrows():
        key = row_key(row, source, index)
        text = format_row(source, row)
        payload = row_payload(row, source, path, grain, key)
        payload["text"] = text
        yield {
            "id": point_id(source, key),
            "text": text,
            "payload": payload,
        }


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, timeout=120, **kwargs)
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def ensure_services(provider: str) -> None:
    request_json("GET", f"{QDRANT_URL}/collections")
    if provider == "fastembed":
        return
    tags = request_json("GET", f"{OLLAMA_URL}/api/tags")
    models = {model["name"] for model in tags.get("models", [])}
    if MODEL not in models and f"{MODEL}:latest" not in models:
        raise RuntimeError(f"Ollama model {MODEL!r} is not installed")


def ensure_collection(provider: str, recreate: bool = False) -> None:
    if recreate:
        requests.delete(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=120)

    response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=120)
    if response.status_code == 200:
        return
    if response.status_code != 404:
        response.raise_for_status()

    body = {
        "vectors": {
            "size": VECTOR_SIZES[provider],
            "distance": "Cosine",
        }
    }
    request_json("PUT", f"{QDRANT_URL}/collections/{COLLECTION}", json=body)


class Embedder:
    batch_size = OLLAMA_BATCH_SIZE

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts_ollama(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_texts_ollama([text])[0]


class FastEmbedder(Embedder):
    batch_size = FASTEMBED_BATCH_SIZE

    def __init__(self, model_name: str = FASTEMBED_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("fastembed is not installed; run pip install -r requirements.txt") from exc
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {text}" for text in texts]
        return [vector.tolist() for vector in self.model.embed(prefixed, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(self.model.query_embed(f"search_query: {text}"))
        return vector.tolist()


def embed_texts_ollama(texts: list[str]) -> list[list[float]]:
    body = {"model": MODEL, "input": texts, "keep_alive": "30m"}
    response = requests.post(f"{OLLAMA_URL}/api/embed", json=body, timeout=300)
    if response.status_code == 404:
        vectors = []
        for text in texts:
            legacy = request_json(
                "POST",
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": MODEL, "prompt": text, "keep_alive": "30m"},
            )
            vectors.append(legacy["embedding"])
        return vectors

    response.raise_for_status()
    data = response.json()
    vectors = data.get("embeddings")
    if vectors is None:
        raise RuntimeError(f"Unexpected Ollama embed response: {data.keys()}")
    return vectors


def upsert(points: list[dict[str, Any]]) -> None:
    body = {"points": points}
    request_json(
        "PUT",
        f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
        json=body,
    )


def collection_count() -> int:
    data = request_json("POST", f"{QDRANT_URL}/collections/{COLLECTION}/points/count", json={"exact": True})
    return int(data["result"]["count"])


def embed_dataset(source: str, path: Path, grain: str, limit: int | None, embedder: Embedder) -> int:
    started = time.monotonic()
    count = 0
    pending = []

    for chunk_batch in batched(chunks_for_dataset(source, path, grain, limit), embedder.batch_size):
        vectors = embedder.embed_documents([chunk["text"] for chunk in chunk_batch])
        if len(vectors) != len(chunk_batch):
            raise RuntimeError(f"Expected {len(chunk_batch)} embeddings, got {len(vectors)}")
        for chunk, vector in zip(chunk_batch, vectors):
            pending.append(
                {
                    "id": chunk["id"],
                    "vector": vector,
                    "payload": chunk["payload"],
                }
            )
            count += 1
        if len(pending) >= UPSERT_SIZE:
            upsert(pending)
            pending = []
            print(f"  {source}: {count:,} chunks upserted", flush=True)

    if pending:
        upsert(pending)

    elapsed = time.monotonic() - started
    print(f"{source}: embedded {count:,} chunks in {elapsed:.1f}s", flush=True)
    return count


def search(
    query: str,
    embedder: Embedder,
    limit: int = 5,
    qdrant_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    vector = embedder.embed_query(query)
    body = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
    }
    if qdrant_filter:
        body["filter"] = qdrant_filter
    return request_json("POST", f"{QDRANT_URL}/collections/{COLLECTION}/points/search", json=body)


def smoke_test(embedder: Embedder) -> None:
    query = "Justin Jefferson 2023 fantasy points receptions receiving yards target share"
    data = search(
        query,
        embedder,
        limit=5,
        qdrant_filter={"must": [{"key": "season", "match": {"value": 2023}}]},
    )
    print("\nSmoke test: Justin Jefferson 2023")
    for i, point in enumerate(data.get("result", []), start=1):
        payload = point.get("payload", {})
        print(
            f"{i}. score={point.get('score'):.4f} "
            f"source={payload.get('data_source')} "
            f"player={payload.get('player_name')} "
            f"season={payload.get('season')} "
            f"text={payload.get('text')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Embed only the first N rows from each dataset")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection first")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the Justin Jefferson smoke search")
    parser.add_argument("--source", help="Comma-separated data_source names to embed")
    parser.add_argument(
        "--provider",
        choices=["ollama", "fastembed", "fastembed-bge"],
        default="ollama",
        help=(
            "Embedding provider. Ollama matches FR-18 exactly; fastembed uses the same "
            "Nomic 768-dim family locally; fastembed-bge is a faster 384-dim dev corpus."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_services(args.provider)
        ensure_collection(args.provider, recreate=args.recreate)
        if args.provider == "fastembed":
            embedder: Embedder = FastEmbedder(FASTEMBED_MODEL)
        elif args.provider == "fastembed-bge":
            embedder = FastEmbedder(FASTEMBED_BGE_MODEL)
        else:
            embedder = OllamaEmbedder()

        if args.smoke_only:
            smoke_test(embedder)
            return

        selected_sources = set(parse_source_filter(args.source))
        total = 0
        for source, path, grain in DATASETS:
            if selected_sources and source not in selected_sources:
                continue
            if not path.exists():
                print(f"{source}: skipped missing path {path}", file=sys.stderr)
                continue
            total += embed_dataset(source, path, grain, args.limit, embedder)

        print(f"\nEmbedded/upserted {total:,} chunks this run.")
        print(f"Collection {COLLECTION} point count: {collection_count():,}")
        smoke_test(embedder)
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        raise SystemExit(f"HTTP error: {exc}\n{body}") from exc


def parse_source_filter(value: str | None) -> list[str]:
    if not value:
        return []
    available = {source for source, _, _ in DATASETS}
    selected = [source.strip() for source in value.split(",") if source.strip()]
    unknown = sorted(set(selected) - available)
    if unknown:
        raise SystemExit(f"Unknown data_source in --source: {', '.join(unknown)}")
    return selected


if __name__ == "__main__":
    main()
