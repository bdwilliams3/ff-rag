"""
FR-20: query the local ff-rag dev corpus.

Embeds a natural-language draft query with FastEmbed's BGE model, searches the
local Qdrant collection, and prints ranked context chunks as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests
from fastembed import TextEmbedding


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "ff-rag-v1"
EMBEDDING_MODEL = "BAAI/bge-small-en"
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")


def qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)


def parse_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [source.strip() for source in value.split(",") if source.strip()]


def build_filter(args: argparse.Namespace) -> dict[str, Any] | None:
    must: list[dict[str, Any]] = []

    if args.position:
        position = args.position.upper()
        if position not in VALID_POSITIONS:
            raise SystemExit(f"--position must be one of: {', '.join(sorted(VALID_POSITIONS))}")
        must.append({"key": "position", "match": {"value": position}})

    sources = parse_sources(args.source)
    if len(sources) == 1:
        must.append({"key": "data_source", "match": {"value": sources[0]}})
    elif len(sources) > 1:
        must.append({"key": "data_source", "match": {"any": sources}})

    season_range = {}
    if args.season_min is not None:
        season_range["gte"] = args.season_min
    if args.season_max is not None:
        season_range["lte"] = args.season_max
    if season_range:
        must.append({"key": "season", "range": season_range})

    return {"must": must} if must else None


def embed_query(model: TextEmbedding, query: str) -> list[float]:
    return next(model.query_embed(f"search_query: {query}")).tolist()


def search_qdrant(vector: list[float], qdrant_filter: dict[str, Any] | None, top: int) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "vector": vector,
        "limit": max(top * 8, top),
        "with_payload": True,
    }
    if qdrant_filter:
        body["filter"] = qdrant_filter

    response = requests.post(
        f"{qdrant_url()}/collections/{qdrant_collection()}/points/search",
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("result", [])


def diversify(points: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    """Prefer source variety first, then fill remaining slots by raw score."""
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_sources: set[str] = set()

    for point in points:
        source = point.get("payload", {}).get("data_source")
        point_id = str(point.get("id"))
        if source and source not in seen_sources:
            selected.append(point)
            selected_ids.add(point_id)
            seen_sources.add(source)
        if len(selected) >= top:
            return selected

    for point in points:
        point_id = str(point.get("id"))
        if point_id not in selected_ids:
            selected.append(point)
        if len(selected) >= top:
            return sorted(selected, key=lambda item: item.get("score", 0), reverse=True)

    return sorted(selected, key=lambda item: item.get("score", 0), reverse=True)


def result_item(point: dict[str, Any]) -> dict[str, Any]:
    payload = point.get("payload") or {}
    score = point.get("score")
    item = {
        "score": round(score, 6) if isinstance(score, float) else score,
        "text": payload.get("text"),
        "data_source": payload.get("data_source"),
    }
    for key in [
        "player_id",
        "player_name",
        "position",
        "season",
        "week",
        "team",
        "opponent_team",
        "format",
        "league_type",
        "coordinator",
        "head_coach",
        "grain",
        "source_path",
        "row_key",
    ]:
        if key in payload:
            value = payload[key]
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            item[key] = value
    return item


def run_query(model: TextEmbedding, args: argparse.Namespace) -> list[dict[str, Any]]:
    vector = embed_query(model, args.query)
    points = search_qdrant(vector, build_filter(args), args.top)
    return [result_item(point) for point in diversify(points, args.top)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--position", help="Filter by fantasy position: QB, RB, WR, TE")
    parser.add_argument("--season-min", type=int)
    parser.add_argument("--season-max", type=int)
    parser.add_argument("--source", help="Comma-separated data_source filter")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Run the same query N times after model load and print latency summary to stderr",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be >= 1")

    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    if args.benchmark:
        latencies = []
        output = []
        for _ in range(args.benchmark):
            started = time.perf_counter()
            output = run_query(model, args)
            latencies.append((time.perf_counter() - started) * 1000)
        avg = sum(latencies) / len(latencies)
        print(
            f"benchmark_ms min={min(latencies):.1f} avg={avg:.1f} max={max(latencies):.1f} n={len(latencies)}",
            file=sys.stderr,
        )
    else:
        output = run_query(model, args)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
