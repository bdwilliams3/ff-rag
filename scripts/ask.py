"""
FR-22: Ask the fantasy football RAG system a natural-language question.

Embeds the question, retrieves relevant corpus chunks from Qdrant,
and sends them as context to Gemini for a grounded answer.

Usage:
  .venv/bin/python scripts/ask.py "Who are the best WR targets this season?"
  .venv/bin/python scripts/ask.py --top 20 --position WR "rookies with high upside"
  .venv/bin/python scripts/ask.py --source defensive_coordinator_profiles "best defenses to stream against"
  echo "start/sit: Davante Adams or Stefon Diggs week 6?" | .venv/bin/python scripts/ask.py -

Environment:
  GEMINI_API_KEY  — required
  QDRANT_URL      — default http://localhost:6333
  QDRANT_COLLECTION — default ff-rag-v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from fastembed import TextEmbedding
from openai import OpenAI


EMBEDDING_MODEL = "BAAI/bge-small-en"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "ff-rag-v1"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

SYSTEM_PROMPT = """\
You are a fantasy football analyst helping a team manager make season-long decisions.
Your job: give specific, actionable recommendations grounded in the evidence provided.

The evidence below comes from a RAG corpus containing NFL player stats (2012-2025),
ADP history, combine/draft profiles, coaching staff tendencies (OC/DC profiles),
injury history, college production, and championship roster frequency data.

Rules:
- Always cite specific evidence from the context (player name, season, stat line, source).
- Rank or tier your recommendations when the question asks "who" or "best".
- Flag uncertainty when context is thin (e.g. "limited data on this player").
- For matchup questions, pull from DC profiles to assess defensive tendencies.
- For rookie questions, use combine measurables + college production + ADP.
- For new-team fit, reference OC pass rate/target share splits.
- Be concise: bullet points over paragraphs, numbers over adjectives.
"""


def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")


def qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)


def parse_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def build_filter(args: argparse.Namespace) -> dict | None:
    must = []
    if args.position:
        pos = args.position.upper()
        if pos not in VALID_POSITIONS:
            raise SystemExit(f"--position must be one of: {', '.join(sorted(VALID_POSITIONS))}")
        must.append({"key": "position", "match": {"value": pos}})
    sources = parse_sources(args.source)
    if len(sources) == 1:
        must.append({"key": "data_source", "match": {"value": sources[0]}})
    elif len(sources) > 1:
        must.append({"key": "data_source", "match": {"any": sources}})
    if args.season_min is not None:
        must.append({"key": "season", "range": {"gte": args.season_min}})
    if args.season_max is not None:
        must.append({"key": "season", "range": {"lte": args.season_max}})
    return {"must": must} if must else None


def embed(model: TextEmbedding, query: str) -> list[float]:
    return next(model.query_embed(f"search_query: {query}")).tolist()


def search(vector: list[float], qdrant_filter: dict | None, top: int) -> list[dict]:
    body: dict = {"vector": vector, "limit": top * 6, "with_payload": True}
    if qdrant_filter:
        body["filter"] = qdrant_filter
    r = requests.post(
        f"{qdrant_url()}/collections/{qdrant_collection()}/points/search",
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("result", [])


def diversify(points: list[dict], top: int) -> list[dict]:
    selected, selected_ids, seen_sources = [], set(), set()
    for p in points:
        src = p.get("payload", {}).get("data_source")
        pid = str(p.get("id"))
        if src and src not in seen_sources:
            selected.append(p)
            selected_ids.add(pid)
            seen_sources.add(src)
        if len(selected) >= top:
            return selected
    for p in points:
        if str(p.get("id")) not in selected_ids:
            selected.append(p)
        if len(selected) >= top:
            break
    return sorted(selected, key=lambda x: x.get("score", 0), reverse=True)


def build_context(points: list[dict]) -> str:
    lines = []
    for i, p in enumerate(points, 1):
        payload = p.get("payload") or {}
        text = payload.get("text", "").strip()
        src = payload.get("data_source", "unknown")
        score = p.get("score", 0)
        lines.append(f"[{i}] ({src}, score={score:.3f}) {text}")
    return "\n".join(lines)


def ask_gemini(client: OpenAI, question: str, context: str, stream: bool) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"RETRIEVED EVIDENCE ({context.count(chr(10)) + 1} chunks):\n\n"
                f"{context}\n\n"
                f"---\n\n"
                f"QUESTION: {question}"
            ),
        },
    ]

    if stream:
        with client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=messages,
            stream=True,
        ) as response:
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
        print()
    else:
        response = client.chat.completions.create(model=GEMINI_MODEL, messages=messages)
        print(response.choices[0].message.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the ff-rag system a question")
    parser.add_argument("question", nargs="?", help="Question (or '-' to read from stdin)")
    parser.add_argument("--top", type=int, default=15, help="Chunks to retrieve (default 15)")
    parser.add_argument("--position", help="Filter by position: QB, RB, WR, TE")
    parser.add_argument("--season-min", type=int, help="Earliest season to include")
    parser.add_argument("--season-max", type=int, help="Latest season to include")
    parser.add_argument("--source", help="Comma-separated data_source filter")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before the answer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY environment variable")

    question = args.question
    if question == "-" or (question is None and not sys.stdin.isatty()):
        question = sys.stdin.read().strip()
    if not question:
        raise SystemExit("Provide a question as an argument or via stdin")

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    print("Embedding query…", file=sys.stderr)
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    vector = embed(model, question)

    print("Searching corpus…", file=sys.stderr)
    points = search(vector, build_filter(args), args.top)
    chunks = diversify(points, args.top)
    print(f"Retrieved {len(chunks)} chunks from {len({p.get('payload',{}).get('data_source') for p in chunks})} sources", file=sys.stderr)

    context = build_context(chunks)

    if args.show_context:
        print("\n--- RETRIEVED CONTEXT ---")
        print(context)
        print("--- END CONTEXT ---\n")

    print(f"\n{'='*60}", file=sys.stderr)
    ask_gemini(client, question, context, stream=not args.no_stream)


if __name__ == "__main__":
    main()
