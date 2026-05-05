"""Retrieval and prompt assembly for the FR-22 chat app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from fastembed import TextEmbedding

from app.analysis import ProjectionContext


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "ff-rag-v1"
EMBEDDING_MODEL = "BAAI/bge-small-en"
VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


DATA_SOURCE_GUIDE = {
    "weekly_stats": "player-week NFL production; best for recent form, volatility, and game-level usage",
    "seasonal_stats": "player-season NFL totals; seasonal share fields are avg_weekly_* when present",
    "combine_draft": "draft capital, college, size, and athletic profile",
    "college_stats": "college production by season; useful for rookies and prospects",
    "injuries": "aggregated injury report history by player-season",
    "adp": "FantasyPros draft cost by season and scoring format",
    "coaching_weekly": "team-week head coach and coordinator assignments",
    "offensive_coordinator_profiles": "team-season offensive tendencies under an OC",
    "defensive_coordinator_profiles": "team-season production allowed under a DC",
    "player_salaries": "salary, cap, and team investment by player-season",
    "salary_cap": "league salary-cap context by season",
    "league_winner_frequency": "championship roster appearance rates; sample sizes may be small",
    "fantasypros_championship_rosters": "small external championship roster notes",
    "nfl_schedule_2026": "team-game schedule rows when the 2026 schedule is available",
}


@dataclass
class QueryFilters:
    position: str | None = None
    season_min: int | None = None
    season_max: int | None = None
    source: str | None = None
    top: int = 14


class RagClient:
    def __init__(self) -> None:
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.projections = ProjectionContext()

    def qdrant_url(self) -> str:
        return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")

    def qdrant_collection(self) -> str:
        return os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)

    def retrieve(self, query: str, filters: QueryFilters) -> list[dict[str, Any]]:
        filters = self._augment_filters_for_projection(query, filters)
        vector = next(self.model.query_embed(f"search_query: {query}")).tolist()
        points = self._search_qdrant(vector, self._build_filter(filters), filters.top)
        return [self._result_item(point) for point in self._diversify(points, filters.top)]

    def build_prompt(self, question: str, chunks: list[dict[str, Any]], recipients: dict[str, str]) -> str:
        context_lines = []
        for idx, chunk in enumerate(chunks, start=1):
            label_bits = [chunk.get("data_source", "unknown")]
            if chunk.get("season"):
                label_bits.append(str(chunk["season"]))
            if chunk.get("format"):
                label_bits.append(str(chunk["format"]))
            label = " ".join(label_bits)
            context_lines.append(f"[{idx}] [{label}] {chunk.get('text', '')}")

        source_guide = "\n".join(f"- {key}: {value}" for key, value in DATA_SOURCE_GUIDE.items())
        recipient_aliases = ", ".join(sorted(recipients)) or "(none configured)"
        derived_context = self.projections.build(question)

        return f"""\
You are answering inside a local fantasy football draft assistant.

User question:
{question}

Retrieved evidence:
{chr(10).join(context_lines)}

Derived analysis context:
{derived_context or "(none)"}

Data source guide:
{source_guide}

Configured email recipient aliases:
{recipient_aliases}

Return one strict JSON object with this schema:
{{
  "answer": "concise recommendation or analysis in markdown",
  "tables": [
    {{
      "title": "table title",
      "columns": ["Player", "Why", "Risk"],
      "rows": [["Name", "Evidence-backed reason", "Risk note"]]
    }}
  ],
  "lists": [
    {{
      "title": "list title",
      "items": ["ranked or bulleted item"]
    }}
  ],
  "citations": [
    {{
      "data_source": "seasonal_stats",
      "detail": "Justin Jefferson 2023: 68 receptions, 1074 receiving yards"
    }}
  ],
  "actions": [
    {{
      "type": "send_email",
      "recipient_alias": "ant",
      "subject": "Top 10 WRs",
      "body": "Email body to send"
    }}
  ]
}}

Rules:
- Use only evidence from the retrieved chunks. If evidence is thin, say so.
- For future or upcoming season questions, perform forward-looking draft analysis from the latest available data; clearly label it as a projection/ranking derived from historical data instead of refusing because no future stat rows exist.
- When Derived analysis context is present, use it as the primary ranking scaffold and use retrieved chunks as supporting evidence.
- Never label latest-season actual points as projected points. For example, call them "2025 Actual Points" and reserve "2026 Projection" for rank/tier/opinion columns.
- If the question asks for top/ranked players, return at least 10 rows unless the user asks for a smaller number.
- Do not let isolated weekly outliers override the Derived analysis context for future-season rankings.
- For rookie/prospect questions, use Derived analysis context as the ranking scaffold and explain that draft capital plus production drives the board, with athletic testing as an upside/tie-breaker.
- Put rankings and comparisons in tables or lists when useful.
- Cite data_source values in citations and in the answer when relevant.
- Do not cite raw seasonal target_share or air_yards_share values; use avg_weekly_* labels if present.
- Only request email actions for configured aliases. Never invent email addresses.
- If the user asks to email something, include exactly one send_email action with the answer content summarized in the body.
- Return JSON only. No markdown fences.
"""

    def _search_qdrant(
        self,
        vector: list[float],
        qdrant_filter: dict[str, Any] | None,
        top: int,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "vector": vector,
            "limit": max(top * 8, top),
            "with_payload": True,
        }
        if qdrant_filter:
            body["filter"] = qdrant_filter
        response = requests.post(
            f"{self.qdrant_url()}/collections/{self.qdrant_collection()}/points/search",
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("result", [])

    def _build_filter(self, filters: QueryFilters) -> dict[str, Any] | None:
        must: list[dict[str, Any]] = []

        if filters.position:
            position = filters.position.upper()
            if position not in VALID_POSITIONS:
                raise ValueError(f"position must be one of: {', '.join(sorted(VALID_POSITIONS))}")
            must.append({"key": "position", "match": {"value": position}})

        sources = [source.strip() for source in (filters.source or "").split(",") if source.strip()]
        if len(sources) == 1:
            must.append({"key": "data_source", "match": {"value": sources[0]}})
        elif len(sources) > 1:
            must.append({"key": "data_source", "match": {"any": sources}})

        season_range = {}
        if filters.season_min is not None:
            season_range["gte"] = filters.season_min
        if filters.season_max is not None:
            season_range["lte"] = filters.season_max
        if season_range:
            must.append({"key": "season", "range": season_range})

        return {"must": must} if must else None

    def _diversify(self, points: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
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
                break

        return sorted(selected, key=lambda item: item.get("score", 0), reverse=True)

    def _augment_filters_for_projection(self, query: str, filters: QueryFilters) -> QueryFilters:
        if self.projections.is_rookie_question(query):
            source = filters.source or "combine_draft,college_stats,adp"
            return QueryFilters(
                position=filters.position,
                season_min=filters.season_min,
                season_max=filters.season_max,
                source=source,
                top=filters.top,
            )

        if not self.projections.is_projection_question(query):
            return filters

        position = filters.position or self.projections.detect_position(query)
        source = filters.source
        if not source:
            source = ",".join([
                "seasonal_stats",
                "adp",
                "injuries",
                "player_salaries",
                "league_winner_frequency",
            ])
        return QueryFilters(
            position=position,
            season_min=filters.season_min,
            season_max=filters.season_max,
            source=source,
            top=filters.top,
        )

    def _result_item(self, point: dict[str, Any]) -> dict[str, Any]:
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
        ]:
            if key in payload:
                value = payload[key]
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                item[key] = value
        return item
