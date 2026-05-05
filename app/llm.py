"""LLM client for FR-22."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-2.0-flash-lite"]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM_PROMPT = """\
You are a fantasy football draft advisor with access to retrieved historical NFL,
fantasy, injury, ADP, coaching, salary, athletic, and league-winner context.

Be specific, practical, and evidence-grounded. Do not invent stats. When the user
asks for rankings, create clear tables or ranked lists. When the user asks to email
an answer, request a send_email action only for an allowlisted alias supplied in
the prompt.
"""


def call_llm(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to call the LLM")

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    }
    configured_model = os.getenv("GEMINI_MODEL")
    models = [configured_model] if configured_model else [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
    last_error = None
    response = None
    for model in models:
        response = requests.post(
            f"{GEMINI_BASE_URL}/models/{model}:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if response.status_code < 400:
            break
        last_error = f"Gemini API error {response.status_code}: {response.text}"
        if response.status_code not in {404, 429, 503}:
            break
    if response is None or response.status_code >= 400:
        raise RuntimeError(last_error or "Gemini API request failed")
    content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return parse_structured_response(content)


def parse_structured_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"answer": content, "tables": [], "lists": [], "citations": [], "actions": []}
    return normalize_response(data)


def normalize_response(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": str(data.get("answer") or ""),
        "tables": data.get("tables") if isinstance(data.get("tables"), list) else [],
        "lists": data.get("lists") if isinstance(data.get("lists"), list) else [],
        "citations": data.get("citations") if isinstance(data.get("citations"), list) else [],
        "actions": data.get("actions") if isinstance(data.get("actions"), list) else [],
    }
