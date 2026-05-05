"""FastAPI web app for FR-22."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.emailer import execute_actions, load_recipients
from app.llm import call_llm
from app.rag import QueryFilters, RagClient


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

app = FastAPI(title="FF RAG Draft Advisor")
rag_client = RagClient()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top: int = Field(default=14, ge=1, le=40)
    position: Optional[str] = None
    season_min: Optional[int] = None
    season_max: Optional[int] = None
    source: Optional[str] = None


class EmailRequest(BaseModel):
    recipient_alias: str
    subject: str
    body: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "recipients": sorted(load_recipients())}


@app.get("/api/recipients")
def recipients() -> dict[str, str]:
    return load_recipients()


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        filters = QueryFilters(
            position=request.position,
            season_min=request.season_min,
            season_max=request.season_max,
            source=request.source,
            top=request.top,
        )
        chunks = rag_client.retrieve(request.message, filters)
        prompt = rag_client.build_prompt(request.message, chunks, load_recipients())
        llm_response = call_llm(prompt)
        action_results = execute_actions(llm_response.get("actions", []))
        return {
            "answer": llm_response["answer"],
            "tables": llm_response["tables"],
            "lists": llm_response["lists"],
            "citations": llm_response["citations"],
            "actions": llm_response["actions"],
            "action_results": action_results,
            "chunks": chunks,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/email")
def send_email(request: EmailRequest) -> dict[str, Any]:
    return execute_actions([
        {
            "type": "send_email",
            "recipient_alias": request.recipient_alias,
            "subject": request.subject,
            "body": request.body,
        }
    ])[0]
