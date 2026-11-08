"""FastAPI app: single and batch triage endpoints plus a dashboard page."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .classifier import CATEGORIES, PRIORITIES
from .llm import client_from_env
from .triage import ROUTING, SLA_HOURS, TriageService

load_dotenv()
STATIC_DIR = Path(__file__).resolve().parent / "static"


class TicketIn(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000)


class BatchIn(BaseModel):
    tickets: list[str] = Field(..., min_length=1, max_length=200)


def create_app(service: TriageService | None = None) -> FastAPI:
    service = service or TriageService.from_training_data(client_from_env())
    app = FastAPI(
        title="watsonx Ticket Triage",
        description="Classifies, prioritizes and routes EHR support tickets with IBM watsonx.ai.",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "engine": service.engine}

    @app.get("/config")
    def config() -> dict:
        return {"categories": CATEGORIES, "priorities": PRIORITIES, "routing": ROUTING, "sla_hours": SLA_HOURS}

    @app.post("/triage")
    def triage(ticket: TicketIn) -> dict:
        return service.triage(ticket.text).to_dict()

    @app.post("/triage/batch")
    def triage_batch(batch: BatchIn) -> dict:
        texts = [t.strip() for t in batch.tickets if t.strip()]
        return service.triage_batch(texts)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
