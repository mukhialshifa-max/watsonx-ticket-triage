"""Combines watsonx.ai and the local model into one triage result with routing."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass

import requests

from .classifier import NaiveBayesClassifier, load_tickets, score_priority
from .llm import LLMResponseError, WatsonxClient

logger = logging.getLogger("triage")

ROUTING = {
    "Access & Login": "Identity & Access Team",
    "Scheduling": "Scheduling Application Team",
    "Registration": "Patient Registration / HIM Team",
    "Clinical Documentation": "Clinical Applications Team",
    "Results & Inbox": "Clinical Applications Team",
    "Printing & Devices": "Desktop & Device Support",
    "Network & Connectivity": "Network Operations",
}

# Target time to first response, in hours.
SLA_HOURS = {"P1": 0.25, "P2": 1, "P3": 8, "P4": 24}

DEFAULT_REPLIES = {
    "P1": "Thank you, this is being treated as urgent and a support analyst is working on it now. Please follow your unit's downtime or safety procedure in the meantime.",
    "P2": "Thank you, we understand this is stopping your work and an analyst will contact you within the hour. Please keep any error message or screenshot ready.",
    "P3": "Thank you for reporting this. An analyst will follow up within one business day.",
    "P4": "Thank you for your request. We will reply within one business day with the steps or the access you need.",
}

LOW_CONFIDENCE = 0.5


@dataclass
class TriageResult:
    category: str
    priority: str
    team: str
    sla_hours: float
    summary: str
    reply: str
    reason: str
    confidence: float | None
    needs_review: bool
    engine: str

    def to_dict(self) -> dict:
        return asdict(self)


class TriageService:
    def __init__(self, model: NaiveBayesClassifier, llm: WatsonxClient | None = None):
        self.model = model
        self.llm = llm

    @classmethod
    def from_training_data(cls, llm: WatsonxClient | None = None) -> "TriageService":
        tickets = load_tickets()
        model = NaiveBayesClassifier().fit([t.text for t in tickets], [t.category for t in tickets])
        return cls(model, llm)

    @property
    def engine(self) -> str:
        return "watsonx" if self.llm else "local"

    def triage(self, text: str) -> TriageResult:
        text = text.strip()
        if self.llm:
            try:
                ai = self.llm.triage(text)
                return self._build(
                    text, ai["category"], ai["priority"], ai["summary"], ai["reply"],
                    reason="Classified by IBM watsonx.ai", confidence=None, engine="watsonx",
                )
            except (requests.RequestException, LLMResponseError, KeyError, IndexError) as exc:
                logger.warning("watsonx.ai triage failed, using local model: %s", exc)
                return self._local(text, engine="local-fallback")
        return self._local(text, engine="local")

    def _local(self, text: str, engine: str) -> TriageResult:
        category, confidence = self.model.predict(text)
        prio = score_priority(text)
        return self._build(text, category, prio.priority, "", "", prio.reason, confidence, engine)

    def _build(self, text, category, priority, summary, reply, reason, confidence, engine) -> TriageResult:
        # Rules can only raise urgency, never lower it: a safety signal always wins.
        rule = score_priority(text)
        if rule.priority == "P1" and priority != "P1":
            priority, reason = "P1", rule.reason
        needs_review = priority == "P1" or (confidence is not None and confidence < LOW_CONFIDENCE)
        return TriageResult(
            category=category,
            priority=priority,
            team=ROUTING[category],
            sla_hours=SLA_HOURS[priority],
            summary=summary or (text[:117] + "..." if len(text) > 120 else text),
            reply=reply or DEFAULT_REPLIES[priority],
            reason=reason,
            confidence=confidence,
            needs_review=needs_review,
            engine=engine,
        )

    def triage_batch(self, texts: list[str]) -> dict:
        results = [self.triage(t) for t in texts]
        order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
        ranked = sorted(zip(texts, results), key=lambda tr: order[tr[1].priority])
        return {
            "total": len(results),
            "by_priority": dict(sorted(Counter(r.priority for r in results).items())),
            "by_team": dict(Counter(r.team for r in results).most_common()),
            "needs_review": sum(r.needs_review for r in results),
            "tickets": [{"text": t, **r.to_dict()} for t, r in ranked],
        }
