"""IBM watsonx.ai client for ticket triage.

The model returns JSON with category, priority, summary and a first reply. Every
field is validated against the allowed values; anything invalid is rejected so
the caller falls back to the local model. The AI never invents a category.
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

from .classifier import CATEGORIES, PRIORITIES

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
API_VERSION = "2024-05-01"
DEFAULT_MODEL = "ibm/granite-3-8b-instruct"


class LLMResponseError(ValueError):
    """The model answered, but not with valid triage JSON."""


def build_prompt(ticket: str) -> str:
    return (
        "You are a service desk analyst triaging tickets for a hospital EHR system.\n"
        f"Categories: {', '.join(CATEGORIES)}.\n"
        "Priorities: P1 = patient-safety risk or outage for many users; "
        "P2 = a user or team is blocked from working; "
        "P3 = limited impact or a workaround exists; "
        "P4 = how-to question or service request.\n"
        "Return ONLY a JSON object with keys: category, priority, summary "
        "(one short sentence), reply (2 sentences to the user, no passwords or patient details).\n\n"
        f"Ticket: {ticket}\nJSON:"
    )


def parse_triage_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise LLMResponseError("No JSON object in model output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON: {exc}") from exc
    if data.get("category") not in CATEGORIES:
        raise LLMResponseError(f"Unknown category: {data.get('category')!r}")
    if data.get("priority") not in PRIORITIES:
        raise LLMResponseError(f"Unknown priority: {data.get('priority')!r}")
    return {
        "category": data["category"],
        "priority": data["priority"],
        "summary": str(data.get("summary", "")).strip(),
        "reply": str(data.get("reply", "")).strip(),
    }


class WatsonxClient:
    name = "watsonx"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str = "https://us-south.ml.cloud.ibm.com",
        model_id: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.session = session or requests.Session()
        self._token: str | None = None
        self._token_expiry = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        resp = self.session.post(
            IAM_URL,
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = float(body.get("expiration", time.time() + 3600))
        return self._token

    def triage(self, ticket: str) -> dict:
        payload = {
            "model_id": self.model_id,
            "project_id": self.project_id,
            "input": build_prompt(ticket),
            "parameters": {"decoding_method": "greedy", "max_new_tokens": 250},
        }
        resp = self.session.post(
            f"{self.url}/ml/v1/text/generation",
            params={"version": API_VERSION},
            json=payload,
            headers={"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return parse_triage_json(resp.json()["results"][0]["generated_text"])


def client_from_env() -> WatsonxClient | None:
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not (api_key and project_id):
        return None
    return WatsonxClient(
        api_key=api_key,
        project_id=project_id,
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        model_id=os.getenv("WATSONX_MODEL_ID", DEFAULT_MODEL),
    )
