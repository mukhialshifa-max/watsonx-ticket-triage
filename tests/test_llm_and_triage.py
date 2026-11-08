import json
from unittest.mock import MagicMock

import pytest
import requests

from app.llm import LLMResponseError, WatsonxClient, build_prompt, client_from_env, parse_triage_json
from app.triage import TriageService


def _resp(body):
    r = MagicMock()
    r.json.return_value = body
    r.raise_for_status.return_value = None
    return r


def _watsonx_session(generated_text):
    session = MagicMock()
    session.post.side_effect = [
        _resp({"access_token": "tok", "expiration": 9_999_999_999}),
        _resp({"results": [{"generated_text": generated_text}]}),
    ]
    return session


GOOD = json.dumps({
    "category": "Printing & Devices", "priority": "P3",
    "summary": "Ward printer jammed.", "reply": "We will send a technician.",
})


def test_prompt_lists_allowed_values():
    prompt = build_prompt("printer jammed")
    assert "Network & Connectivity" in prompt and "P4 = how-to" in prompt
    assert prompt.endswith("Ticket: printer jammed\nJSON:")


def test_parse_valid_json_with_extra_text():
    assert parse_triage_json("Sure! " + GOOD + " Hope this helps")["priority"] == "P3"


@pytest.mark.parametrize(
    "text",
    ["no json here", "{not valid json}",
     '{"category": "Billing", "priority": "P2"}',
     '{"category": "Scheduling", "priority": "urgent"}'],
)
def test_parse_rejects_invalid_output(text):
    with pytest.raises(LLMResponseError):
        parse_triage_json(text)


def test_watsonx_client_requests():
    session = _watsonx_session(GOOD)
    client = WatsonxClient("key", "proj", url="https://ca-tor.ml.cloud.ibm.com/", session=session)
    assert client.triage("printer jammed")["category"] == "Printing & Devices"
    iam, gen = session.post.call_args_list
    assert iam.args[0] == "https://iam.cloud.ibm.com/identity/token"
    assert gen.args[0] == "https://ca-tor.ml.cloud.ibm.com/ml/v1/text/generation"
    assert gen.kwargs["headers"]["Authorization"] == "Bearer tok"
    assert gen.kwargs["json"]["project_id"] == "proj"


def test_client_from_env(monkeypatch):
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    assert client_from_env() is None
    monkeypatch.setenv("WATSONX_API_KEY", "k")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    assert isinstance(client_from_env(), WatsonxClient)


@pytest.fixture(scope="module")
def local():
    return TriageService.from_training_data()


def test_local_triage_routes_and_sets_sla(local):
    r = local.triage("Labels are not printing on the lab printer")
    assert r.category == "Printing & Devices"
    assert r.team == "Desktop & Device Support"
    assert r.priority == "P2" and r.sla_hours == 1
    assert r.engine == "local" and r.reply


def test_p1_always_needs_human_review(local):
    r = local.triage("The EHR is down for the whole hospital")
    assert r.priority == "P1" and r.needs_review


def test_watsonx_result_is_used():
    svc = TriageService.from_training_data(WatsonxClient("k", "p", session=_watsonx_session(GOOD)))
    r = svc.triage("The ward printer is jammed")
    assert r.engine == "watsonx"
    assert r.summary == "Ward printer jammed."
    assert r.team == "Desktop & Device Support"


def test_safety_rule_overrides_low_ai_priority():
    ai = json.dumps({"category": "Clinical Documentation", "priority": "P3", "summary": "s", "reply": "r"})
    svc = TriageService.from_training_data(WatsonxClient("k", "p", session=_watsonx_session(ai)))
    r = svc.triage("MAR shows the wrong dose for my patient")
    assert r.priority == "P1"
    assert r.reason == "Possible patient-safety risk"


def test_falls_back_when_ai_returns_bad_json():
    svc = TriageService.from_training_data(WatsonxClient("k", "p", session=_watsonx_session("I am not sure")))
    assert svc.triage("Wi-Fi keeps disconnecting").engine == "local-fallback"


def test_falls_back_when_watsonx_unreachable():
    session = MagicMock()
    session.post.side_effect = requests.ConnectionError("down")
    svc = TriageService.from_training_data(WatsonxClient("k", "p", session=session))
    assert svc.triage("Wi-Fi keeps disconnecting").engine == "local-fallback"


def test_batch_is_sorted_by_priority(local):
    out = local.triage_batch([
        "How do I change my default printer",
        "The EHR is down for the whole hospital",
        "I am locked out of my account",
    ])
    assert [t["priority"] for t in out["tickets"]] == ["P1", "P2", "P4"]
    assert out["total"] == 3 and out["needs_review"] >= 1
