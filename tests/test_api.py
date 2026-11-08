from fastapi.testclient import TestClient

from app.main import create_app
from app.triage import TriageService

client = TestClient(create_app(TriageService.from_training_data()))


def test_health():
    assert client.get("/health").json() == {"status": "ok", "engine": "local"}


def test_triage_single():
    body = client.post("/triage", json={"text": "I forgot my password and I am locked out"}).json()
    assert body["category"] == "Access & Login"
    assert body["priority"] == "P2"
    assert body["team"] == "Identity & Access Team"


def test_triage_batch():
    body = client.post("/triage/batch", json={"tickets": ["Wi-Fi keeps disconnecting on 4 West", "  ", "How do I forward a result"]}).json()
    assert body["total"] == 2
    assert sum(body["by_priority"].values()) == 2


def test_validation():
    assert client.post("/triage", json={"text": "hi"}).status_code == 422
    assert client.post("/triage/batch", json={"tickets": []}).status_code == 422


def test_config_and_page():
    cfg = client.get("/config").json()
    assert len(cfg["categories"]) == 7 and cfg["sla_hours"]["P1"] == 0.25
    assert "Ticket Triage" in client.get("/").text
