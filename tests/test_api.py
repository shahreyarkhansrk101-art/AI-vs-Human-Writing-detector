import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

AI_LIKE_TEXT = (
    "Furthermore, artificial intelligence continues to reshape modern industries. "
    "In addition, it is worth noting that cloud computing has transformed daily "
    "productivity. Overall, this reflects a broader trend worth considering."
)

HUMAN_LIKE_TEXT = (
    "honestly idk why but I love how smartphones hav changed everything lol!! "
    "anyway cloud computing is pretty cool too, at least imo. could be wrong tho."
)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_flags_ai_like_text():
    r = client.post("/predict", json={"text": AI_LIKE_TEXT})
    assert r.status_code == 200
    body = r.json()
    assert body["is_ai_generated"] is True
    assert "features" in body


def test_predict_passes_human_like_text():
    r = client.post("/predict", json={"text": HUMAN_LIKE_TEXT})
    assert r.status_code == 200
    assert r.json()["is_ai_generated"] is False


def test_predict_rejects_too_short_text():
    r = client.post("/predict", json={"text": "too short"})
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "best_model" in r.json()


def test_example_each_subtype():
    for subtype in ("ai_standard", "ai_casual", "human_casual", "human_formal"):
        r = client.get(f"/example/{subtype}")
        assert r.status_code == 200
        assert r.json()["subtype"] == subtype


def test_example_invalid_subtype_rejected():
    r = client.get("/example/not_a_real_subtype")
    assert r.status_code == 422
