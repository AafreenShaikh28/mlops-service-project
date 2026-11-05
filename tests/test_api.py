import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def valid_payload():
    return {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }


@pytest.fixture(autouse=True)
def isolated_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    price = response.json()["predicted_price"]
    assert isinstance(price, float)
    assert price > 0


def test_missing_field():
    payload = valid_payload()
    del payload["area"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_wrong_type():
    payload = valid_payload()
    payload["area"] = "large"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_area():
    payload = valid_payload()
    payload["area"] = -500
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_category():
    payload = valid_payload()
    payload["mainroad"] = "maybe"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_service_survives_invalid_input():
    client.post("/predict", json={"area": "bad"})
    assert client.get("/health").status_code == 200