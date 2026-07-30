import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


PROJECT_DIR = Path(__file__).resolve().parents[1]

VALID_CLIENT_PATH = (
    PROJECT_DIR
    / "tests"
    / "fixtures"
    / "valid_client.json"
)

EXPECTED_PREDICTION_PATH = (
    PROJECT_DIR
    / "tests"
    / "fixtures"
    / "expected_prediction.json"
)


@pytest.fixture
def client():
    # Le context manager déclenche bien le lifespan FastAPI
    # et donc le chargement du modèle.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_client():
    with VALID_CLIENT_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@pytest.fixture
def expected_prediction():
    with EXPECTED_PREDICTION_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "home_credit_scoring_model"
    assert data["model_version"] == "1.0.0"
    assert data["model_type"] == "LogisticRegression"
    assert data["decision_threshold"] == pytest.approx(0.51)
    assert data["model_loaded"] is True
    assert data["feature_count"] == 443


def test_predict_valid_client(
    client,
    valid_client,
    expected_prediction,
):
    response = client.post(
        "/predict",
        json={
            "features": valid_client
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["default_probability"] == pytest.approx(
        expected_prediction["default_probability"],
        rel=1e-10,
        abs=1e-12,
    )

    assert (
        data["prediction"]
        == expected_prediction["prediction"]
    )

    assert data["decision_threshold"] == pytest.approx(
        expected_prediction["decision_threshold"]
    )

    assert data["decision_label"] == "crédit accordable"


def test_predict_missing_features(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "AMT_INCOME_TOTAL": 200000
            }
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert "Variables manquantes" in data["detail"]

def test_predict_creates_log(
    client,
    valid_client,
    tmp_path,
    monkeypatch,
):
    from src.monitoring import prediction_logger

    log_file = tmp_path / "predictions.jsonl"

    monkeypatch.setattr(
        prediction_logger,
        "LOG_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        prediction_logger,
        "LOG_FILE",
        log_file,
    )

    response = client.post(
        "/predict",
        json={
            "features": valid_client
        },
    )

    assert response.status_code == 200
    assert log_file.exists()

    lines = log_file.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 1

    log_data = json.loads(lines[0])

    assert "timestamp" in log_data
    assert "features" in log_data
    assert "default_probability" in log_data
    assert "prediction" in log_data
    assert "decision_threshold" in log_data
    assert "decision_label" in log_data
    assert "latency_ms" in log_data

    assert log_data["latency_ms"] >= 0