from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_model_info() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    assert "model_name" in response.json()


def test_predict_without_model_returns_503() -> None:
    response = client.post(
        "/predict",
        json={"features": {"AMT_INCOME_TOTAL": 200000}},
    )

def test_predict_without_model_returns_503() -> None:
    response = client.post(
        "/predict",
        json={
            "features": {
                "AMT_INCOME_TOTAL": 200000
            }
        },
    )

    assert response.status_code == 503
    assert "modèle n'est pas disponible" in response.json()["detail"]
