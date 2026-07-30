from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.inference.model_loader import ModelService
from time import perf_counter

from src.monitoring.prediction_logger import log_prediction

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage sans empêcher l'API de fonctionner."""
    model_service.load()
    yield


app = FastAPI(
    title="Home Credit Scoring API",
    description="API de prédiction du risque de défaut client.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Home Credit Scoring API",
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info() -> ModelInfoResponse:
    return model_service.get_model_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Le modèle n'est pas disponible. "
                "Ajoute artifacts/model_pipeline.joblib "
                "puis redémarre l'API."
            ),
        )

    try:
        start_time = perf_counter()

        result = model_service.predict(
            request.features
        )

        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        log_prediction(
            features=request.features,
            default_probability=result.default_probability,
            prediction=result.prediction,
            decision_threshold=result.decision_threshold,
            decision_label=result.decision_label,
            latency_ms=latency_ms,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne pendant la prédiction.",
        ) from error
