from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.inference.model_loader import ModelService


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
        return model_service.predict(request.features)
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
