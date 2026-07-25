from pathlib import Path
import json
from typing import Any

import joblib
import pandas as pd

from src.api.schemas import ModelInfoResponse, PredictionResponse


PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.config: dict[str, Any] = {}
        self.feature_schema: dict[str, Any] = {}
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Charge les artefacts disponibles sans faire planter l'API."""
        self.load_error = None

        self.config = self._load_json(MODEL_CONFIG_PATH)
        self.feature_schema = self._load_json(FEATURE_SCHEMA_PATH)

        if not MODEL_PATH.exists():
            self.model = None
            self.load_error = f"Modèle introuvable : {MODEL_PATH}"
            return

        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as error:
            self.model = None
            self.load_error = str(error)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_model_info(self) -> ModelInfoResponse:
        feature_count = self.feature_schema.get("feature_count")

        if feature_count is None:
            features = self.feature_schema.get("features", [])
            feature_count = len(features) or None

        return ModelInfoResponse(
            model_name=self.config.get(
                "model_name",
                "home_credit_scoring_model",
            ),
            model_version=self.config.get(
                "model_version",
                "unknown",
            ),
            model_type=self.config.get("model_type"),
            decision_threshold=self.config.get(
                "decision_threshold"
            ),
            model_loaded=self.is_loaded,
            feature_count=feature_count,
        )

    def _expected_features(self) -> list[str]:
        feature_order = self.feature_schema.get("feature_order")

        if isinstance(feature_order, list):
            return feature_order

        features = self.feature_schema.get("features", [])
        return [
            feature["name"]
            for feature in features
            if "name" in feature
        ]

    def predict(
        self,
        features: dict[str, Any],
    ) -> PredictionResponse:
        if not self.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé.")

        expected_features = self._expected_features()

        if expected_features:
            missing_features = [
                feature
                for feature in expected_features
                if feature not in features
            ]

            if missing_features:
                preview = ", ".join(missing_features[:10])
                suffix = (
                    "..."
                    if len(missing_features) > 10
                    else ""
                )
                raise ValueError(
                    "Variables manquantes : "
                    f"{preview}{suffix}"
                )

            row = {
                feature: features.get(feature)
                for feature in expected_features
            }
        else:
            row = features

        dataframe = pd.DataFrame([row])

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "Le modèle chargé ne possède pas predict_proba()."
            )

        probability = float(
            self.model.predict_proba(dataframe)[0, 1]
        )

        threshold = float(
            self.config.get("decision_threshold", 0.5)
        )

        prediction = int(probability >= threshold)

        return PredictionResponse(
            default_probability=probability,
            prediction=prediction,
            decision_threshold=threshold,
            decision_label=(
                "risque de défaut"
                if prediction == 1
                else "crédit accordable"
            ),
        )
