from pathlib import Path
from datetime import datetime, timezone
import json
import threading

from src.monitoring.database import save_prediction


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.jsonl"

_lock = threading.Lock()


def log_prediction(
    features: dict,
    default_probability: float,
    prediction: int,
    decision_threshold: float,
    decision_label: str,
    latency_ms: float,
) -> None:

    # Timestamp unique utilisé pour le fichier et PostgreSQL
    timestamp = datetime.now(timezone.utc)

    # ============================================================
    # 1. LOG LOCAL JSONL
    # ============================================================

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": timestamp.isoformat(),
        "features": features,
        "default_probability": float(
            default_probability
        ),
        "prediction": int(prediction),
        "decision_threshold": float(
            decision_threshold
        ),
        "decision_label": decision_label,
        "latency_ms": float(latency_ms),
    }

    with _lock:
        with LOG_FILE.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ============================================================
    # 2. STOCKAGE POSTGRESQL
    # ============================================================

    try:
        save_prediction(
            timestamp=timestamp,
            features=features,
            default_probability=default_probability,
            prediction=prediction,
            decision_threshold=decision_threshold,
            decision_label=decision_label,
            latency_ms=latency_ms,
        )

    except Exception as error:
        # Le monitoring ne doit jamais empêcher
        # l'API de retourner une prédiction.
        print(
            "Erreur lors de l'enregistrement "
            f"PostgreSQL : {error}"
        )