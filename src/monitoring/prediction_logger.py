from pathlib import Path
from datetime import datetime, timezone
import json
import threading


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
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
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
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )