import json
import os
from datetime import datetime

import psycopg


DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL n'est pas définie."
        )

    return psycopg.connect(DATABASE_URL)


def init_database() -> None:
    query = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        features JSONB NOT NULL,
        default_probability DOUBLE PRECISION NOT NULL,
        prediction INTEGER NOT NULL,
        decision_threshold DOUBLE PRECISION NOT NULL,
        decision_label TEXT NOT NULL,
        latency_ms DOUBLE PRECISION NOT NULL
    );
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

        connection.commit()


def save_prediction(
    timestamp: datetime,
    features: dict,
    default_probability: float,
    prediction: int,
    decision_threshold: float,
    decision_label: str,
    latency_ms: float,
) -> None:
    query = """
    INSERT INTO prediction_logs (
        timestamp,
        features,
        default_probability,
        prediction,
        decision_threshold,
        decision_label,
        latency_ms
    )
    VALUES (
        %s,
        %s::jsonb,
        %s,
        %s,
        %s,
        %s,
        %s
    );
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    timestamp,
                    json.dumps(
                        features,
                        ensure_ascii=False
                    ),
                    float(default_probability),
                    int(prediction),
                    float(decision_threshold),
                    decision_label,
                    float(latency_ms),
                ),
            )

        connection.commit()