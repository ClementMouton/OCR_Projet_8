import os

import pandas as pd
import psycopg


def load_production_data() -> pd.DataFrame:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "La variable d'environnement DATABASE_URL n'est pas définie."
        )

    query = """
        SELECT
            id,
            timestamp,
            features,
            default_probability,
            prediction,
            decision_threshold,
            decision_label,
            latency_ms
        FROM prediction_logs
        ORDER BY timestamp ASC;
    """

    with psycopg.connect(database_url) as connection:
        dataframe = pd.read_sql(
            query,
            connection
        )

    return dataframe


def extract_features(
    production_logs: pd.DataFrame
) -> pd.DataFrame:

    if production_logs.empty:
        return pd.DataFrame()

    features = pd.json_normalize(
        production_logs["features"]
    )

    features.index = production_logs.index

    return features


if __name__ == "__main__":
    logs = load_production_data()

    print(f"Nombre de prédictions : {len(logs)}")

    features = extract_features(logs)

    print(f"Nombre de variables : {features.shape[1]}")
    print(features.head())