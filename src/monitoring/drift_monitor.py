import os

import pandas as pd
import psycopg


REFERENCE_DATA_PATH = "data/reference/reference_data.parquet"


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
            connection,
        )

    return dataframe


def extract_features(
    production_logs: pd.DataFrame,
) -> pd.DataFrame:

    if production_logs.empty:
        return pd.DataFrame()

    features = pd.json_normalize(
        production_logs["features"]
    )

    features.index = production_logs.index

    return features


def load_reference_data() -> pd.DataFrame:
    return pd.read_parquet(
        REFERENCE_DATA_PATH
    )


def validate_feature_schema(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
) -> None:

    reference_columns = set(reference_data.columns)
    production_columns = set(production_data.columns)

    missing_in_production = (
        reference_columns - production_columns
    )

    unexpected_in_production = (
        production_columns - reference_columns
    )

    print("\n=== Validation du schéma ===")

    print(
        f"Features référence : "
        f"{len(reference_columns)}"
    )

    print(
        f"Features production : "
        f"{len(production_columns)}"
    )

    if missing_in_production:
        print(
            f"Features manquantes en production : "
            f"{len(missing_in_production)}"
        )
        print(sorted(missing_in_production))

    if unexpected_in_production:
        print(
            f"Features supplémentaires en production : "
            f"{len(unexpected_in_production)}"
        )
        print(sorted(unexpected_in_production))

    if (
        not missing_in_production
        and not unexpected_in_production
    ):
        print("✓ Schémas compatibles")


if __name__ == "__main__":

    logs = load_production_data()
    production_features = extract_features(logs)

    print(
        f"Nombre de prédictions production : "
        f"{len(production_features)}"
    )

    reference_data = load_reference_data()

    print(
        f"Nombre de lignes référence : "
        f"{len(reference_data)}"
    )

    validate_feature_schema(
        reference_data,
        production_features,
    )