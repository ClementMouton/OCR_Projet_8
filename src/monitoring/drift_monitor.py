import os
from pathlib import Path

import pandas as pd
import psycopg

from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_DATA_PATH = "data/reference/reference_data.parquet"
REPORT_DIR = Path("reports")
DRIFT_REPORT_PATH = REPORT_DIR / "drift_report.html"


def load_production_data() -> pd.DataFrame:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "La variable d'environnement DATABASE_URL n'est pas définie."
        )

    min_id_raw = os.getenv("PRODUCTION_MIN_ID", "0")

    try:
        min_id = int(min_id_raw)
    except ValueError as error:
        raise ValueError(
            "PRODUCTION_MIN_ID doit être un entier."
        ) from error

    print(
        f"PRODUCTION_MIN_ID utilisé : {min_id}"
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
        WHERE id > %s
        ORDER BY id ASC;
    """

    with psycopg.connect(database_url) as connection:
        dataframe = pd.read_sql(
            query,
            connection,
            params=(min_id,),
        )

    if not dataframe.empty:
        print(
            f"IDs récupérés : "
            f"{dataframe['id'].min()} "
            f"→ {dataframe['id'].max()}"
        )
    else:
        print(
            "Aucune prédiction trouvée "
            f"avec id > {min_id}"
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

    reference_columns = set(
        reference_data.columns
    )

    production_columns = set(
        production_data.columns
    )

    missing_in_production = (
        reference_columns
        - production_columns
    )

    unexpected_in_production = (
        production_columns
        - reference_columns
    )

    print(
        "\n=== Validation du schéma ==="
    )

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

        print(
            sorted(
                missing_in_production
            )
        )

    if unexpected_in_production:
        print(
            f"Features supplémentaires en production : "
            f"{len(unexpected_in_production)}"
        )

        print(
            sorted(
                unexpected_in_production
            )
        )

    if (
        not missing_in_production
        and not unexpected_in_production
    ):
        print(
            "✓ Schémas compatibles"
        )


def generate_drift_report(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
) -> None:

    if production_data.empty:
        raise ValueError(
            "Impossible de calculer le drift : "
            "aucune donnée de production disponible."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    production_data = (
        production_data[
            reference_data.columns
        ]
        .copy()
    )

    reference_data = (
        reference_data.copy()
    )

    numerical_columns = (
        reference_data
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column
        in reference_data.columns
        if column
        not in numerical_columns
    ]

    print(
        "\n=== Types pour Evidently ==="
    )

    print(
        f"Variables numériques : "
        f"{len(numerical_columns)}"
    )

    print(
        f"Variables catégorielles : "
        f"{len(categorical_columns)}"
    )

    for column in numerical_columns:

        reference_data[column] = (
            pd.to_numeric(
                reference_data[column],
                errors="coerce",
            )
        )

        production_data[column] = (
            pd.to_numeric(
                production_data[column],
                errors="coerce",
            )
        )

    for column in categorical_columns:

        reference_data[column] = (
            reference_data[column]
            .astype("string")
        )

        production_data[column] = (
            production_data[column]
            .astype("string")
        )

    report = Report([
        DataDriftPreset(
            columns=(
                reference_data
                .columns
                .tolist()
            ),
        )
    ])

    result = report.run(
        current_data=production_data,
        reference_data=reference_data,
    )

    result.save_html(
        str(
            DRIFT_REPORT_PATH
        )
    )

    print(
        f"\n✓ Rapport de drift généré : "
        f"{DRIFT_REPORT_PATH}"
    )


def main() -> None:

    logs = load_production_data()

    production_features = (
        extract_features(
            logs
        )
    )

    print(
        f"Nombre de prédictions production : "
        f"{len(production_features)}"
    )

    reference_data = (
        load_reference_data()
    )

    print(
        f"Nombre de lignes référence : "
        f"{len(reference_data)}"
    )

    validate_feature_schema(
        reference_data,
        production_features,
    )

    generate_drift_report(
        reference_data,
        production_features,
    )


if __name__ == "__main__":
    main()