import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg

from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_DATA_PATH = "data/reference/reference_data.parquet"
REPORT_DIR = Path("reports")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Génération d'un rapport de data drift "
            "à partir des prédictions PostgreSQL."
        )
    )

    parser.add_argument(
        "--min-id",
        type=int,
        default=0,
        help=(
            "ID minimum exclu. "
            "Exemple : --min-id 1000 sélectionne id > 1000."
        ),
    )

    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        help=(
            "ID maximum inclus. "
            "Exemple : --max-id 1501 sélectionne id <= 1501."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="drift_report.html",
        help="Nom du rapport HTML généré.",
    )

    return parser.parse_args()


def load_production_data(
    min_id: int = 0,
    max_id: int | None = None,
) -> pd.DataFrame:

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "La variable d'environnement DATABASE_URL "
            "n'est pas définie."
        )

    params = [min_id]

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
    """

    if max_id is not None:
        query += " AND id <= %s"
        params.append(max_id)

    query += " ORDER BY id ASC;"

    print("\n=== Sélection production ===")
    print(f"ID minimum exclu : {min_id}")

    if max_id is not None:
        print(f"ID maximum inclus : {max_id}")
    else:
        print("ID maximum : aucun")

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                tuple(params),
            )

            rows = cursor.fetchall()

            columns = [
                description.name
                for description in cursor.description
            ]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    if dataframe.empty:
        print(
            "Aucune prédiction trouvée "
            "pour cette plage d'IDs."
        )

    else:
        print(
            f"IDs récupérés : "
            f"{dataframe['id'].min()} "
            f"→ {dataframe['id'].max()}"
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
    output_path: Path,
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
        for column in reference_data.columns
        if column not in numerical_columns
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
        str(output_path)
    )

    print(
        f"\n✓ Rapport de drift généré : "
        f"{output_path}"
    )


def main():

    args = parse_arguments()

    output_path = (
        REPORT_DIR
        / args.output
    )

    logs = load_production_data(
        min_id=args.min_id,
        max_id=args.max_id,
    )

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
        output_path,
    )


if __name__ == "__main__":
    main()