import argparse
import json
import os
from pathlib import Path

import pandas as pd
import psycopg


METRICS_DIR = Path("reports/metrics")


def export_metrics(
    metrics: dict,
    output_filename: str,
) -> Path:

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        METRICS_DIR
        / output_filename
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"\n✓ Métriques exportées : "
        f"{output_path}"
    )

    return output_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calcul des métriques de monitoring "
            "sur les prédictions PostgreSQL."
        )
    )

    parser.add_argument(
        "--min-id",
        type=int,
        default=0,
        help="ID minimum exclu.",
    )

    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        help="ID maximum inclus.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Nom du fichier JSON de sortie. "
            "Exemple : prediction_metrics_nominal.json"
        ),
    )

    return parser.parse_args()


def load_prediction_logs(
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

    print("\n=== Sélection des prédictions ===")
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
        raise ValueError(
            "Aucune prédiction disponible "
            "pour cette plage d'IDs."
        )

    print(
        f"IDs récupérés : "
        f"{dataframe['id'].min()} "
        f"→ {dataframe['id'].max()}"
    )

    return dataframe


def compute_prediction_metrics(
    logs: pd.DataFrame,
) -> dict:

    total_predictions = len(logs)

    default_count = int(
        (logs["prediction"] == 1).sum()
    )

    accordable_count = int(
        (logs["prediction"] == 0).sum()
    )

    default_rate = (
        default_count
        / total_predictions
    )

    return {
        "total_predictions": total_predictions,
        "credit_accordable_count": accordable_count,
        "default_risk_count": default_count,
        "default_risk_rate": default_rate,
        "mean_default_probability": float(
            logs["default_probability"].mean()
        ),
        "median_default_probability": float(
            logs["default_probability"].median()
        ),
        "mean_latency_ms": float(
            logs["latency_ms"].mean()
        ),
        "p95_latency_ms": float(
            logs["latency_ms"].quantile(0.95)
        ),
        "max_latency_ms": float(
            logs["latency_ms"].max()
        ),
    }


def display_metrics(
    metrics: dict,
) -> None:

    print("\n=== Monitoring des prédictions ===")

    print(
        f"Nombre total de prédictions : "
        f"{metrics['total_predictions']}"
    )

    print(
        f"Crédits accordables : "
        f"{metrics['credit_accordable_count']}"
    )

    print(
        f"Risques de défaut : "
        f"{metrics['default_risk_count']}"
    )

    print(
        f"Taux de risque de défaut : "
        f"{metrics['default_risk_rate']:.2%}"
    )

    print(
        f"Probabilité moyenne de défaut : "
        f"{metrics['mean_default_probability']:.4f}"
    )

    print(
        f"Probabilité médiane de défaut : "
        f"{metrics['median_default_probability']:.4f}"
    )

    print(
        f"Latence moyenne : "
        f"{metrics['mean_latency_ms']:.2f} ms"
    )

    print(
        f"Latence p95 : "
        f"{metrics['p95_latency_ms']:.2f} ms"
    )

    print(
        f"Latence maximale : "
        f"{metrics['max_latency_ms']:.2f} ms"
    )


def main() -> None:

    args = parse_arguments()

    logs = load_prediction_logs(
        min_id=args.min_id,
        max_id=args.max_id,
    )

    metrics = compute_prediction_metrics(
        logs
    )

    display_metrics(
        metrics
    )

    if args.output:
        export_metrics(
            metrics,
            args.output,
        )


if __name__ == "__main__":
    main()