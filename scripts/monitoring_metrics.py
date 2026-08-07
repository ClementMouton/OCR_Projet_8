import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "La variable d'environnement DATABASE_URL n'est pas définie."
    )

engine = create_engine(DATABASE_URL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcul des métriques de monitoring du modèle."
    )

    parser.add_argument(
        "--min-id",
        type=int,
        default=None,
        help="ID minimum des prédictions à analyser."
    )

    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        help="ID maximum des prédictions à analyser."
    )

    return parser.parse_args()


def load_prediction_logs(min_id=None, max_id=None):
    conditions = []
    params = {}

    if min_id is not None:
        conditions.append("id >= :min_id")
        params["min_id"] = min_id

    if max_id is not None:
        conditions.append("id <= :max_id")
        params["max_id"] = max_id

    where_clause = ""

    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = text(f"""
        SELECT
            id,
            timestamp,
            default_probability,
            prediction,
            decision_threshold,
            decision_label,
            latency_ms
        FROM prediction_logs
        {where_clause}
        ORDER BY id
    """)

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params=params
        )

    return df


def calculate_metrics(df):
    if df.empty:
        raise ValueError(
            "Aucune prédiction trouvée pour la plage sélectionnée."
        )

    probabilities = df["default_probability"].dropna()
    latencies = df["latency_ms"].dropna()

    total_predictions = len(df)

    accepted_count = int(
        (df["prediction"] == 0).sum()
    )

    refused_count = int(
        (df["prediction"] == 1).sum()
    )

    acceptance_rate = (
        accepted_count / total_predictions
        if total_predictions > 0
        else 0
    )

    refusal_rate = (
        refused_count / total_predictions
        if total_predictions > 0
        else 0
    )

    metrics = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "production_period": {
            "first_id": int(df["id"].min()),
            "last_id": int(df["id"].max()),

            "first_prediction": (
                df["timestamp"].min().isoformat()
                if not df["timestamp"].isna().all()
                else None
            ),

            "last_prediction": (
                df["timestamp"].max().isoformat()
                if not df["timestamp"].isna().all()
                else None
            ),
        },

        "predictions": {
            "total_predictions": total_predictions,
            "credit_accordable_count": accepted_count,
            "credit_refused_count": refused_count,
            "acceptance_rate": round(
                acceptance_rate,
                4
            ),
            "refusal_rate": round(
                refusal_rate,
                4
            ),
        },

        "default_probability": {
            "mean": round(
                float(probabilities.mean()),
                4
            ),
            "median": round(
                float(probabilities.median()),
                4
            ),
            "std": round(
                float(probabilities.std()),
                4
            ),
            "min": round(
                float(probabilities.min()),
                4
            ),
            "max": round(
                float(probabilities.max()),
                4
            ),
        },

        "latency_ms": {
            "mean": round(
                float(latencies.mean()),
                2
            ),
            "median": round(
                float(latencies.median()),
                2
            ),
            "p95": round(
                float(np.percentile(latencies, 95)),
                2
            ),
            "max": round(
                float(latencies.max()),
                2
            ),
        },
    }

    return metrics


def print_metrics(metrics):
    predictions = metrics["predictions"]
    probabilities = metrics["default_probability"]
    latency = metrics["latency_ms"]
    period = metrics["production_period"]

    print("\n" + "=" * 55)
    print("              PRODUCTION MONITORING")
    print("=" * 55)

    print("\nDonnées analysées")
    print("-" * 55)
    print(
        f"IDs                    : "
        f"{period['first_id']} -> {period['last_id']}"
    )
    print(
        f"Première prédiction    : "
        f"{period['first_prediction']}"
    )
    print(
        f"Dernière prédiction    : "
        f"{period['last_prediction']}"
    )

    print("\nPrédictions")
    print("-" * 55)
    print(
        f"Total                  : "
        f"{predictions['total_predictions']}"
    )
    print(
        f"Crédits accordables    : "
        f"{predictions['credit_accordable_count']}"
    )
    print(
        f"Crédits refusés        : "
        f"{predictions['credit_refused_count']}"
    )
    print(
        f"Taux d'acceptation     : "
        f"{predictions['acceptance_rate'] * 100:.2f} %"
    )
    print(
        f"Taux de refus          : "
        f"{predictions['refusal_rate'] * 100:.2f} %"
    )

    print("\nProbabilité de défaut")
    print("-" * 55)
    print(
        f"Moyenne                : "
        f"{probabilities['mean']:.4f}"
    )
    print(
        f"Médiane                : "
        f"{probabilities['median']:.4f}"
    )
    print(
        f"Écart-type             : "
        f"{probabilities['std']:.4f}"
    )
    print(
        f"Minimum                : "
        f"{probabilities['min']:.4f}"
    )
    print(
        f"Maximum                : "
        f"{probabilities['max']:.4f}"
    )

    print("\nLatence")
    print("-" * 55)
    print(
        f"Moyenne                : "
        f"{latency['mean']:.2f} ms"
    )
    print(
        f"Médiane                : "
        f"{latency['median']:.2f} ms"
    )
    print(
        f"P95                    : "
        f"{latency['p95']:.2f} ms"
    )
    print(
        f"Maximum                : "
        f"{latency['max']:.2f} ms"
    )

    print("\n" + "=" * 55)


def export_metrics(metrics):
    output_dir = Path("reports") / "metrics"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    period = metrics["production_period"]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = output_dir / (
        f"monitoring_metrics_"
        f"{period['first_id']}_"
        f"{period['last_id']}_"
        f"{timestamp}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    return output_path


def main():
    args = parse_args()

    df = load_prediction_logs(
        min_id=args.min_id,
        max_id=args.max_id
    )

    metrics = calculate_metrics(df)

    print_metrics(metrics)

    output_path = export_metrics(metrics)

    print(
        f"\nMétriques exportées vers : "
        f"{output_path}\n"
    )


if __name__ == "__main__":
    main()