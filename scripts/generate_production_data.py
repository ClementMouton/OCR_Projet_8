import os
import time

import pandas as pd
import requests


REFERENCE_DATA_PATH = "data/reference/reference_data.parquet"

API_URL = os.getenv(
    "PREDICTION_API_URL",
    "https://ocr-projet-8.onrender.com/predict",
)

N_SAMPLES = 500
RANDOM_STATE = 42


def prepare_payload(row: pd.Series) -> dict:
    features = (
        row.astype(object)
        .where(pd.notnull(row), None)
        .to_dict()
    )

    return {
        "features": features
    }


def main() -> None:
    reference_data = pd.read_parquet(
        REFERENCE_DATA_PATH
    )

    sample = reference_data.sample(
        n=min(N_SAMPLES, len(reference_data)),
        random_state=RANDOM_STATE,
    )

    print(
        f"Envoi de {len(sample)} prédictions "
        f"vers {API_URL}"
    )

    success_count = 0
    error_count = 0

    start_time = time.time()

    for index, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):
        payload = prepare_payload(row)

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1
                print(
                    f"[{index}] Erreur "
                    f"{response.status_code}: "
                    f"{response.text[:200]}"
                )

        except requests.RequestException as error:
            error_count += 1
            print(
                f"[{index}] Erreur réseau : "
                f"{error}"
            )

        if index % 50 == 0:
            print(
                f"{index}/{len(sample)} "
                f"- succès : {success_count} "
                f"- erreurs : {error_count}"
            )

    duration = time.time() - start_time

    print("\n=== Simulation terminée ===")
    print(f"Succès : {success_count}")
    print(f"Erreurs : {error_count}")
    print(f"Durée : {duration:.1f} secondes")


if __name__ == "__main__":
    main()