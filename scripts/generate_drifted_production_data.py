import os
import time

import numpy as np
import pandas as pd
import requests


REFERENCE_DATA_PATH = "data/reference/reference_data.parquet"

API_URL = os.getenv(
    "PREDICTION_API_URL",
    "https://ocr-projet-8.onrender.com/predict",
)

N_SAMPLES = 500
N_DRIFTED_COLUMNS = 250
RANDOM_STATE = 123


def create_drifted_sample(
    reference_data: pd.DataFrame,
) -> pd.DataFrame:

    sample = reference_data.sample(
        n=min(N_SAMPLES, len(reference_data)),
        random_state=RANDOM_STATE,
    ).copy()

    numerical_columns = (
        reference_data
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    # On évite les variables quasi binaires / constantes.
    candidate_columns = []

    for column in numerical_columns:
        unique_count = reference_data[column].nunique(
            dropna=True
        )

        std = reference_data[column].std()

        if (
            unique_count > 10
            and pd.notna(std)
            and std > 0
        ):
            candidate_columns.append(column)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n_columns = min(
        N_DRIFTED_COLUMNS,
        len(candidate_columns),
    )

    drifted_columns = rng.choice(
        candidate_columns,
        size=n_columns,
        replace=False,
    )

    # Décalage volontaire de distribution :
    # + 2 écarts-types pour les variables choisies.
    for column in drifted_columns:
        std = reference_data[column].std()

        sample[column] = (
            sample[column] + 2 * std
        )

    print(
        f"Variables volontairement dérivées : "
        f"{len(drifted_columns)}"
    )

    return sample


def prepare_payload(
    row: pd.Series,
) -> dict:

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

    sample = create_drifted_sample(
        reference_data
    )

    print(
        f"Envoi de {len(sample)} clients "
        f"dérivés vers {API_URL}"
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

    print("\n=== Simulation drift terminée ===")
    print(f"Succès : {success_count}")
    print(f"Erreurs : {error_count}")
    print(
        f"Durée : {duration:.1f} secondes"
    )


if __name__ == "__main__":
    main()