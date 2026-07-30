import json
import joblib
import pandas as pd

with open(
    "tests/fixtures/valid_client.json",
    "r",
    encoding="utf-8"
) as f:
    valid_client = json.load(f)

with open(
    "tests/fixtures/expected_prediction.json",
    "r",
    encoding="utf-8"
) as f:
    expected = json.load(f)

model = joblib.load(
    "artifacts/model_pipeline.joblib"
)

X = pd.DataFrame([valid_client])
import numpy as np

X = X.replace({None: np.nan})

probability = float(
    model.predict_proba(X)[0, 1]
)

print("Prédiction directe :", probability)
print(
    "Prédiction attendue :",
    expected["default_probability"]
)
print(
    "Écart :",
    abs(
        probability
        - expected["default_probability"]
    )
)

