from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path("notebooks/00_training_pipeline_reference.ipynb")
OUTPUT_PATH = Path("notebooks/data_pipeline_updated.ipynb")

if not NOTEBOOK_PATH.exists():
    raise FileNotFoundError(
        f"Notebook introuvable : {NOTEBOOK_PATH.resolve()}"
    )

nb = nbf.read(NOTEBOOK_PATH, as_version=4)

marker = "# EXPORT OCR PROJET 8"

nb.cells = [
    cell
    for cell in nb.cells
    if marker not in "".join(cell.get("source", ""))
]

nb.cells.append(
    nbf.v4.new_markdown_cell(
        """# Export des artefacts pour OCR Projet 8

Cette section génère les artefacts nécessaires à l'API, aux tests et au monitoring.

Elle utilise directement les variables créées plus haut dans le notebook :

- `best_model_optuna`
- `best_threshold`
- `X_train`
- `X_test`
- `y_test`
- `y_proba`
- `y_pred`
- `FN_COST`
- `FP_COST`
"""
    )
)

export_code = '# EXPORT OCR PROJET 8\n\nfrom pathlib import Path\nimport json\nimport joblib\nimport numpy as np\nimport pandas as pd\n\nfrom sklearn.metrics import (\n    accuracy_score,\n    precision_score,\n    recall_score,\n    f1_score,\n    roc_auc_score,\n    confusion_matrix,\n)\n\nrequired_variables = [\n    "best_model_optuna",\n    "best_threshold",\n    "X_train",\n    "X_test",\n    "y_test",\n    "y_proba",\n    "y_pred",\n    "FN_COST",\n    "FP_COST",\n]\n\nmissing_variables = [\n    variable\n    for variable in required_variables\n    if variable not in globals()\n]\n\nif missing_variables:\n    raise NameError(\n        "Variables manquantes : "\n        + ", ".join(missing_variables)\n        + ". Exécute d\'abord toutes les cellules précédentes."\n    )\n\nPROJECT_DIR = Path.cwd()\nif PROJECT_DIR.name == "notebooks":\n    PROJECT_DIR = PROJECT_DIR.parent\n\nARTIFACTS_DIR = PROJECT_DIR / "artifacts"\nREFERENCE_DIR = PROJECT_DIR / "data" / "reference"\nFIXTURES_DIR = PROJECT_DIR / "tests" / "fixtures"\n\nARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)\nREFERENCE_DIR.mkdir(parents=True, exist_ok=True)\nFIXTURES_DIR.mkdir(parents=True, exist_ok=True)\n\ndecision_threshold = float(best_threshold)\ny_test_array = np.asarray(y_test)\ny_proba_array = np.asarray(y_proba, dtype=float)\n\nif len(y_test_array) != len(y_proba_array):\n    raise ValueError("y_test et y_proba n\'ont pas la même longueur.")\n\ny_pred_export = (\n    y_proba_array >= decision_threshold\n).astype(int)\n\nmodel_path = ARTIFACTS_DIR / "model_pipeline.joblib"\njoblib.dump(best_model_optuna, model_path)\nreloaded_model = joblib.load(model_path)\n\nmodel_config = {\n    "model_name": "home_credit_scoring_model",\n    "model_version": "1.0.0",\n    "model_type": "LogisticRegression",\n    "decision_threshold": decision_threshold,\n    "fn_cost": int(FN_COST),\n    "fp_cost": int(FP_COST),\n    "positive_class": 1,\n    "positive_class_meaning": "client en défaut",\n}\n\nmodel_config_path = ARTIFACTS_DIR / "model_config.json"\nwith model_config_path.open("w", encoding="utf-8") as file:\n    json.dump(model_config, file, indent=2, ensure_ascii=False)\n\nfeature_schema = {\n    "feature_count": int(X_train.shape[1]),\n    "feature_order": X_train.columns.tolist(),\n    "features": [\n        {\n            "name": column,\n            "dtype": str(X_train[column].dtype),\n            "nullable": bool(X_train[column].isna().any()),\n        }\n        for column in X_train.columns\n    ],\n}\n\nfeature_schema_path = ARTIFACTS_DIR / "feature_schema.json"\nwith feature_schema_path.open("w", encoding="utf-8") as file:\n    json.dump(feature_schema, file, indent=2, ensure_ascii=False)\n\ntn, fp, fn, tp = confusion_matrix(\n    y_test_array,\n    y_pred_export,\n    labels=[0, 1]\n).ravel()\n\nbusiness_cost_value = (\n    int(fn) * int(FN_COST)\n    + int(fp) * int(FP_COST)\n)\n\nreference_metrics = {\n    "accuracy": float(accuracy_score(y_test_array, y_pred_export)),\n    "precision": float(\n        precision_score(y_test_array, y_pred_export, zero_division=0)\n    ),\n    "recall": float(\n        recall_score(y_test_array, y_pred_export, zero_division=0)\n    ),\n    "f1_score": float(\n        f1_score(y_test_array, y_pred_export, zero_division=0)\n    ),\n    "roc_auc": float(\n        roc_auc_score(y_test_array, y_proba_array)\n    ),\n    "business_cost": int(business_cost_value),\n    "decision_threshold": decision_threshold,\n    "true_negatives": int(tn),\n    "false_positives": int(fp),\n    "false_negatives": int(fn),\n    "true_positives": int(tp),\n    "test_sample_count": int(len(y_test_array)),\n}\n\nreference_metrics_path = ARTIFACTS_DIR / "reference_metrics.json"\nwith reference_metrics_path.open("w", encoding="utf-8") as file:\n    json.dump(reference_metrics, file, indent=2, ensure_ascii=False)\n\nreference_data = X_train.sample(\n    n=min(10_000, len(X_train)),\n    random_state=42\n).copy()\n\nreference_data_path = REFERENCE_DIR / "reference_data.parquet"\nreference_data.to_parquet(reference_data_path, index=False)\n\nvalid_client_dataframe = X_test.iloc[[0]].copy()\nvalid_client_series = valid_client_dataframe.iloc[0]\n\ndef to_json_compatible(value):\n    if pd.isna(value):\n        return None\n    if isinstance(value, np.generic):\n        return value.item()\n    if isinstance(value, pd.Timestamp):\n        return value.isoformat()\n    return value\n\nvalid_client = {\n    column: to_json_compatible(valid_client_series[column])\n    for column in X_test.columns\n}\n\nvalid_client_path = FIXTURES_DIR / "valid_client.json"\nwith valid_client_path.open("w", encoding="utf-8") as file:\n    json.dump(valid_client, file, indent=2, ensure_ascii=False)\n\ndefault_probability = float(\n    reloaded_model.predict_proba(\n        valid_client_dataframe\n    )[0, 1]\n)\n\nprediction = int(\n    default_probability >= decision_threshold\n)\n\nexpected_prediction = {\n    "default_probability": default_probability,\n    "prediction": prediction,\n    "decision_threshold": decision_threshold,\n}\n\nexpected_prediction_path = (\n    FIXTURES_DIR / "expected_prediction.json"\n)\n\nwith expected_prediction_path.open("w", encoding="utf-8") as file:\n    json.dump(\n        expected_prediction,\n        file,\n        indent=2,\n        ensure_ascii=False\n    )\n\nexpected_files = [\n    model_path,\n    model_config_path,\n    feature_schema_path,\n    reference_metrics_path,\n    reference_data_path,\n    valid_client_path,\n    expected_prediction_path,\n]\n\nmissing_files = [\n    str(path)\n    for path in expected_files\n    if not path.exists()\n]\n\nif missing_files:\n    raise FileNotFoundError(\n        "Certains fichiers n\'ont pas été générés : "\n        + ", ".join(missing_files)\n    )\n\nprint("Export terminé avec succès.\\n")\nfor path in expected_files:\n    print(f"✓ {path.relative_to(PROJECT_DIR)}")\n\nprint("\\nRésumé :")\nprint(f"- Variables : {X_train.shape[1]}")\nprint(f"- Référence : {len(reference_data)} lignes")\nprint(f"- Seuil métier : {decision_threshold:.6f}")\nprint(f"- Probabilité test : {default_probability:.6f}")\nprint(f"- Prédiction test : {prediction}")\nprint(f"- ROC-AUC : {reference_metrics[\'roc_auc\']:.6f}")\nprint(f"- Coût métier : {business_cost_value}")\n'
nb.cells.append(nbf.v4.new_code_cell(export_code))

nbf.write(nb, OUTPUT_PATH)
print(f"Notebook généré : {OUTPUT_PATH.resolve()}")
