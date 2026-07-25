# API de scoring

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
python -m uvicorn src.api.main:app --reload
```

Interfaces disponibles :

- API : `http://127.0.0.1:8000`
- Swagger : `http://127.0.0.1:8000/docs`
- Health check : `http://127.0.0.1:8000/health`
- Informations modèle : `http://127.0.0.1:8000/model-info`

L'API peut démarrer sans modèle. Dans ce cas :

- `/health` fonctionne ;
- `/model-info` fonctionne ;
- `/predict` renvoie une erreur HTTP 503.

Pour activer les prédictions, ajouter :

```text
artifacts/model_pipeline.joblib
artifacts/model_config.json
artifacts/feature_schema.json
```

## Tests

```bash
python -m pytest -q
```
