# OCR Projet 6 — Credit Scoring MLOps avec MLflow

## Contexte métier

Ce projet s’inscrit dans le cadre de la mission **"Prêt à Dépenser"**, une société financière spécialisée dans le crédit à la consommation pour des clients disposant de peu ou pas d’historique bancaire.

L’objectif est de construire un **modèle de scoring crédit** capable de prédire automatiquement la probabilité de défaut d’un client afin d’aider à la décision d’octroi de crédit.

Le projet intègre également une dimension **MLOps**, avec le suivi du cycle de vie complet du modèle via MLflow.

---

## Objectifs du projet

- Construire un modèle de classification binaire de risque de défaut
- Gérer le déséquilibre des classes
- Définir une métrique métier personnalisée
- Optimiser les hyperparamètres du modèle
- Optimiser le seuil de décision métier
- Suivre les expérimentations avec MLflow
- Enregistrer le modèle dans un Model Registry
- Tester le serving du modèle
- Expliquer les prédictions globalement et localement avec SHAP

---

## Dataset

Dataset utilisé :

**Home Credit Default Risk**

Source :
https://www.kaggle.com/c/home-credit-default-risk/data

Les fichiers doivent être téléchargés manuellement puis placés dans :

```text
src/
```

Fichiers attendus :

```text
src/
├── application_train.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── installments_payments.csv
├── POS_CASH_balance.csv
└── credit_card_balance.csv
```

---

## Stack technique

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Optuna
- SHAP
- Matplotlib
- Seaborn

---

## Installation

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activation Windows :

```bash
.venv\Scripts\activate
```

Installation des dépendances :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow optuna shap requests
```

---

## Structure du projet

```text
OCR_Projet_6/
│
├── data_pipeline.ipynb
├── README.md
├── .gitignore
│
├── src/
│   └── datasets Kaggle
│
├── artifacts/
│   └── exports temporaires
│
├── mlruns/
│   └── tracking MLflow
│
└── mlartifacts/
    └── modèles enregistrés
```

---

## Exécution du projet

### 1. Lancer MLflow UI

Dans un terminal :

```bash
python -m mlflow ui
```

Interface accessible ici :

```text
http://127.0.0.1:5000
```

---

### 2. Exécuter le notebook

Ouvrir Jupyter :

```bash
jupyter notebook
```

Puis lancer :

```text
data_pipeline.ipynb
```

Le notebook couvre :

- chargement des données
- feature engineering
- fusion des tables
- preprocessing
- modèles baseline
- optimisation Optuna
- tracking MLflow
- threshold tuning métier
- SHAP explainability

---

## MLflow Model Serving

Après entraînement et enregistrement du modèle.

### Git Bash

```bash
export __pyfunc_model_path__="models:/home_credit_scoring_model/1"
python -m uvicorn --host 127.0.0.1 --port 5001 --workers 1 mlflow.pyfunc.scoring_server.app:app
```

### CMD Windows

```cmd
set __pyfunc_model_path__=models:/home_credit_scoring_model/1
python -m uvicorn --host 127.0.0.1 --port 5001 --workers 1 mlflow.pyfunc.scoring_server.app:app
```

API disponible sur :

```text
http://127.0.0.1:5001
```

---

## Test de l’API

Depuis Python :

```python
import requests
import json
import pandas as pd
import numpy as np

sample = X_test.iloc[:1].copy()
sample = sample.astype(object).where(pd.notnull(sample), None)

payload = {
    "dataframe_split": sample.to_dict(orient="split")
}

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"}
)

print(response.status_code)
print(response.text)
```

Résultat attendu :

```text
200
{"predictions": [0]}
```

---

## Métrique métier

Le projet prend en compte un déséquilibre métier :

- Faux négatif (mauvais client accepté) = coût élevé
- Faux positif (bon client refusé) = coût plus faible

Hypothèse retenue :

```text
FN = 10 × FP
```

Une optimisation du seuil de classification a été réalisée afin de minimiser ce coût métier.

---

## Modèle retenu

Modèle final :

```text
Logistic Regression
```

Optimisation :

```text
Optuna (Bayesian Optimization)
```

Explicabilité :

```text
SHAP
```

---

## Résultats

Pipeline validé avec :

- MLflow experiment tracking
- MLflow UI
- Model Registry
- Model Serving
- SHAP global explainability
- SHAP local explainability

---

## Sécurité / GitHub

Les éléments suivants ne sont pas versionnés :

- datasets
- artefacts temporaires
- modèles générés
- fichiers d’environnement
- logs MLflow

Voir :

```text
.gitignore
```

---

## Auteur

Clément Mouton
Master 2 Data Officer Machine Learning
Crédit Agricole Lorraine
