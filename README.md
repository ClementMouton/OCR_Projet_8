<div align="center">

# Home Credit Default Risk
# Production API, Monitoring & MLOps

**Déployez et monitorez un modèle de scoring crédit**

Projet réalisé dans le cadre de la formation **Data Scientist – OpenClassrooms**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![Evidently](https://img.shields.io/badge/Evidently-0.7.21-orange)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)

</div>
---

## Quick Start

```bash
git clone ...

cd OCR_Projet_8

python -m venv .venv

pip install -r requirements.txt

uvicorn src.api.app:app --reload
```

---

# Sommaire

- [Présentation](#-présentation)
- [Architecture du projet](#-architecture-du-projet)
- [Structure du dépôt](#-structure-du-dépôt)
- [Installation](#-installation)
- [Lancement du projet](#-lancement-du-projet)
- [Déploiement Render](#-déploiement-render)
- [Base PostgreSQL](#-base-postgresql)
- [Monitoring](#-monitoring)
- [Reproduire les expériences](#-reproduire-les-expériences)
- [Résultats](#-résultats)
- [Améliorations possibles](#-améliorations-possibles)

---

# Présentation

## Contexte

Ce projet a été réalisé dans le cadre du parcours **Data Scientist** d'OpenClassrooms.

L'objectif est de mettre en production un modèle de **scoring crédit** capable d'estimer le risque de défaut d'un client à partir de ses caractéristiques.

Contrairement à un projet de Machine Learning classique, ce projet couvre également les aspects **MLOps** :

- déploiement d'une API de prédiction ;
- conteneurisation Docker ;
- intégration continue ;
- stockage des prédictions ;
- monitoring des données ;
- monitoring des performances du modèle.

---

## Fonctionnalités

- API REST FastAPI
- Pipeline Scikit-Learn
- Déploiement Docker
- Déploiement Render
- Journalisation PostgreSQL
- Monitoring Evidently
- Calcul automatique des métriques
- Génération de rapports HTML
- Exports JSON
- Intégration Continue GitHub Actions

---

## Objectifs

Le projet répond aux objectifs suivants :

- entraîner un pipeline de Machine Learning reproductible ;
- exposer le modèle via une API FastAPI ;
- déployer l'application sur Render ;
- enregistrer les prédictions dans PostgreSQL ;
- détecter automatiquement une dérive des données (Data Drift) ;
- suivre les principales métriques de production.

---

## Technologies utilisées

| Domaine | Technologies |
|----------|--------------|
| Machine Learning | Scikit-Learn |
| API | FastAPI |
| Conteneurisation | Docker |
| Déploiement | Render |
| Base de données | PostgreSQL |
| Monitoring | Evidently |
| Tests | Pytest |
| CI/CD | GitHub Actions |

---

# Architecture du projet

Le pipeline complet est présenté ci-dessous.

```mermaid
flowchart TD

A[Notebook d'entraînement]

A --> B[Pipeline Scikit-Learn]

B --> C[Artefacts du modèle]

C --> D[FastAPI]

D --> E[Docker]

E --> F[Render]

F --> G[API de prédiction]

G --> H[(PostgreSQL)]

H --> I[Prediction Logs]

I --> J[Evidently]

I --> K[Prediction Metrics]

J --> L[Rapports HTML]

K --> M[Exports JSON]
```

Le fonctionnement global est le suivant :

1. le pipeline est entraîné à partir des données Home Credit ;
2. les artefacts sont exportés (`joblib`, schéma des variables, données de référence...) ;
3. l'API FastAPI charge le pipeline et expose un endpoint `/predict` ;
4. chaque prédiction est enregistrée dans PostgreSQL ;
5. les données de production sont comparées aux données de référence afin de détecter une éventuelle dérive ;
6. des métriques de monitoring sont calculées automatiquement afin de suivre le comportement du modèle en production.

# Structure du dépôt

Le projet est organisé de manière à séparer les différentes briques fonctionnelles : entraînement du modèle, API, monitoring, scripts utilitaires et tests.

```text
OCR_Projet_8
│
├── artifacts/                  # Artefacts générés lors de l'entraînement
│   ├── model_pipeline.joblib
│   ├── feature_schema.json
│   ├── model_config.json
│   ├── reference_metrics.json
│   ├── reference_data.parquet
│   ├── valid_client.json
│   └── expected_prediction.json
│
├── data/
│   └── reference/
│       └── reference_data.parquet
│
├── logs/                       # Journaux locaux des prédictions
│
├── reports/
│   ├── drift_report_nominal.html
│   ├── drift_report_simulated.html
│   └── metrics/
│       ├── prediction_metrics_nominal.json
│       └── prediction_metrics_simulated.json
│
├── scripts/
│   ├── generate_nominal_data.py
│   └── generate_drift_data.py
│
├── src/
│   ├── api/
│   │   ├── app.py
│   │   ├── prediction_logger.py
│   │   └── ...
│   │
│   ├── monitoring/
│   │   ├── database.py
│   │   ├── drift_monitor.py
│   │   └── prediction_metrics.py
│   │
│   └── ...
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

---

## Description des principaux dossiers

### `artifacts/`

Contient l'ensemble des artefacts produits lors de l'entraînement du modèle :

- pipeline Scikit-Learn sérialisé ;
- schéma des variables ;
- configuration du modèle ;
- données de référence utilisées pour le monitoring ;
- exemples de prédictions permettant de valider l'API.

Ces fichiers sont chargés automatiquement par l'API au démarrage.

---

### `data/`

Contient les données utilisées par le projet.

Dans la version finale, seules les données nécessaires au monitoring (jeu de référence) sont conservées.

---

### `src/api/`

Contient toute la logique de l'API FastAPI :

- chargement du pipeline ;
- validation des entrées ;
- endpoint `/predict` ;
- endpoint `/health` ;
- endpoint `/model_info` ;
- journalisation des prédictions.

---

### `src/monitoring/`

Contient les différents scripts de monitoring :

- connexion PostgreSQL ;
- calcul des métriques de production ;
- génération des rapports de Data Drift avec Evidently.

Le monitoring est totalement indépendant de l'API et peut être exécuté à tout moment.

---

### `scripts/`

Scripts permettant de générer des jeux de données de test.

Deux scénarios sont disponibles :

- **Nominal** : données proches de la distribution d'entraînement ;
- **Drift simulé** : modification volontaire de nombreuses variables afin de provoquer une dérive détectable.

Ces scripts permettent de reproduire facilement les expériences présentées dans ce projet.

---

### `reports/`

Contient les résultats du monitoring.

Deux types de rapports sont générés :

- rapports HTML Evidently ;
- métriques JSON.

Cette séparation facilite l'automatisation et l'intégration dans un pipeline de monitoring.

---

### `tests/`

Contient les tests unitaires du projet.

Les tests sont exécutés automatiquement par GitHub Actions à chaque push sur le dépôt.

---

### Dockerfile

Permet de construire une image Docker contenant l'ensemble de l'application.

Cette image est utilisée pour le déploiement sur Render.

---

### GitHub Actions

Le workflow CI vérifie automatiquement :

- l'installation des dépendances ;
- l'exécution des tests ;
- la validité du projet avant déploiement.

# Installation

## Prérequis

Avant de commencer, assurez-vous de disposer des outils suivants :

| Outil | Version recommandée |
|--------|---------------------|
| Python | 3.11.x |
| Git | dernière version |
| Docker Desktop | dernière version |
| PostgreSQL | 16+ (ou base Render) |

---

## Cloner le dépôt

```bash
git clone https://github.com/ClementMouton/OCR_Projet_8.git

cd OCR_Projet_8
```

---

## Créer un environnement virtuel

### Windows (PowerShell)

```powershell
python -m venv .venv

.\.venv\Scripts\activate
```

### Windows (Git Bash)

```bash
python -m venv .venv

source .venv/Scripts/activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## Installer les dépendances

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

## Variables d'environnement

Le projet utilise une variable d'environnement permettant de se connecter à PostgreSQL.

Créer un fichier `.env` à la racine du projet :

```env
DATABASE_URL=postgresql://user:password@host:5432/database
```

Ou définir directement la variable dans le terminal.

### Git Bash

```bash
export DATABASE_URL="postgresql://user:password@host:5432/database"
```

### PowerShell

```powershell
$env:DATABASE_URL="postgresql://user:password@host:5432/database"
```

---

## Vérification de l'installation

Lancer les tests :

```bash
pytest
```

Tous les tests doivent être validés.

---

```mermaid
flowchart LR

A[Clone Git]

--> B[Installation dépendances]

--> C[Docker ou Uvicorn]

--> D[FastAPI]

--> E[PostgreSQL]

--> F[Monitoring]
```

---

# Lancement du projet

## Lancement local de l'API

Depuis la racine du projet :

```bash
uvicorn src.api.app:app --reload
```

L'API est alors accessible à l'adresse :

```
http://127.0.0.1:8000
```

---

## Documentation interactive

FastAPI génère automatiquement une documentation OpenAPI.

Swagger UI :

```
http://127.0.0.1:8000/docs
```

Documentation ReDoc :

```
http://127.0.0.1:8000/redoc
```

---

## Endpoints disponibles

| Endpoint | Description |
|-----------|-------------|
| `/health` | Vérifie que l'API fonctionne correctement |
| `/model_info` | Informations sur le modèle chargé |
| `/predict` | Effectue une prédiction de risque de défaut |

---

## Test rapide

Une fois l'API démarrée, il est possible de vérifier son bon fonctionnement avec :

```bash
curl http://127.0.0.1:8000/health
```

Réponse attendue :

```json
{
    "status": "healthy"
}
```

---

## Exécution avec Docker

Construire l'image :

```bash
docker build -t home-credit-api .
```

Lancer le conteneur :

```bash
docker run -p 8000:8000 \
    -e DATABASE_URL="<DATABASE_URL>" \
    home-credit-api
```

L'application est ensuite disponible sur :

```
http://localhost:8000
```

## Environnement dédié au monitoring

```bash
python -m venv .venv-monitoring

source .venv-monitoring/Scripts/activate

pip install -r requirements.txt
```

# Déploiement Render

L'API est déployée sur **Render** à l'aide du Dockerfile présent à la racine du projet.

Le déploiement est automatiquement déclenché après validation de la pipeline CI GitHub Actions.

## Configuration

Créer un nouveau **Web Service** sur Render :

- Environment : Docker
- Branch : `main`
- Root Directory : `/`
- Dockerfile Path : `Dockerfile`

Variables d'environnement :

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Chaîne de connexion PostgreSQL |

---

## Vérification du déploiement

Une fois le service lancé, plusieurs endpoints permettent de vérifier son bon fonctionnement.

### Vérification de l'API

```text
GET /health
```

Réponse attendue :

```json
{
    "status": "healthy"
}
```

---

### Informations sur le modèle

```text
GET /model_info
```

Cet endpoint retourne notamment :

- nom du modèle ;
- seuil de décision ;
- version ;
- caractéristiques principales.

---

### Prédiction

```text
POST /predict
```

Chaque appel :

- réalise une prédiction ;
- retourne le score de défaut ;
- enregistre automatiquement la prédiction dans PostgreSQL.

# Base PostgreSQL

Toutes les prédictions réalisées par l'API sont enregistrées dans PostgreSQL afin de permettre leur analyse a posteriori.

Cette approche constitue la base du monitoring en production.

```mermaid
flowchart LR

A[API FastAPI]

--> B[Prediction Logger]

--> C[(PostgreSQL)]

C --> D[Monitoring]
```

---

## Table `prediction_logs`

Chaque prédiction enregistrée contient notamment :

| Champ | Description |
|--------|-------------|
| id | Identifiant |
| timestamp | Date de la prédiction |
| features | Variables d'entrée |
| default_probability | Probabilité de défaut |
| prediction | Classe prédite |
| decision_threshold | Seuil utilisé |
| decision_label | Décision finale |
| latency_ms | Temps de réponse |

Cette table est utilisée comme source unique pour l'ensemble des scripts de monitoring.

# Monitoring

Le monitoring est indépendant de l'API.

Il repose sur deux briques complémentaires :

- le **Data Drift**, permettant de comparer les données de production aux données d'entraînement ;
- les **Prediction Metrics**, permettant de suivre le comportement du modèle.

---

## Data Drift

Le Data Drift est calculé grâce à **Evidently**.

Les données de production sont comparées au jeu de référence exporté lors de l'entraînement.

Deux scénarios sont disponibles.

### Jeu nominal

```bash
python -m src.monitoring.drift_monitor \
    --min-id 0 \
    --max-id 501 \
    --output drift_report_nominal.html
```

---

### Jeu avec dérive simulée

```bash
python -m src.monitoring.drift_monitor \
    --min-id 1000 \
    --max-id 1501 \
    --output drift_report_simulated.html
```

Les rapports HTML sont générés dans :

```text
reports/
```

---

## Monitoring des prédictions

Les métriques sont calculées directement à partir de PostgreSQL.

### Jeu nominal

```bash
python -m src.monitoring.prediction_metrics \
    --min-id 0 \
    --max-id 501 \
    --output prediction_metrics_nominal.json
```

---

### Jeu avec dérive simulée

```bash
python -m src.monitoring.prediction_metrics \
    --min-id 1000 \
    --max-id 1501 \
    --output prediction_metrics_simulated.json
```

Les exports JSON sont générés dans :

```text
reports/metrics/
```

Les métriques calculées comprennent notamment :

- nombre total de prédictions ;
- taux de risque de défaut ;
- probabilité moyenne de défaut ;
- probabilité médiane ;
- latence moyenne ;
- latence P95 ;
- latence maximale.

# Résultats

Deux scénarios ont été réalisés afin de valider le fonctionnement du monitoring.

## Comparaison

| Scénario | Data Drift | Features en drift | Taux de défaut prédit |
|-----------|-----------:|------------------:|----------------------:|
| Nominal | ❌ Non détecté | 16,25 % | 26,95 % |
| Drift simulé | ✅ Détecté | 58,47 % | 15,57 % |

---

## Analyse

Le premier scénario reproduit une situation proche des données d'entraînement.

Le monitoring détecte uniquement **16,25 %** de variables en dérive, ce qui reste inférieur au seuil configuré par Evidently.

Le second scénario modifie volontairement une partie importante des variables.

Cette modification entraîne :

- une dérive détectée sur **58,47 %** des variables ;
- une évolution significative des prédictions du modèle ;
- la génération automatique d'un rapport HTML de Data Drift.

Ces deux expériences montrent que le pipeline de monitoring est capable :

- d'identifier une dérive des données d'entrée ;
- de mesurer son impact sur les prédictions ;
- de produire automatiquement des rapports exploitables.

---

# Améliorations possibles

...

---

# Auteur

Clément Mouton

Projet réalisé dans le cadre de la formation Data Scientist OpenClassrooms.

GitHub : https://github.com/ClementMouton/
LinkedIn : https://www.linkedin.com/in/clement-mouton/