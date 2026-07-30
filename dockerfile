FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie du code et des artefacts nécessaires
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Documentation OpenAPI / FastAPI exposée sur le port 8000
EXPOSE 8000

# Lancement de l'API
CMD [
    "python",
    "-m",
    "uvicorn",
    "src.api.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]