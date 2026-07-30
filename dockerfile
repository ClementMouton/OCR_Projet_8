FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API et du modèle
COPY src/ ./src/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Lancement de FastAPI
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]