FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by XGBoost and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model artifacts into container
COPY src/ ./src/
COPY artifacts/preprocessor.pkl ./artifacts/preprocessor.pkl
COPY artifacts/churn_model.pkl ./artifacts/churn_model.pkl
COPY artifacts/review_classifier.pt ./artifacts/review_classifier.pt
COPY artifacts/vocab.json ./artifacts/vocab.json
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]