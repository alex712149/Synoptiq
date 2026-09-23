# SkillBlend API - lightweight container matching the blueprint's
# "separate the UI/API from heavy inference" deployment guardrail
# (Section 19). No GPU dependency; LightGBM/XGBoost training is meant
# to run offline via scripts/seed_and_train.py, and its output artifacts
# are what this image serves.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Seed synthetic data + train models at build time so the image is
# immediately ready to serve (swap this for a scheduled job hitting
# real GFS/IFS/AIFS feeds in production - see app/data/synthetic_generator.py).
RUN python -m scripts.seed_and_train

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
