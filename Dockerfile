# ---- Base image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /code

# ---- Install dependencies ----
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# ---- Copy application code (includes pre-trained model.joblib) ----
COPY app /code/app

# ---- Expose port ----
EXPOSE 8000

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---- Run the API ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
