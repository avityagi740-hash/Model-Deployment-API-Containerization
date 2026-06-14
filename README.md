# Iris Classifier API

A simple, containerized REST API that serves a trained scikit-learn
RandomForest model (Iris flower classification) using **FastAPI** and
**Docker**.

## Project Structure

```
model-api-project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app and endpoints
│   ├── schemas.py        # Pydantic request/response models
│   └── model.joblib       # Pre-trained model artifact
├── examples/
│   ├── request_single.json
│   ├── response_single.json
│   ├── request_batch.json
│   ├── response_batch.json
│   ├── curl_examples.sh
│   └── demo_curl_screenshot.png
├── tests/
│   └── test_api.py
├── train_model.py         # Script to (re)train the model
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## 1. Environment Setup (local, without Docker)

Requires Python 3.11+.

```bash
# Clone the repo
git clone <your-repo-url>
cd model-api-project

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### (Re)train the model

A pre-trained `app/model.joblib` is already included, but you can
regenerate it:

```bash
python train_model.py
```

This trains a `RandomForestClassifier` on the Iris dataset and writes
`app/model.joblib`.

### Run the API locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive docs (Swagger UI): http://localhost:8000/docs
- Alternative docs (ReDoc): http://localhost:8000/redoc

### Run tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## 2. Running with Docker

### Build the image

```bash
docker build -t iris-classifier-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name iris-api iris-classifier-api
```

The API is now available at `http://localhost:8000`.

### Check container health / logs

```bash
docker ps
docker logs iris-api
curl http://localhost:8000/health
```

### Stop / remove the container

```bash
docker stop iris-api
docker rm iris-api
```

## 3. API Endpoints

| Method | Path             | Description                          |
|--------|------------------|---------------------------------------|
| GET    | `/`              | API info / available endpoints        |
| GET    | `/health`        | Health check + model load status      |
| POST   | `/predict`       | Single instance prediction            |
| POST   | `/predict/batch` | Batch prediction for multiple records |

### Input features (Iris dataset)

| Field          | Type  | Description           |
|-----------------|-------|------------------------|
| `sepal_length`  | float | Sepal length in cm     |
| `sepal_width`   | float | Sepal width in cm      |
| `petal_length`  | float | Petal length in cm     |
| `petal_width`   | float | Petal width in cm      |

## 4. Example Requests & Responses

### Health check

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true}
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
```json
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
    ]
  }'
```
```json
{
  "predictions": [
    {
      "predicted_class": "setosa",
      "predicted_class_index": 0,
      "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}
    },
    {
      "predicted_class": "virginica",
      "predicted_class_index": 2,
      "probabilities": {"setosa": 0.0, "versicolor": 0.04, "virginica": 0.96}
    }
  ]
}
```

Ready-to-use payloads are in `examples/`. Run all examples with:

```bash
bash examples/curl_examples.sh
```

## 5. Notes

- The model used is a `RandomForestClassifier` trained on the built-in
  scikit-learn Iris dataset (150 samples, 4 features, 3 classes).
- Swap in your own model by replacing `train_model.py` (or
  `app/model.joblib`) and updating `app/schemas.py` /
  `app/main.py` to match your feature set and output classes.
- The Dockerfile uses `python:3.11-slim` and includes a `HEALTHCHECK`
  that polls `/health`.
