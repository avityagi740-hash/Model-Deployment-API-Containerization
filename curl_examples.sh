#!/bin/bash
# Example curl requests for the Iris Classifier API
# Usage: bash examples/curl_examples.sh
# Assumes the API is running at http://localhost:8000

BASE_URL="http://localhost:8000"

echo "=== 1. Root endpoint ==="
curl -s "$BASE_URL/" | python3 -m json.tool
echo

echo "=== 2. Health check ==="
curl -s "$BASE_URL/health" | python3 -m json.tool
echo

echo "=== 3. Single prediction ==="
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d @examples/request_single.json | python3 -m json.tool
echo

echo "=== 4. Batch prediction ==="
curl -s -X POST "$BASE_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d @examples/request_batch.json | python3 -m json.tool
echo
