"""
Train a simple Iris classifier and save it to disk.
Run: python train_model.py
Produces: app/model.joblib
"""
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X, y = data.data, data.target
target_names = data.target_names.tolist()
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}")

# Save model + metadata
os.makedirs("app", exist_ok=True)
joblib.dump(
    {
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
    },
    "app/model.joblib",
)
print("Model saved to app/model.joblib")
