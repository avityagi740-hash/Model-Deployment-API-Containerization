from pydantic import BaseModel, Field
from typing import List


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    probabilities: dict


class BatchIrisFeatures(BaseModel):
    instances: List[IrisFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
