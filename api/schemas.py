from typing import List, Dict
from pydantic import BaseModel, Field

class SingleSampleRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=16,
        max_length=16,
        description="16-channel EEG reading [X1, X2, ..., X16]",
        example=[4, 7, 18, 25, 28, 27, 20, 10, -10, -18, -20, -16, 13, 32, 12, 10]
    )

class BatchSampleRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of 16-channel EEG sample vectors",
        example=[
            [4, 7, 18, 25, 28, 27, 20, 10, -10, -18, -20, -16, 13, 32, 12, 10],
            [87, 114, 120, 106, 76, 54, 28, 5, -19, -49, -85, -102, -100, -89, -61, -21]
        ]
    )

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    total_samples: int
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

