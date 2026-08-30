import logging
import os
from pathlib import Path
from typing import Annotated, Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "Student-Risk-Model"
MODEL_ALIAS = "champion"
MLFLOW_ARTIFACT_ROOT = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "mlruns"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_champion_model():
    """Resolve the current champion alias and load its mounted MLflow artifacts."""
    model_version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_id = model_version.source.removeprefix("models:/")
    experiment_id = mlflow.get_run(model_version.run_id).info.experiment_id
    model_path = MLFLOW_ARTIFACT_ROOT / experiment_id / "models" / model_id / "artifacts"

    logger.info(
        "Loading champion model from MLflow.",
        extra={"model_name": MODEL_NAME, "alias": MODEL_ALIAS, "model_path": str(model_path)},
    )
    return mlflow.sklearn.load_model(str(model_path))


model = load_champion_model()


app = FastAPI()

class UserInput(BaseModel):
    sleep_duration: Annotated[
        float, Field(..., gt=1, lt=12, description="Sleep duration")
    ]
    heart_rate: Annotated[float, Field(..., ge=50, lt=108, description="Heart rate")]
    bmi: Annotated[float, Field(..., gt=0)]
    calorie_expenditure: Annotated[float, Field(..., gt=0)]
    step_count: Annotated[float, Field(..., gt=0)]
    exercise_duration: Annotated[float, Field(..., gt=0)]
    water_intake: Annotated[float, Field(..., gt=0)]
    diet_type: Annotated[Literal["balanced", "veg", "non-veg", "Unknown"], Field(...)]
    stress_level: Annotated[Literal["medium", "high", "low", "Unknown"], Field(...)]
    sleep_quality: Annotated[Literal["poor", "good", "average"], Field(...)]
    physical_activity_level: Annotated[
        Literal["sedentary", "moderate", "active"], Field(...)
    ]
    smoking_alcohol: Annotated[
        Literal["no", "yes", "occasional", "Unknown"], Field(...)
    ]
    gender: Annotated[Literal["female", "other", "male", "Unknown"], Field(...)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_health_risk(data: UserInput):

    logger.info("Prediction request received.")
    input_df = pd.DataFrame([data.model_dump()])
    try:
        predicted = model.predict(input_df)[0]
        logger.info(
            "Prediction completed successfully.",
            extra={"endpoint": "/predict", "model": "StudentRiskPipeline"},
        )

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")
    return {"health_condition": predicted}
