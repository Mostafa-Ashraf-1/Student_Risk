import logging
from typing import Annotated, Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.pyfunc.load_model(
    model_uri='models:/Student-Risk-Model@champion'
)


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
