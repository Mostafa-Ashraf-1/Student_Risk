
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from logger import configure_logging
from pydantic import BaseModel, Field

configure_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.pkl"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("starting the application")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, 'rb') as f:  # noqa: ASYNC230
        saved = pickle.load(f)
        ml_models['pipeline'] = saved['pipeline']
    logger.info("Model loaded successfully.")

    yield

    logger.info("Shutting down application.")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

inverse_map = {
    0: "fit",
    1: "at-risk",
    2: "unhealthy"
}

class UserInput(BaseModel):

    sleep_duration : Annotated[float, Field(..., gt=1, lt=12, description= 'Sleep duration')]
    heart_rate : Annotated[float, Field(..., ge=50, lt=108, description='Heart rate')]
    bmi: Annotated[float, Field(..., gt=0)]
    calorie_expenditure: Annotated[float, Field(..., gt=0)]
    step_count: Annotated[float, Field(..., gt=0)]
    exercise_duration: Annotated[float, Field(..., gt=0)]
    water_intake: Annotated[float, Field(..., gt=0)]
    diet_type: Annotated[Literal['balanced', 'veg', 'non-veg', 'Unknown'] , Field(...)]
    stress_level: Annotated[Literal['medium', 'high', 'low', 'Unknown'] , Field(...)]
    sleep_quality: Annotated[Literal['poor', 'good', 'average'] , Field(...)]
    physical_activity_level: Annotated[Literal['sedentary', 'moderate', 'active'] , Field(...)]
    smoking_alcohol: Annotated[Literal['no', 'yes', 'occasional', 'Unknown'] , Field(...)]
    gender: Annotated[Literal['female', 'other', 'male', 'Unknown'] , Field(...)]


@app.get('/health')
def health():
    return {"status" : "ok"}


@app.post('/predict')
def predict_health_risk(data: UserInput):
    
    pipeline = ml_models.get('pipeline')

    if not pipeline:
        logger.critical("Model is not loaded")
        raise HTTPException(status_code=500, detail='Model is not loaded')

    # logger.info("Model loaded successfully.") << Shouldn't be in the predict.
    logger.info("Prediction request received.")
    input_df = pd.DataFrame([data.model_dump()])
    try:
        predicted = pipeline.predict(input_df)[0]
        logger.info("Prediction completed successfully.",
                    extra={
                        "endpoint" : "/predict",
                        "model" : "StudentRiskPipeline"
                    })

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail='Prediction failed.')
    return {'health_condition' : inverse_map[predicted]}
