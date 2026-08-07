import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
        "sleep_duration": 8,
        "heart_rate": 72,
        "bmi": 23.5,
        "calorie_expenditure": 2200,
        "step_count": 9000,
        "exercise_duration": 45,
        "water_intake": 2.5,
        "diet_type": "balanced",
        "stress_level": "low",
        "sleep_quality": "good",
        "physical_activity_level": "active",
        "smoking_alcohol": "no",
        "gender": "male",
    }
