import pytest
from fastapi.testclient import TestClient
from app import app

def test_health(client):

    response = client.get('/health')
    assert response.status_code == 200
    # assert response.json() == {"status": "ok"}

@pytest.mark.parametrize(
    "field, value",
    [
        ("gender", "helicopter"),
        ("diet_type", "pizza"),
        ("stress_level", "full"),
        ("sleep_quality", "excellent"),
    ],
)
def test_invalid_literals(client, valid_payload, field, value):

    valid_payload[field] = value
    response = client.post('/predict',
                           json=valid_payload)

    assert response.status_code == 422