import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@patch('app.main.app_ready', True)
@patch('app.main.services')
def test_health_endpoint(mock_services, client):
    mock_services.explainer.llm_pipeline = None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@patch('app.main.app_ready', True)
@patch('app.main.services')
def test_predict_endpoint(mock_services, client):
    mock_services.predict.return_value = (
        0.45,
        "MODERATE",
        [{"feature": "feat_1", "impact": 0.2}],
        [{"feature": "feat_1", "shap_value": 0.15}],
        [{"feature": "feat_1", "impact": 0.18, "direction": "increasing"}],
        [],
        [],
        {},
        None,
    )
    
    payload = {
        "person_age": 30,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "VENTURE",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    }
    
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "risk_level" in data


@patch('app.main.app_ready', True)
def test_predict_invalid_input(client):
    payload = {"person_age": 10}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


@patch('app.main.app_ready', False)
def test_predict_not_ready(client):
    payload = {
        "person_age": 30,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "VENTURE",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 503

