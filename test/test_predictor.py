import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from app.predictor import Predictor
from app.settings import HIGH_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD


@patch('app.predictor.joblib.load')
@patch('app.predictor.keras.models.load_model')
def test_predictor_init(mock_load_model, mock_joblib):
    mock_bundle = {
        'xgb_deep': Mock(),
        'xgb_shallow': Mock(),
        'lgbm_fast': Mock(),
        'catboost_robust': Mock(),
    }
    mock_joblib.return_value = mock_bundle
    mock_load_model.return_value = Mock()
    
    predictor = Predictor()
    assert predictor.base_models is not None
    assert predictor.meta_model is not None
    assert predictor.neural_model is not None


@patch('app.predictor.joblib.load')
@patch('app.predictor.keras.models.load_model')
def test_predict_scores(mock_load_model, mock_joblib):
    # Ensemble collaboration: base models predict, then meta-learner combines
    mock_bundle = {
        'xgb_deep': Mock(),
        'xgb_shallow': Mock(),
        'lgbm_fast': Mock(),
        'catboost_robust': Mock(),
    }
    for model in mock_bundle.values():
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    mock_joblib.return_value = mock_bundle
    mock_neural = Mock()
    mock_neural.predict.return_value = np.array([[0.6]])
    mock_meta = Mock()
    mock_meta.predict.return_value = np.array([[0.55]])
    mock_load_model.side_effect = [mock_meta, mock_neural]
    
    predictor = Predictor()
    frame = pd.DataFrame([[1.0] * 50])
    scores = predictor.predict_scores(frame)
    
    assert len(scores) == 1
    assert 0 <= scores[0] <= 1


def test_risk_label():
    assert Predictor.risk_label(0.6) == "HIGH"
    assert Predictor.risk_label(0.4) == "MODERATE"
    assert Predictor.risk_label(0.2) == "LOW"
    assert Predictor.risk_label(HIGH_RISK_THRESHOLD) == "HIGH"
    assert Predictor.risk_label(MODERATE_RISK_THRESHOLD) == "MODERATE"

