import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from app.explainer import Explainer
from app.predictor import Predictor


@patch('app.explainer.pd.read_pickle')
@patch('app.explainer.LimeTabularExplainer')
@patch('app.explainer.np.load')
@patch('app.explainer.np.save')
def test_lime_top_features(mock_save, mock_load, mock_lime_class, mock_read_pickle):
    mock_read_pickle.return_value = pd.DataFrame([[1.0] * 50], columns=[f'feat_{i}' for i in range(50)])
    mock_load.side_effect = FileNotFoundError()
    
    mock_lime = Mock()
    mock_explanation = Mock()
    mock_explanation.as_list.return_value = [
        ('feat_0 <= 0.5', 0.3),
        ('feat_1 > 1.0', -0.2),
    ]
    mock_lime.explain_instance.return_value = mock_explanation
    mock_lime_class.return_value = mock_lime
    
    predictor = Mock(spec=Predictor)
    predictor.predict_scores.return_value = [0.5]
    
    explainer = Explainer.build(predictor)
    processed = pd.DataFrame([[1.0] * 50], columns=[f'feat_{i}' for i in range(50)])
    result = explainer.lime_top_features(processed)
    
    assert len(result) > 0
    assert 'feature' in result[0]
    assert 'impact' in result[0]


@patch('app.explainer.pd.read_pickle')
@patch('app.explainer.shap.PermutationExplainer')
def test_shap_top_features(mock_shap_class, mock_read_pickle):
    mock_read_pickle.return_value = pd.DataFrame([[1.0] * 50], columns=[f'feat_{i}' for i in range(50)])
    
    mock_shap_explainer = Mock()
    mock_shap_values = Mock()
    mock_shap_values.values = np.array([[0.1, -0.05, 0.02] + [0.0] * 47])
    mock_shap_explainer.return_value = mock_shap_values
    mock_shap_class.return_value = mock_shap_explainer
    
    predictor = Mock(spec=Predictor)
    predictor.predict_scores.return_value = [0.5]
    
    explainer = Explainer.build(predictor)
    processed = pd.DataFrame([[1.0] * 50], columns=[f'feat_{i}' for i in range(50)])
    result = explainer.shap_top_features(processed)
    
    assert len(result) > 0
    assert 'feature' in result[0]
    assert 'shap_value' in result[0]


def test_risk_drivers():
    # Explainability collaboration: combines LIME and SHAP by averaging
    lime_features = [
        {'feature': 'feat_1', 'impact': 0.3},
        {'feature': 'feat_2', 'impact': -0.2},
    ]
    shap_features = [
        {'feature': 'feat_1', 'shap_value': 0.25},
        {'feature': 'feat_2', 'shap_value': -0.15},
    ]
    
    predictor = Mock()
    explainer = Explainer(
        predictor=predictor,
        lime_explainer=Mock(),
        feature_names=['feat_1', 'feat_2'],
        llm_pipeline=None,
        global_insights={},
        shap_background=Mock(),
    )
    
    risk_inc, risk_dec = explainer.risk_drivers(lime_features, shap_features)
    
    assert len(risk_inc) > 0
    assert risk_inc[0]['direction'] == 'increasing'
    assert len(risk_dec) > 0
    assert risk_dec[0]['direction'] == 'decreasing'

