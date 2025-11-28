import pytest
import pandas as pd
from unittest.mock import patch, Mock
from app.preprocessor import Preprocessor


@patch('app.preprocessor.pickle.load')
@patch('app.preprocessor.pd.read_pickle')
@patch('app.preprocessor.pd.read_csv')
@patch('app.preprocessor.KNNImputer')
def test_preprocessor_build(mock_imputer_class, mock_csv, mock_pickle, mock_load):
    mock_load.side_effect = [Mock(), Mock(), Mock()]
    mock_pickle.return_value = pd.DataFrame([[1.0] * 50], columns=[f'feat_{i}' for i in range(50)])
    mock_csv.return_value = pd.DataFrame({'person_age': [30], 'person_income': [50000]})
    mock_imputer = Mock()
    mock_imputer_class.return_value = mock_imputer
    
    preprocessor = Preprocessor.build()
    assert preprocessor.scaler is not None
    assert preprocessor.label_encoder is not None
    assert preprocessor.one_hot_encoder is not None


@patch('app.preprocessor.pickle.load')
@patch('app.preprocessor.pd.read_pickle')
@patch('app.preprocessor.pd.read_csv')
@patch('app.preprocessor.KNNImputer')
def test_preprocessor_transform(mock_imputer_class, mock_csv, mock_pickle, mock_load):
    mock_scaler = Mock()
    mock_scaler.transform.return_value = [[0.5] * 7]
    mock_label_encoder = Mock()
    mock_label_encoder.transform.return_value = [2]
    mock_onehot = Mock()
    mock_onehot.transform.return_value = [[1, 0, 0, 0, 1, 0, 0, 0]]
    mock_onehot.get_feature_names_out.return_value = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8']
    
    mock_load.side_effect = [mock_scaler, mock_label_encoder, mock_onehot]
    mock_pickle.return_value = pd.DataFrame([[1.0] * 50], columns=[f'feat_{i}' for i in range(50)])
    mock_csv.return_value = pd.DataFrame({'person_age': [30], 'person_income': [50000]})
    mock_imputer = Mock()
    mock_imputer.transform.return_value = [[30.0, 50000.0, 5.0, 10000.0, 10.5, 0.2, 3.0]]
    mock_imputer_class.return_value = mock_imputer
    
    preprocessor = Preprocessor.build()
    payload = {
        'person_age': 30,
        'person_income': 50000,
        'person_emp_length': 5.0,
        'loan_amnt': 10000,
        'loan_int_rate': 10.5,
        'loan_percent_income': 0.2,
        'cb_person_cred_hist_length': 3,
        'person_home_ownership': 'RENT',
        'loan_intent': 'VENTURE',
        'loan_grade': 'C',
        'cb_person_default_on_file': 'N',
    }
    
    result = preprocessor.transform(payload)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

