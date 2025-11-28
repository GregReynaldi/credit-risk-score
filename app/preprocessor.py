from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
from sklearn.impute import KNNImputer

from .settings import DATASET_DIR, MODELS_DIR


NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


@dataclass
class Preprocessor:
    scaler: object
    label_encoder: object
    one_hot_encoder: object
    feature_names: List[str]
    imputer: KNNImputer = field(repr=False)

    @classmethod
    def build(cls) -> "Preprocessor":
        with open(MODELS_DIR / "RobutScaler.pkl", "rb") as fh:
            scaler = pickle.load(fh)
        with open(MODELS_DIR / "LabelEncoder.pkl", "rb") as fh:
            label_encoder = pickle.load(fh)
        with open(MODELS_DIR / "OneHotEncoder.pkl", "rb") as fh:
            one_hot_encoder = pickle.load(fh)

        feature_names = pd.read_pickle(DATASET_DIR / "X_train.pkl").columns.tolist()

        raw_df = pd.read_csv(DATASET_DIR / "credit_risk_dataset.csv")
        numeric_data = raw_df[NUMERIC_COLS].copy()
        imputer = KNNImputer(n_neighbors=5)
        imputer.fit(numeric_data)

        return cls(
            scaler=scaler,
            label_encoder=label_encoder,
            one_hot_encoder=one_hot_encoder,
            feature_names=feature_names,
            imputer=imputer,
        )

    def transform(self, payload: Dict) -> pd.DataFrame:
        frame = pd.DataFrame([payload])
        frame["person_age"] = frame["person_age"].clip(upper=100)
        frame["person_emp_length"] = frame["person_emp_length"].clip(upper=50)

        numeric = frame[NUMERIC_COLS].copy()
        numeric_imputed = self.imputer.transform(numeric)
        numeric_scaled = self.scaler.transform(numeric_imputed)
        numeric_df = pd.DataFrame(numeric_scaled, columns=NUMERIC_COLS)

        ordinal_cols = ['loan_grade']
        ordinal_encoded = self.label_encoder.transform(frame[ordinal_cols].values.ravel())
        ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_cols)
        
        nominal_cols = ['person_home_ownership', 'loan_intent']
        nominal_encoded = self.one_hot_encoder.transform(frame[nominal_cols])
        nominal_feature_names = self.one_hot_encoder.get_feature_names_out(nominal_cols)
        nominal_df = pd.DataFrame(nominal_encoded, columns=nominal_feature_names)
        
        binary_cols = ['cb_person_default_on_file']
        binary_encoded = frame[binary_cols].replace({'Y': 1, 'N': 0}).values
        binary_df = pd.DataFrame(binary_encoded, columns=binary_cols)
        
        categorical_df = pd.concat([ordinal_df, nominal_df, binary_df], axis=1)

        processed = pd.concat([numeric_df, categorical_df], axis=1)
        processed = processed.reindex(columns=self.feature_names, fill_value=0)
        return processed

