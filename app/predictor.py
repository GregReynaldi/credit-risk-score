from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from .settings import MODELS_DIR, HIGH_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD


class Predictor:
    def __init__(self) -> None:
        bundle = joblib.load(MODELS_DIR / "ensemble_base_models_neural.joblib")
        self.base_models = {
            "xgb_deep": bundle["xgb_deep"],
            "xgb_shallow": bundle["xgb_shallow"],
            "lgbm_fast": bundle["lgbm_fast"],
            "catboost_robust": bundle["catboost_robust"],
        }
        self.meta_model = keras.models.load_model(
            MODELS_DIR / "meta_learner_neural.h5", compile=False
        )
        self.neural_model = keras.models.load_model(
            MODELS_DIR / "residual_neural.h5", compile=False
        )

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        # Ensemble collaboration (Layer 1): base models predict, then meta-learner combines
        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T

        preds = [
            self.base_models["xgb_deep"].predict_proba(frame)[:, 1],
            self.base_models["xgb_shallow"].predict_proba(frame)[:, 1],
            self.base_models["lgbm_fast"].predict_proba(frame)[:, 1],
            self.base_models["catboost_robust"].predict_proba(frame)[:, 1],
        ]
        neural = self.neural_model.predict(frame.values, verbose=0).ravel()
        stacked = np.column_stack(preds + [neural])
        final_prob = self.meta_model.predict(stacked, verbose=0).ravel()
        return final_prob

    def predict_one(self, frame: pd.DataFrame) -> float:
        return float(self.predict_scores(frame)[0])

    @staticmethod
    def risk_label(probability: float) -> str:
        if probability >= HIGH_RISK_THRESHOLD:
            return "HIGH"
        if probability >= MODERATE_RISK_THRESHOLD:
            return "MODERATE"
        return "LOW"

