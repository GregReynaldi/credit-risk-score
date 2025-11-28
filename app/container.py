from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .explainer import Explainer
from .predictor import Predictor
from .preprocessor import Preprocessor


@dataclass
class Services:
    # Orchestrates three-layer collaboration: preprocessing → ensemble prediction → explainability
    preprocessor: Preprocessor
    predictor: Predictor
    explainer: Explainer

    def predict(self, payload: Dict, include_llm: bool = False):
        processed = self.preprocessor.transform(payload)
        probability = self.predictor.predict_one(processed)
        risk = self.predictor.risk_label(probability)
        
        # Get LIME and SHAP features - these should always return non-empty lists now
        lime_features = self.explainer.lime_top_features(processed)
        shap_features = self.explainer.shap_top_features(processed)
        
        # Ensure we have lists (not None)
        if not isinstance(lime_features, list):
            lime_features = []
        if not isinstance(shap_features, list):
            shap_features = []
        
        risk_increasing, risk_decreasing = self.explainer.risk_drivers(
            lime_features, shap_features
        )
        
        # Ensure risk drivers are lists
        if not isinstance(risk_increasing, list):
            risk_increasing = []
        if not isinstance(risk_decreasing, list):
            risk_decreasing = []
        
        # Get top feature names for global context
        all_features = lime_features + shap_features
        top_feature_names = [
            item["feature"] for item in all_features[:10] if isinstance(item, dict) and "feature" in item
        ]
        
        # If no features found, use feature names from explainer
        if not top_feature_names and hasattr(self.explainer, 'feature_names'):
            top_feature_names = self.explainer.feature_names[:10]
        
        global_context = self.explainer.global_context_for_features(top_feature_names)
        model_agreement = self.explainer.model_agreement_for_features(top_feature_names)
        
        # Ensure global_context is a list
        if not isinstance(global_context, list):
            global_context = []
        if not isinstance(model_agreement, dict):
            model_agreement = {}

        llm_text = None
        if include_llm:
            combined_features = [
                {"feature": item["feature"], "impact": item.get("impact", item.get("shap_value", 0))}
                for item in all_features[:7] if isinstance(item, dict) and "feature" in item
            ]
            # Ensure we have at least some features for LLM
            if not combined_features and all_features:
                combined_features = [
                    {"feature": item.get("feature", "unknown"), "impact": item.get("impact", item.get("shap_value", 0))}
                    for item in all_features[:7]
                ]
            
            llm_text = self.explainer.llm_summary(
                probability, risk, combined_features, payload
            )
            
            # Ensure LLM text is a string
            if not isinstance(llm_text, str):
                llm_text = f"Based on the analysis, this applicant has a {probability:.2%} probability of default, classified as {risk} risk."

        return (
            probability,
            risk,
            lime_features,
            shap_features,
            risk_increasing,
            risk_decreasing,
            global_context,
            model_agreement,
            llm_text,
        )


# Initialize services - will be loaded at startup
# These are built at module import time to ensure they're ready before server starts
print("Initializing services...")
print("  Loading preprocessor...")
preprocessor = Preprocessor.build()
print("  Loading predictor and models...")
predictor = Predictor()
print("  Loading explainer (LIME, SHAP, LLM, global insights)...")
explainer = Explainer.build(predictor=predictor)
print("  Services initialized successfully")

services = Services(
    preprocessor=preprocessor,
    predictor=predictor,
    explainer=explainer,
)

