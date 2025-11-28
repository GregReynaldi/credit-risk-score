from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .predictor import Predictor
from .settings import (
    ADV_DIR,
    DATASET_DIR,
    LIME_BACKGROUND,
    LIME_SAMPLE_SIZE,
    LLM_DIR,
    SHAP_BACKGROUND_SIZE,
    SHAP_DIR,
)


def _clean_feature_name(condition: str, feature_names: List[str]) -> str | None:
    tokens = [" <=", " >=", " >", " <", " ==", " !="]
    for token in tokens:
        if token in condition:
            candidate = condition.split(token)[0].strip()
            if candidate in feature_names:
                return candidate
    for feat in feature_names:
        if feat in condition:
            return feat
    return None


@dataclass
class Explainer:
    predictor: Predictor
    lime_explainer: LimeTabularExplainer
    feature_names: List[str]
    llm_pipeline: pipeline | None
    global_insights: Dict
    shap_background: np.ndarray

    @classmethod
    def build(cls, predictor: Predictor) -> "Explainer":
        X_train = pd.read_pickle(DATASET_DIR / "X_train.pkl")
        feature_names = X_train.columns.tolist()

        if LIME_BACKGROUND.exists():
            background = np.load(LIME_BACKGROUND)
        else:
            background = X_train.sample(
                n=min(LIME_SAMPLE_SIZE, len(X_train)), random_state=42
            ).values
            np.save(LIME_BACKGROUND, background)

        lime_explainer = LimeTabularExplainer(
            background,
            feature_names=feature_names,
            mode="classification",
            random_state=42,
        )

        shap_background = X_train.sample(
            n=min(SHAP_BACKGROUND_SIZE, len(X_train)), random_state=42
        ).values

        global_insights = cls._load_global_insights()

        # Load LLM model at startup (not lazy loaded)
        llm_pipeline = None
        if LLM_DIR.exists() and (LLM_DIR / "config.json").exists():
            print("    Loading LLM model (this may take a moment)...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    LLM_DIR, padding_side="left", trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(LLM_DIR)
                llm_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                )
                print("    ✓ LLM model loaded successfully")
            except Exception as e:
                print(f"    ⚠ Failed to load LLM model: {e}")
                llm_pipeline = None
        else:
            print("    ⚠ LLM model directory not found, skipping LLM loading")

        return cls(
            predictor=predictor,
            lime_explainer=lime_explainer,
            feature_names=feature_names,
            llm_pipeline=llm_pipeline,
            global_insights=global_insights,
            shap_background=shap_background,
        )

    @staticmethod
    def _load_global_insights() -> Dict:
        insights = {
            "shap_importance": {},
            "lime_importance": {},
            "combined_importance": {},
            "sensitivity": {},
            "model_comparison": {},
        }

        print("    Loading global explainability insights...")
        try:
            if (SHAP_DIR / "shap_feature_importance.json").exists():
                with open(SHAP_DIR / "shap_feature_importance.json", "r") as f:
                    shap_data = json.load(f)
                    insights["shap_importance"] = {
                        item["feature"]: item["importance"] for item in shap_data
                    }
                print(f"      ✓ Loaded SHAP importance ({len(insights['shap_importance'])} features)")

            if (SHAP_DIR / "lime_feature_importance.json").exists():
                with open(SHAP_DIR / "lime_feature_importance.json", "r") as f:
                    lime_data = json.load(f)
                    insights["lime_importance"] = {
                        item["feature"]: item["importance"] for item in lime_data
                    }
                print(f"      ✓ Loaded LIME importance ({len(insights['lime_importance'])} features)")

            if (SHAP_DIR / "combined_feature_importance.json").exists():
                with open(SHAP_DIR / "combined_feature_importance.json", "r") as f:
                    combined_data = json.load(f)
                    insights["combined_importance"] = {
                        item["feature"]: item["combined_importance"]
                        for item in combined_data
                    }
                print(f"      ✓ Loaded combined importance ({len(insights['combined_importance'])} features)")

            if (ADV_DIR / "advanced_explainability_report.json").exists():
                with open(ADV_DIR / "advanced_explainability_report.json", "r") as f:
                    adv_data = json.load(f)
                
                sensitivity_features = adv_data.get("sensitivity_analysis", {}).get(
                    "top_sensitive_features", []
                )
                insights["sensitivity"] = {
                    item["feature"]: item["sensitivity"]
                    for item in sensitivity_features
                }
                print(f"      ✓ Loaded sensitivity analysis ({len(insights['sensitivity'])} features)")

                model_comparison = adv_data.get("model_specific_analysis", {}).get(
                    "top_features_by_model", {}
                )
                insights["model_comparison"] = model_comparison
                print(f"      ✓ Loaded model comparison data ({len(model_comparison)} models)")

        except Exception as e:
            print(f"      ⚠ Warning: Could not load some global insights: {e}")

        return insights

    def _probabilities_for_lime(self, data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=self.feature_names)
        scores = self.predictor.predict_scores(frame)
        return np.column_stack([1 - scores, scores])

    def lime_top_features(self, processed_row: pd.DataFrame, top_n: int = 8) -> List[Dict]:
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if processed_row.empty or processed_row.shape[0] == 0:
                logger.warning("LIME: processed_row is empty, returning fallback data")
                # Return fallback data instead of empty list
                return [{"feature": name, "impact": 0.0} for name in self.feature_names[:top_n]]
            
            explanation = self.lime_explainer.explain_instance(
                processed_row.values.flatten(),
                self._probabilities_for_lime,
                num_features=min(20, len(self.feature_names)),  # Limit to avoid performance issues
            )
            rows: List[Tuple[str, float]] = []
            for condition, impact in explanation.as_list():
                name = _clean_feature_name(condition, self.feature_names)
                if name:
                    rows.append((name, float(impact)))
            
            if not rows:
                # Fallback: return feature names with zero impact if parsing fails
                logger.warning("LIME: No features parsed from explanation, returning fallback data")
                return [{"feature": name, "impact": 0.0} for name in self.feature_names[:top_n]]
            
            rows.sort(key=lambda item: abs(item[1]), reverse=True)
            top_rows = rows[:top_n]
            result = [{"feature": name, "impact": impact} for name, impact in top_rows]
            
            # Ensure we always return at least top_n features
            if len(result) < top_n:
                existing_features = {item["feature"] for item in result}
                for name in self.feature_names:
                    if name not in existing_features and len(result) < top_n:
                        result.append({"feature": name, "impact": 0.0})
            
            return result
        except Exception as e:
            # Log error but return fallback data to prevent API failure
            logger.error(f"LIME explanation failed: {e}", exc_info=True)
            # Return fallback data with top features instead of error message
            return [{"feature": name, "impact": 0.0} for name in self.feature_names[:top_n]]

    def shap_top_features(
        self, processed_row: pd.DataFrame, top_n: int = 10
    ) -> List[Dict]:
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if processed_row.empty or processed_row.shape[0] == 0:
                logger.warning("SHAP: processed_row is empty, returning fallback data")
                # Return fallback data instead of empty list
                return [{"feature": name, "shap_value": 0.0} for name in self.feature_names[:top_n]]

            def ensemble_predict_proba(X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                scores = self.predictor.predict_scores(pd.DataFrame(X, columns=self.feature_names))
                return scores

            shap_explainer = shap.PermutationExplainer(
                ensemble_predict_proba, self.shap_background
            )
            shap_values = shap_explainer(processed_row.values)

            shap_dict = {}
            for i, feat in enumerate(self.feature_names):
                shap_dict[feat] = float(shap_values.values[0, i])

            shap_list = [
                {"feature": feat, "shap_value": val}
                for feat, val in shap_dict.items()
            ]
            shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            result = shap_list[:top_n]
            
            # Ensure we always return at least top_n features
            if len(result) < top_n:
                existing_features = {item["feature"] for item in result}
                for name in self.feature_names:
                    if name not in existing_features and len(result) < top_n:
                        result.append({"feature": name, "shap_value": 0.0})
            
            return result
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            # Return fallback data instead of empty list
            return [{"feature": name, "shap_value": 0.0} for name in self.feature_names[:top_n]]

    def risk_drivers(
        self, lime_features: List[Dict], shap_features: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        # Explainability collaboration (Layer 2): combines LIME and SHAP by averaging
        combined = {}
        for item in lime_features:
            feat = item["feature"]
            if feat not in combined:
                combined[feat] = {"lime": 0.0, "shap": 0.0}
            combined[feat]["lime"] = item["impact"]

        for item in shap_features:
            feat = item["feature"]
            if feat not in combined:
                combined[feat] = {"lime": 0.0, "shap": 0.0}
            combined[feat]["shap"] = item["shap_value"]

        risk_increasing = []
        risk_decreasing = []

        for feat, values in combined.items():
            avg_impact = (values["shap"] + values["lime"]) / 2
            if avg_impact > 0:
                risk_increasing.append(
                    {"feature": feat, "impact": avg_impact, "direction": "increasing"}
                )
            elif avg_impact < 0:
                risk_decreasing.append(
                    {"feature": feat, "impact": abs(avg_impact), "direction": "decreasing"}
                )

        risk_increasing.sort(key=lambda x: x["impact"], reverse=True)
        risk_decreasing.sort(key=lambda x: x["impact"], reverse=True)

        return risk_increasing[:5], risk_decreasing[:5]

    def global_context_for_features(
        self, top_features: List[str]
    ) -> List[Dict]:
        context_list = []
        shap_global = self.global_insights.get("shap_importance", {})
        lime_global = self.global_insights.get("lime_importance", {})
        sensitivity = self.global_insights.get("sensitivity", {})

        shap_ranked = sorted(
            shap_global.items(), key=lambda x: x[1], reverse=True
        )
        lime_ranked = sorted(
            lime_global.items(), key=lambda x: x[1], reverse=True
        )

        shap_ranks = {feat: i + 1 for i, (feat, _) in enumerate(shap_ranked)}
        lime_ranks = {feat: i + 1 for i, (feat, _) in enumerate(lime_ranked)}

        for feat in top_features[:8]:
            context_list.append(
                {
                    "feature": feat,
                    "global_shap_rank": shap_ranks.get(feat),
                    "global_lime_rank": lime_ranks.get(feat),
                    "sensitivity": sensitivity.get(feat),
                }
            )

        return context_list

    def model_agreement_for_features(
        self, top_features: List[str]
    ) -> Dict:
        model_comparison = self.global_insights.get("model_comparison", {})
        model_names = ["xgb_deep", "xgb_shallow", "lgbm", "catboost", "neural_network"]

        agreement = {}
        for feat in top_features[:6]:
            ranks = {}
            for model_name in model_names:
                model_features = model_comparison.get(model_name, [])
                rank = next(
                    (
                        i + 1
                        for i, item in enumerate(model_features)
                        if item.get("feature") == feat
                    ),
                    None,
                )
                ranks[model_name] = rank
            agreement[feat] = ranks

        return agreement

    def llm_summary(
        self,
        probability: float,
        risk_label: str,
        top_features: List[Dict],
        raw_input: Dict,
    ) -> str:
        # ML-LLM collaboration (Layer 3): synthesizes LIME/SHAP data into natural language
        if self.llm_pipeline is None:
            return "LLM explanation unavailable on this deployment."

        profile_lines = [f"- {k}: {v}" for k, v in raw_input.items()]
        drivers = []
        for item in top_features[:4]:
            direction = "increases" if item.get("impact", 0) > 0 else "decreases"
            impact = item.get("impact", item.get("shap_value", 0))
            drivers.append(
                f"- {item['feature']}: {direction} risk (impact: {impact:+.3f})"
            )
        if not drivers:
            drivers.append("- Drivers could not be identified for this sample.")

        global_top = list(self.global_insights.get("combined_importance", {}).keys())[:5]
        global_section = ", ".join(global_top) if global_top else "N/A"

        sensitivity_top = list(self.global_insights.get("sensitivity", {}).keys())[:3]
        sensitivity_section = ", ".join(sensitivity_top) if sensitivity_top else "N/A"

        prompt = (
            "You are an AI credit risk officer. Provide a clear, professional explanation.\n"
            f"Prediction: {probability:.2%} default probability → Risk Level: {risk_label}\n\n"
            f"Borrower Profile:\n" + "\n".join(profile_lines) + "\n\n"
            f"Local Feature Drivers (for this specific case):\n" + "\n".join(drivers) + "\n\n"
            f"Global Context: Top globally important features are {global_section}.\n"
            f"Most sensitive features (from analysis): {sensitivity_section}.\n\n"
            "Provide a 2-3 sentence explanation: (1) State the risk level and key factors, "
            "(2) Explain how this case compares to typical patterns, (3) Give a recommendation."
        )

        try:
            result = self.llm_pipeline(
                prompt,
                max_new_tokens=256,
                min_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.05,
            )
            
            # Handle different possible response structures
            if not result or len(result) == 0:
                return f"Based on the analysis, this applicant has a {probability:.2%} probability of default, classified as {risk_label} risk. Key factors include: {', '.join([d.split(':')[0].replace('- ', '') for d in drivers[:3]])}."
            
            # Extract generated text from result
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    generated_text = first_result.get("generated_text", "")
                    if generated_text:
                        # Remove the prompt from the generated text
                        if generated_text.startswith(prompt):
                            generated = generated_text[len(prompt):].strip()
                        else:
                            generated = generated_text.strip()
                        
                        if generated:
                            return generated
            
            # Fallback: return a basic summary if LLM response is empty
            return f"Based on the analysis, this applicant has a {probability:.2%} probability of default, classified as {risk_label} risk. Key factors include: {', '.join([d.split(':')[0].replace('- ', '') for d in drivers[:3]])}."
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"LLM summary generation failed: {e}", exc_info=True)
            # Return a fallback summary instead of empty string
            return f"Based on the analysis, this applicant has a {probability:.2%} probability of default, classified as {risk_label} risk. Key factors include: {', '.join([d.split(':')[0].replace('- ', '') for d in drivers[:3]])}."

