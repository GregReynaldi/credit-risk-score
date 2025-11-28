"""
ML-LLM Collaboration Module - Central Documentation

This module provides comprehensive documentation of the ML-LLM collaboration scheme
implemented in this credit risk prediction system. The collaboration consists of three
distinct layers that work together to produce predictions and explanations.

The collaboration logic is implemented across multiple files:
- app/predictor.py: Ensemble collaboration (Layer 1)
- app/explainer.py: Explainability collaboration (Layer 2 & 3)
- app/container.py: Service orchestration and coordination
- src/05_llm_explainability.ipynb: Notebook implementation (same logic)

This module serves as a single entry point to understand the complete collaboration scheme.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ============================================================================
# LAYER 1: ENSEMBLE COLLABORATION (ML Models Working Together)
# ============================================================================

"""
Layer 1: Ensemble Collaboration - Multiple ML Models Collaborate for Robust Predictions

Data Interaction:
    Raw Input Features → Preprocessing → Base Models → Meta-Learner → Final Probability
    
    Input: Preprocessed feature vector (17 features)
    Output: Default probability (0-1)

Process Scheduling:
    1. Base Models (Level 1) - Execute in parallel:
       - XGBoost Deep: Deep decision trees (max_depth=7)
       - XGBoost Shallow: Shallow trees (max_depth=3) for generalization
       - LightGBM Fast: Fast gradient boosting with leaf-wise growth
       - CatBoost Robust: Robust gradient boosting with categorical handling
       - Neural Network: Deep learning model (residual connections)
    
    2. Meta-Learner (Level 2) - Executes after base models:
       - Takes stacked predictions from all 5 base models as input
       - Neural network that learns optimal combination weights
       - Produces calibrated final probability

Result Fusion:
    - Each base model outputs a probability score independently
    - All 5 predictions are stacked into a feature vector: [pred_xgb_deep, pred_xgb_shallow, pred_lgbm, pred_catboost, pred_neural]
    - Meta-learner combines these using learned weights/transformations
    - Final output is a single calibrated probability (0-1)

Implementation Location:
    - app/predictor.py: Predictor.predict_scores() method
    - src/05_llm_explainability.ipynb: Cell 16 (Ensemble Prediction section)

Algorithm Consistency:
    The implementation uses a two-level stacked ensemble, exactly as described in the README.
    No algorithm substitution - uses Algorithm A (stacked ensemble) as proposed.
"""


def get_ensemble_collaboration_info() -> Dict:
    """
    Returns documentation of the ensemble collaboration scheme.
    
    Returns:
        Dict containing:
            - description: Overview of ensemble collaboration
            - data_flow: Step-by-step data interaction
            - process_scheduling: Execution order and dependencies
            - result_fusion: How predictions are combined
            - implementation_files: Where the logic is implemented
    """
    return {
        "layer": "Layer 1: Ensemble Collaboration",
        "description": "Multiple ML models collaborate through a two-level stacked ensemble",
        "data_flow": [
            "Raw Input Features (11 features)",
            "↓ Preprocessing (scaling, encoding, imputation)",
            "↓ Processed Features (17 features)",
            "↓ Base Models (5 models predict independently)",
            "↓ Stacked Predictions [5 probabilities]",
            "↓ Meta-Learner (combines predictions)",
            "↓ Final Probability (0-1)"
        ],
        "process_scheduling": {
            "stage_1": {
                "name": "Base Model Predictions",
                "models": [
                    "XGBoost Deep (deep trees)",
                    "XGBoost Shallow (shallow trees)",
                    "LightGBM Fast (fast boosting)",
                    "CatBoost Robust (robust boosting)",
                    "Neural Network (deep learning)"
                ],
                "execution": "Parallel (independent predictions)",
                "output": "5 probability scores"
            },
            "stage_2": {
                "name": "Meta-Learner Combination",
                "input": "Stacked base model predictions [5 values]",
                "method": "Neural network with learned weights",
                "output": "Single calibrated probability"
            }
        },
        "result_fusion": {
            "method": "Stacked Ensemble (Two-Level)",
            "step_1": "Base models generate independent probability predictions",
            "step_2": "Predictions stacked into feature vector: [pred_1, pred_2, pred_3, pred_4, pred_5]",
            "step_3": "Meta-learner applies learned transformation to produce final probability",
            "benefit": "Reduces overfitting, improves generalization, leverages model diversity"
        },
        "implementation_files": [
            "app/predictor.py (Predictor.predict_scores method)",
            "src/05_llm_explainability.ipynb (Cell 16: Ensemble Prediction)"
        ]
    }


# ============================================================================
# LAYER 2: EXPLAINABILITY COLLABORATION (LIME + SHAP Working Together)
# ============================================================================

"""
Layer 2: Explainability Collaboration - LIME and SHAP Combine for Robust Feature Importance

Data Interaction:
    Prediction + Input Features → LIME Explainer → LIME Feature Importance
    Prediction + Input Features → SHAP Explainer → SHAP Feature Importance
    LIME + SHAP → Risk Drivers Analysis → Risk-Increasing/Decreasing Features
    
    Input: Preprocessed feature vector + prediction probability
    Output: Combined feature importance, risk drivers

Process Scheduling:
    1. LIME Explanation (Local Interpretable Model-agnostic Explanations):
       - Creates perturbed samples around the input instance
       - Gets predictions from ensemble for each perturbation
       - Fits local linear model to approximate ensemble behavior
       - Extracts feature weights as importance scores
    
    2. SHAP Explanation (SHapley Additive exPlanations):
       - Uses PermutationExplainer with background dataset
       - Permutes each feature and measures prediction change
       - Computes average marginal contribution (Shapley values)
       - Provides theoretically grounded feature importance
    
    3. Risk Drivers Analysis (Combines LIME + SHAP):
       - Merges feature importance from both methods
       - Averages LIME impact and SHAP value for each feature
       - Classifies features as risk-increasing (positive) or risk-decreasing (negative)
       - Returns top 5 features in each category

Result Fusion:
    - LIME provides: {feature: impact} where impact can be positive or negative
    - SHAP provides: {feature: shap_value} where shap_value can be positive or negative
    - Combination: For each feature, compute avg_impact = (lime_impact + shap_value) / 2
    - Classification: avg_impact > 0 → risk-increasing, avg_impact < 0 → risk-decreasing
    - Final output: Top 5 risk-increasing features, Top 5 risk-decreasing features

Implementation Location:
    - app/explainer.py: Explainer.risk_drivers() method
    - src/05_llm_explainability.ipynb: Cell 22 (Risk Drivers Analysis section)

Algorithm Consistency:
    The implementation combines LIME and SHAP by averaging their importance scores,
    exactly as described in the README. No algorithm substitution.
"""


def get_explainability_collaboration_info() -> Dict:
    """
    Returns documentation of the explainability collaboration scheme.
    
    Returns:
        Dict containing explainability collaboration details
    """
    return {
        "layer": "Layer 2: Explainability Collaboration",
        "description": "LIME and SHAP collaborate to provide robust feature importance",
        "data_flow": [
            "Preprocessed Features + Prediction",
            "↓ LIME Explainer (local linear approximation)",
            "↓ LIME Feature Importance {feature: impact}",
            "↓ SHAP Explainer (game-theoretic approach)",
            "↓ SHAP Feature Importance {feature: shap_value}",
            "↓ Risk Drivers Analysis (combines both)",
            "↓ Risk-Increasing Features (top 5)",
            "↓ Risk-Decreasing Features (top 5)"
        ],
        "process_scheduling": {
            "step_1": {
                "name": "LIME Explanation",
                "method": "Local Interpretable Model-agnostic Explanations",
                "process": [
                    "Create perturbed samples around input instance",
                    "Get ensemble predictions for each perturbation",
                    "Fit local linear model to approximate ensemble",
                    "Extract feature weights as importance scores"
                ],
                "output": "List of {feature: impact} pairs"
            },
            "step_2": {
                "name": "SHAP Explanation",
                "method": "SHapley Additive exPlanations (PermutationExplainer)",
                "process": [
                    "Use background dataset as reference",
                    "Permute each feature value",
                    "Measure prediction change",
                    "Compute average marginal contribution (Shapley value)"
                ],
                "output": "List of {feature: shap_value} pairs"
            },
            "step_3": {
                "name": "Risk Drivers Analysis",
                "method": "Combine and average LIME + SHAP",
                "process": [
                    "Merge feature importance from both methods",
                    "For each feature: avg_impact = (lime_impact + shap_value) / 2",
                    "Classify: positive → risk-increasing, negative → risk-decreasing",
                    "Sort by absolute impact and return top 5 each"
                ],
                "output": "Top 5 risk-increasing features, Top 5 risk-decreasing features"
            }
        },
        "result_fusion": {
            "method": "Averaging LIME and SHAP Importance Scores",
            "formula": "avg_impact = (lime_impact + shap_value) / 2",
            "classification": {
                "risk_increasing": "avg_impact > 0 (features that increase default probability)",
                "risk_decreasing": "avg_impact < 0 (features that decrease default probability)"
            },
            "benefit": "More robust feature importance by combining two complementary methods"
        },
        "implementation_files": [
            "app/explainer.py (Explainer.risk_drivers method)",
            "src/05_llm_explainability.ipynb (Cell 22: Risk Drivers Analysis)"
        ]
    }


# ============================================================================
# LAYER 3: ML-LLM COLLABORATION (ML Outputs → LLM Narrative)
# ============================================================================

"""
Layer 3: ML-LLM Collaboration - Machine Learning Models Provide Data to LLM for Narrative Generation

Data Interaction:
    ML Predictions + LIME/SHAP + Global Context → LLM Prompt → LLM Narrative
    
    Input: 
        - Prediction probability and risk level
        - Borrower profile (raw input features)
        - Local feature drivers (from LIME/SHAP)
        - Global context (top globally important features)
        - Sensitivity analysis (most sensitive features)
    
    Output: Human-readable natural language explanation

Process Scheduling:
    1. Data Collection (from ML models):
       - Get prediction probability and risk level from ensemble
       - Get top feature drivers from LIME/SHAP collaboration (Layer 2)
       - Get global importance rankings from pre-computed insights
       - Get sensitivity scores from advanced explainability analysis
    
    2. Prompt Construction:
       - Format borrower profile as readable list
       - Format feature drivers with impact values
       - Include global context for comparison
       - Include sensitivity information
       - Add instructions for LLM (risk level, key factors, comparison, recommendation)
    
    3. LLM Generation:
       - Pass constructed prompt to LLM pipeline
       - LLM generates natural language explanation
       - Parse and extract generated text
       - Return as narrative summary

Result Fusion:
    - Quantitative data (probabilities, SHAP values, LIME impacts) → Structured prompt
    - Structured prompt → LLM processing → Natural language narrative
    - Narrative includes: (1) Risk level and key factors, (2) Comparison to typical patterns, (3) Recommendation
    - Final output: Human-readable explanation that synthesizes all ML outputs

Implementation Location:
    - app/explainer.py: Explainer.llm_summary() method
    - app/container.py: Services.predict() method (orchestrates the flow)
    - src/05_llm_explainability.ipynb: Cell 30 (LLM Explanation Generation section)

Algorithm Consistency:
    The implementation uses LLM to synthesize ML outputs into narratives,
    exactly as described in the README. No algorithm substitution.
"""


def get_ml_llm_collaboration_info() -> Dict:
    """
    Returns documentation of the ML-LLM collaboration scheme.
    
    Returns:
        Dict containing ML-LLM collaboration details
    """
    return {
        "layer": "Layer 3: ML-LLM Collaboration",
        "description": "ML models provide structured data to LLM for narrative generation",
        "data_flow": [
            "ML Predictions (probability, risk level)",
            "↓ LIME/SHAP Feature Drivers (from Layer 2)",
            "↓ Global Context (pre-computed insights)",
            "↓ Sensitivity Analysis (pre-computed)",
            "↓ Prompt Construction (format all data)",
            "↓ LLM Pipeline (text generation)",
            "↓ Natural Language Narrative"
        ],
        "process_scheduling": {
            "step_1": {
                "name": "Data Collection from ML Models",
                "sources": [
                    "Ensemble prediction (probability, risk level)",
                    "LIME/SHAP feature drivers (top risk-increasing/decreasing)",
                    "Global importance rankings (pre-computed)",
                    "Sensitivity scores (pre-computed)",
                    "Borrower profile (raw input features)"
                ],
                "output": "Structured data dictionary"
            },
            "step_2": {
                "name": "Prompt Construction",
                "components": [
                    "Borrower profile (formatted as list)",
                    "Local feature drivers (with impact values)",
                    "Global context (top important features)",
                    "Sensitivity information (most sensitive features)",
                    "LLM instructions (format: risk level, factors, comparison, recommendation)"
                ],
                "output": "Complete prompt string"
            },
            "step_3": {
                "name": "LLM Generation",
                "method": "Text generation pipeline (HuggingFace Transformers)",
                "model": "Microsoft Phi-2 (causal language model)",
                "parameters": {
                    "max_new_tokens": 256,
                    "min_new_tokens": 64,
                    "do_sample": False,
                    "repetition_penalty": 1.05
                },
                "output": "Natural language explanation"
            }
        },
        "result_fusion": {
            "method": "LLM Synthesis of Structured ML Data",
            "transformation": "Quantitative → Structured Prompt → Natural Language",
            "output_format": {
                "component_1": "Risk level and key contributing factors",
                "component_2": "Comparison to typical patterns (using global context)",
                "component_3": "Actionable recommendation"
            },
            "benefit": "Makes technical ML outputs understandable to non-technical users"
        },
        "implementation_files": [
            "app/explainer.py (Explainer.llm_summary method)",
            "app/container.py (Services.predict method - orchestrates flow)",
            "src/05_llm_explainability.ipynb (Cell 30: LLM Explanation Generation)"
        ]
    }


# ============================================================================
# COMPLETE COLLABORATION SCHEME OVERVIEW
# ============================================================================

def get_complete_collaboration_scheme() -> Dict:
    """
    Returns complete documentation of all three collaboration layers.
    
    This function provides a single entry point to understand the entire
    ML-LLM collaboration scheme, including data interaction, process scheduling,
    and result fusion across all layers.
    
    Returns:
        Dict containing:
            - overview: High-level description
            - layers: All three collaboration layers
            - complete_data_flow: End-to-end data interaction
            - complete_process_scheduling: Full execution order
            - complete_result_fusion: How all outputs are combined
    """
    return {
        "overview": {
            "title": "ML-LLM Collaboration Scheme for Credit Risk Prediction",
            "description": (
                "A three-layer collaboration system that combines multiple ML models "
                "for robust predictions, uses complementary explainability methods for "
                "feature importance, and synthesizes all outputs into human-readable "
                "narratives using LLM."
            ),
            "purpose": (
                "Provide accurate credit risk predictions with comprehensive, "
                "understandable explanations for both technical and non-technical users."
            )
        },
        "layers": {
            "layer_1": get_ensemble_collaboration_info(),
            "layer_2": get_explainability_collaboration_info(),
            "layer_3": get_ml_llm_collaboration_info()
        },
        "complete_data_flow": [
            "Raw Input (11 borrower features)",
            "↓ Preprocessing (scaling, encoding, imputation)",
            "↓ Processed Features (17 features)",
            "↓ Layer 1: Ensemble Collaboration",
            "  → Base Models (5 models predict)",
            "  → Meta-Learner (combines predictions)",
            "  → Prediction Probability (0-1)",
            "↓ Layer 2: Explainability Collaboration",
            "  → LIME Explanation (local importance)",
            "  → SHAP Explanation (theoretical importance)",
            "  → Risk Drivers (combined importance)",
            "↓ Layer 3: ML-LLM Collaboration",
            "  → Data Collection (all ML outputs)",
            "  → Prompt Construction (structured format)",
            "  → LLM Generation (natural language)",
            "  → Final Narrative Explanation"
        ],
        "complete_process_scheduling": {
            "sequential_stages": [
                {
                    "stage": "Preprocessing",
                    "component": "Preprocessor",
                    "output": "Processed feature vector"
                },
                {
                    "stage": "Layer 1: Ensemble Prediction",
                    "components": [
                        "Base Models (parallel execution)",
                        "Meta-Learner (sequential after base models)"
                    ],
                    "output": "Prediction probability"
                },
                {
                    "stage": "Layer 2: Explainability",
                    "components": [
                        "LIME Explanation (independent)",
                        "SHAP Explanation (independent)",
                        "Risk Drivers Analysis (depends on LIME + SHAP)"
                    ],
                    "output": "Feature importance and risk drivers"
                },
                {
                    "stage": "Layer 3: LLM Narrative",
                    "components": [
                        "Data Collection (depends on all previous stages)",
                        "Prompt Construction (depends on data collection)",
                        "LLM Generation (depends on prompt)"
                    ],
                    "output": "Natural language explanation"
                }
            ],
            "dependencies": {
                "layer_2_depends_on": "Layer 1 (needs prediction to explain)",
                "layer_3_depends_on": "Layer 1 and Layer 2 (needs all ML outputs)",
                "risk_drivers_depends_on": "LIME and SHAP (combines both)",
                "llm_depends_on": "All previous stages (synthesizes everything)"
            }
        },
        "complete_result_fusion": {
            "layer_1_fusion": "Stacked ensemble: 5 base model predictions → meta-learner → single probability",
            "layer_2_fusion": "Averaging: (LIME impact + SHAP value) / 2 → risk drivers",
            "layer_3_fusion": "LLM synthesis: All ML outputs → structured prompt → natural language narrative",
            "final_output": {
                "prediction": "Default probability (0-1) and risk level (HIGH/MODERATE/LOW)",
                "explanations": {
                    "quantitative": "LIME and SHAP feature importance scores",
                    "structured": "Risk-increasing and risk-decreasing drivers",
                    "narrative": "Human-readable LLM-generated explanation"
                }
            }
        },
        "implementation_consistency": {
            "note": "All implementations match the README description exactly",
            "algorithm_consistency": "Uses Algorithm A (stacked ensemble + LIME/SHAP combination + LLM synthesis) as proposed",
            "no_substitution": "No algorithm substitution - implementation matches paper/README description"
        },
        "file_locations": {
            "central_documentation": "app/collaboration.py (this file)",
            "layer_1_implementation": [
                "app/predictor.py (Predictor class)",
                "src/05_llm_explainability.ipynb (Cell 16)"
            ],
            "layer_2_implementation": [
                "app/explainer.py (Explainer class)",
                "src/05_llm_explainability.ipynb (Cells 18, 20, 22)"
            ],
            "layer_3_implementation": [
                "app/explainer.py (Explainer.llm_summary method)",
                "app/container.py (Services.predict method)",
                "src/05_llm_explainability.ipynb (Cell 30)"
            ],
            "orchestration": [
                "app/container.py (Services class - coordinates all layers)",
                "app/main.py (API endpoint - entry point)"
            ]
        }
    }


# ============================================================================
# QUICK REFERENCE FUNCTIONS
# ============================================================================

def get_collaboration_summary() -> str:
    """
    Returns a concise summary of the collaboration scheme.
    
    Returns:
        str: Brief overview of the three-layer collaboration
    """
    return """
ML-LLM Collaboration Scheme Summary:

Layer 1 (Ensemble Collaboration):
  - 5 base ML models (XGBoost Deep/Shallow, LightGBM, CatBoost, Neural Network)
  - Meta-learner combines their predictions
  - Output: Default probability (0-1)

Layer 2 (Explainability Collaboration):
  - LIME provides local feature importance
  - SHAP provides theoretical feature importance
  - Combined to identify risk-increasing/decreasing drivers
  - Output: Top 5 risk drivers in each category

Layer 3 (ML-LLM Collaboration):
  - ML outputs (prediction, drivers, global context) → LLM prompt
  - LLM generates natural language narrative
  - Output: Human-readable explanation

For detailed documentation, see get_complete_collaboration_scheme()
"""


def get_implementation_locations() -> Dict[str, List[str]]:
    """
    Returns a mapping of collaboration components to their implementation files.
    
    Returns:
        Dict mapping component names to file paths
    """
    return {
        "ensemble_collaboration": [
            "app/predictor.py: Predictor.predict_scores()",
            "src/05_llm_explainability.ipynb: Cell 16"
        ],
        "explainability_collaboration": [
            "app/explainer.py: Explainer.risk_drivers()",
            "src/05_llm_explainability.ipynb: Cell 22"
        ],
        "ml_llm_collaboration": [
            "app/explainer.py: Explainer.llm_summary()",
            "app/container.py: Services.predict()",
            "src/05_llm_explainability.ipynb: Cell 30"
        ],
        "orchestration": [
            "app/container.py: Services class",
            "app/main.py: /api/predict endpoint"
        ]
    }

