from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
LIME_BACKGROUND = MODELS_DIR / "lime_background.npy"
LLM_DIR = MODELS_DIR / "LLM_MODEL"

# Explainability artifact paths
SHAP_DIR = ARTIFACTS_DIR / "04_shap_images"
ADV_DIR = ARTIFACTS_DIR / "04b_images"

# Default sample size for quickly rebuilding LIME background when cache missing
LIME_SAMPLE_SIZE = 400

# SHAP background size for fast computation
SHAP_BACKGROUND_SIZE = 50

# Risk thresholds for UI + API
HIGH_RISK_THRESHOLD = 0.5
MODERATE_RISK_THRESHOLD = 0.3

