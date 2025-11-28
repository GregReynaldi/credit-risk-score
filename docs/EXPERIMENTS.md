# Experiment Logs and Results

This document summarizes the model training experiments, performance metrics, and key findings from the credit risk prediction project.

## Experiment Overview

The project involved training multiple machine learning models using different algorithms and architectures, ultimately combining them into a stacked ensemble for optimal performance.

## Evaluation Scripts Location

**Evaluation code is integrated into the training notebooks** for seamless workflow:

- **Traditional ML Evaluation**: `src/03_traditional_machine_learning.ipynb`
  - Evaluation cells compute ROC-AUC, PR-AUC, F1, Precision, Recall, Specificity
  - Generate confusion matrices, ROC curves, and Precision-Recall curves
  - Save results to `models/traditional_ml_results.json`

- **Neural Network Evaluation**: `src/03_2_neural_network_model.ipynb`
  - Comprehensive evaluation for each model architecture (TabNet, DCN, Residual, Ensemble)
  - All metrics computed on both training and test sets
  - Visualizations and comparison plots generated
  - Results saved to `models/neural_network_results.json`

All evaluation metrics follow standard practices and are fully reproducible with the provided random seeds.

## Traditional Machine Learning Models

### Models Tested

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

### Performance Results

| Model | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Optimal Threshold |
|-------|--------------|-------------|---------|----------------|-------------|-------------------|
| **Random Forest** | 0.9396 | 0.8948 | 0.8356 | 0.9599 | 0.7398 | 0.44 |
| **XGBoost** | 0.9358 | 0.8867 | 0.8241 | 0.9787 | 0.7117 | 0.68 |
| **Logistic Regression** | 0.8663 | 0.6956 | 0.6454 | 0.6248 | 0.6674 | 0.64 |

### Key Findings

- **Random Forest** achieved the best overall performance with highest ROC-AUC (0.9396) and F1 score (0.8356)
- **XGBoost** showed excellent precision (0.9787) but slightly lower recall
- **Logistic Regression** provided a baseline but was outperformed by tree-based methods

### Best Hyperparameters

**Logistic Regression**:
- C: 0.1
- Penalty: L1

**Random Forest**:
- max_depth: None (unlimited)
- min_samples_leaf: 1
- min_samples_split: 2

**XGBoost**:
- max_depth: 7
- min_child_weight: 5
- gamma: 0.3
- reg_lambda: 1.0

## Neural Network Models

### Models Tested

1. **Pure TabNet**
2. **TabNet + Tokenizer**
3. **Deep & Cross Network (DCN)**
4. **Residual Neural Network**
5. **Multi-Scale Ensemble** (Final)

### Performance Results

| Model | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Optimal Threshold |
|-------|--------------|-------------|---------|----------------|-------------|-------------------|
| **Multi-Scale Ensemble** | **0.9540** | **0.9140** | **0.8376** | **0.9558** | **0.7454** | 0.32 |
| Residual Neural Network | 0.9313 | 0.8766 | 0.8052 | 0.9165 | 0.7180 | 0.32 |
| Deep & Cross Network | 0.9241 | 0.8663 | 0.7962 | 0.9222 | 0.7004 | 0.28 |
| Pure TabNet | 0.9218 | 0.8608 | 0.7946 | 0.9180 | 0.7004 | 0.77 |
| TabNet + Tokenizer | 0.9121 | 0.8496 | 0.7881 | 0.8775 | 0.7152 | 0.65 |

### Key Findings

- **Multi-Scale Ensemble** achieved the best performance across all metrics
- **Residual Neural Network** showed strong performance with good balance
- **TabNet** models demonstrated competitive results with interpretability features
- **Deep & Cross Network** provided good baseline for deep learning approaches

## Final Ensemble Architecture

The production system uses a **stacked ensemble** combining:

### Base Models (Level 1)
1. XGBoost Deep
2. XGBoost Shallow
3. LightGBM Fast
4. CatBoost Robust
5. Neural Network (Advanced)

### Meta-Learner (Level 2)
- Neural network that learns optimal combination of base model predictions

### Final Performance

- **Test ROC-AUC**: 0.9540
- **Test PR-AUC**: 0.9140
- **Test F1**: 0.8376
- **Test Precision**: 0.9558
- **Test Recall**: 0.7454
- **Test Specificity**: 0.9904

## Explainability Analysis Results

### SHAP Analysis
- Identified top globally important features
- Generated feature importance rankings
- Computed SHAP values for individual predictions

### LIME Analysis
- Generated local explanations for predictions
- Identified feature contributions per instance
- Created feature importance rankings

### Combined Analysis
- Merged SHAP and LIME importance scores
- Identified consensus features across methods
- Generated risk driver rankings

### Advanced Explainability
- **Sensitivity Analysis**: Identified most sensitive features
- **Model Agreement**: Analyzed consensus across ensemble models
- **Decision Boundaries**: Visualized classification boundaries
- **Feature Sensitivity Curves**: Analyzed prediction sensitivity to feature changes

## Key Insights

### Model Performance
1. **Ensemble outperforms individual models**: The stacked ensemble achieved 0.9540 ROC-AUC, significantly better than any single model
2. **Tree-based models excel**: Random Forest and XGBoost showed strong performance on tabular data
3. **Neural networks add value**: Deep learning models captured complex feature interactions
4. **Meta-learning improves generalization**: The meta-learner effectively combined base model strengths

### Feature Importance
- Top important features identified through multiple explainability methods
- Consistent patterns across different models
- Some features show high sensitivity (small changes cause large prediction shifts)

### Model Robustness
- High specificity (0.9904) indicates low false positive rate
- Good precision (0.9558) means reliable positive predictions
- Balanced recall (0.7454) captures most defaults

## Experiment Timeline

1. **Phase 1**: Exploratory Data Analysis
   - Dataset exploration and visualization
   - Feature distribution analysis
   - Correlation analysis

2. **Phase 2**: Preprocessing
   - Missing value imputation
   - Feature scaling and encoding
   - Train/test split

3. **Phase 3**: Traditional ML Training
   - Logistic Regression baseline
   - Random Forest optimization
   - XGBoost tuning

4. **Phase 4**: Neural Network Training
   - TabNet experiments
   - Deep & Cross Network
   - Residual networks
   - Ensemble construction

5. **Phase 5**: Explainability
   - SHAP analysis
   - LIME explanations
   - Advanced explainability
   - LLM integration

## Model Artifacts

All trained models and results are saved in:
- `models/traditional_ml_results.json`: Traditional ML performance metrics
- `models/neural_network_results.json`: Neural network performance metrics
- `models/multiscale_base_models.joblib`: Base ensemble models
- `models/multiscale_meta_learner.h5`: Meta-learner model
- `models/neural_network_advanced.h5`: Neural network base model

## Reproducibility

To reproduce these results, use the following settings:

- **Random seed**: 42 (set at the beginning of each notebook)
- **Cross-validation**: 5-fold for hyperparameter tuning
- **Train/Test split**: 80/20 (stratified to maintain class distribution)
- **Python version**: 3.10 recommended (3.8+ required)
- **Dependencies**: Install from `requirements.txt` or `environment.yml`

**Important**: Run the notebooks in order (01 → 02 → 03 → 03_2) to ensure data preprocessing and model training are consistent. The random seed ensures that data splits and model initialization are reproducible.

## Future Improvements

Potential areas for enhancement:
1. Feature engineering: Additional derived features
2. Hyperparameter optimization: More extensive grid search
3. Model diversity: Additional base models
4. Explainability: Enhanced LLM prompts
5. Performance: Model quantization for faster inference

