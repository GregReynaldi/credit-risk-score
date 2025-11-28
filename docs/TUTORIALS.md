# Module Tutorials

This document provides step-by-step tutorials for running and understanding the key modules in this project.

## Tutorial 1: Exploratory Data Analysis (EDA)

**Notebook**: `src/01_eda.ipynb`

### Purpose
Perform comprehensive exploratory data analysis to understand the dataset, identify patterns, detect issues, and inform preprocessing decisions.

### Prerequisites
- Python environment with dependencies installed
- Dataset file: `dataset/credit_risk_dataset.csv`
- Jupyter Notebook or JupyterLab

### Step-by-Step Guide

1. **Open the notebook**:
   ```bash
   jupyter notebook src/01_eda.ipynb
   ```

2. **Run data loading cell**:
   - Loads the CSV dataset
   - Displays basic information (shape, memory usage, data types)

3. **Execute data quality assessment**:
   - Identifies numeric and categorical columns
   - Checks for missing values
   - Analyzes data distributions

4. **Run univariate analysis**:
   - Distribution plots for numeric features
   - Frequency plots for categorical features
   - Target variable distribution

5. **Execute bivariate analysis**:
   - Correlation matrix
   - Feature-target relationships
   - Pair plots for key features

6. **Run advanced analysis**:
   - Outlier detection and visualization
   - PCA analysis
   - Feature importance analysis

7. **Review outputs**:
   - All visualizations saved to `artifacts/01_eda_images/`
   - Summary statistics in `artifacts/01_eda_images/eda_summary.json`

### Expected Outputs
- Distribution plots for all features
- Correlation heatmap
- Missing values visualization
- Outlier detection plots
- Feature importance rankings
- EDA summary JSON file

### Key Insights to Look For
- Class imbalance in target variable
- Missing value patterns
- Outlier presence
- Feature correlations
- Distribution shapes

---

## Tutorial 2: Training Models

### Part A: Traditional Machine Learning

**Notebook**: `src/03_traditional_machineLearning.ipynb`

#### Purpose
Train and evaluate traditional machine learning models (Logistic Regression, Random Forest, XGBoost) for credit risk prediction.

#### Prerequisites
- Completed preprocessing (`02_preprocessing_feature_engineering.ipynb`)
- Preprocessed data files: `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`

#### Step-by-Step Guide

1. **Open the notebook**:
   ```bash
   jupyter notebook src/03_traditional_machineLearning.ipynb
   ```

2. **Load preprocessed data**:
   - Loads training and test sets
   - Verifies data shapes and distributions

3. **Train Logistic Regression**:
   - Hyperparameter tuning with cross-validation
   - Model training
   - Evaluation metrics calculation

4. **Train Random Forest**:
   - Grid search for optimal parameters
   - Model training with best parameters
   - Feature importance extraction

5. **Train XGBoost**:
   - Hyperparameter optimization
   - Model training
   - Performance evaluation

6. **Model comparison**:
   - Compare all models on test set
   - Generate ROC curves
   - Create confusion matrices
   - Save results to `models/traditional_ml_results.json`

#### Expected Outputs
- Trained model files (`.joblib` format)
- Performance metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
- Visualization plots (ROC curves, confusion matrices, feature importance)
- Results JSON file

### Part B: Neural Network Models

**Notebook**: `src/03_2_neuralNetworkModel.ipynb`

#### Purpose
Train deep learning models (TabNet, DCN, Residual Networks) and create a stacked ensemble.

#### Prerequisites
- Completed traditional ML training
- TensorFlow/Keras installed
- PyTorch installed (for TabNet)

#### Step-by-Step Guide

1. **Open the notebook**:
   ```bash
   jupyter notebook src/03_2_neuralNetworkModel.ipynb
   ```

2. **Train TabNet models**:
   - Pure TabNet architecture
   - TabNet with tokenizer
   - Save trained models

3. **Train Deep & Cross Network (DCN)**:
   - Define DCN architecture
   - Train with early stopping
   - Evaluate performance

4. **Train Residual Network**:
   - Build residual architecture
   - Train model
   - Evaluate results

5. **Create ensemble**:
   - Combine base models (XGBoost, LightGBM, CatBoost, Neural Network)
   - Train meta-learner
   - Evaluate ensemble performance

6. **Save models**:
   - Save all neural network models (`.h5` format)
   - Save base models bundle (`.joblib` format)
   - Save meta-learner
   - Save results to `models/neural_network_results.json`

#### Expected Outputs
- Neural network model files (`.h5`, `.zip`)
- Ensemble base models (`.joblib`)
- Meta-learner model (`.h5`)
- Performance metrics and visualizations
- Results JSON file

---

## Tutorial 3: Generating Explainability

### Part A: Basic Explainability

**Notebook**: `src/04_explainability.ipynb`

#### Purpose
Generate LIME and SHAP explanations for model predictions, analyze feature importance, and create explainability visualizations.

#### Prerequisites
- Trained ensemble models
- SHAP and LIME libraries installed

#### Step-by-Step Guide

1. **Open the notebook**:
   ```bash
   jupyter notebook src/04_explainability.ipynb
   ```

2. **Load models and data**:
   - Load ensemble models
   - Load training data for background samples

3. **Generate SHAP explanations**:
   - Create SHAP explainer
   - Compute SHAP values for test samples
   - Generate feature importance rankings
   - Create SHAP visualizations

4. **Generate LIME explanations**:
   - Create LIME explainer with background data
   - Explain individual predictions
   - Extract feature contributions
   - Generate LIME visualizations

5. **Combine explanations**:
   - Merge SHAP and LIME importance scores
   - Create combined feature importance rankings
   - Identify consensus features

6. **Save results**:
   - Save SHAP importance to `artifacts/04_shap_images/shap_feature_importance.json`
   - Save LIME importance to `artifacts/04_shap_images/lime_feature_importance.json`
   - Save combined importance
   - Save visualization plots

#### Expected Outputs
- SHAP feature importance JSON
- LIME feature importance JSON
- Combined importance rankings
- SHAP plots (waterfall, summary, etc.)
- LIME explanation plots

### Part B: Advanced Explainability

**Notebook**: `src/04b_advanced_explainability.ipynb`

#### Purpose
Perform advanced explainability analysis including sensitivity analysis, model agreement, decision boundaries, and consistency checks.

#### Prerequisites
- Completed basic explainability analysis
- Ensemble models trained

#### Step-by-Step Guide

1. **Open the notebook**:
   ```bash
   jupyter notebook src/04b_advanced_explainability.ipynb
   ```

2. **Sensitivity analysis**:
   - Perturb features and measure prediction changes
   - Identify most sensitive features
   - Generate sensitivity curves

3. **Model agreement analysis**:
   - Compare feature importance across base models
   - Identify consensus features
   - Create agreement heatmaps

4. **Decision boundary visualization**:
   - Apply PCA for dimensionality reduction
   - Visualize decision boundaries
   - Analyze risk threshold regions

5. **Explanation consistency**:
   - Check consistency of explanations across similar samples
   - Analyze explanation stability
   - Generate consistency reports

6. **Save advanced results**:
   - Save to `artifacts/04b_images/advanced_explainability_report.json`
   - Save visualizations (sensitivity curves, heatmaps, decision boundaries)

#### Expected Outputs
- Advanced explainability report JSON
- Sensitivity analysis results
- Model agreement heatmaps
- Decision boundary visualizations
- Explanation consistency metrics

---

## Tutorial 4: LLM Explainability

**Notebook**: `src/05_llm_explainability.ipynb`

### Purpose
Generate human-readable narrative explanations using Large Language Models (LLMs) that combine structured explainability data into natural language.

### Prerequisites
- Completed explainability analysis
- LLM model files in `models/LLM_MODEL/` (optional)
- Transformers library installed

### Step-by-Step Guide

1. **Open the notebook**:
   ```bash
   jupyter notebook src/05_llm_explainability.ipynb
   ```

2. **Load models and explainers**:
   - Load ensemble models
   - Load LIME and SHAP explainers
   - Initialize LLM pipeline (if available)

3. **Generate structured explanations**:
   - Get LIME explanations for sample predictions
   - Get SHAP explanations
   - Identify risk drivers

4. **Create LLM prompts**:
   - Combine prediction results
   - Include feature importance
   - Add global context
   - Format for LLM input

5. **Generate narratives**:
   - Use LLM to generate explanations
   - Format output for readability
   - Handle LLM errors gracefully

6. **Evaluate explanations**:
   - Review narrative quality
   - Check for accuracy
   - Ensure actionable insights

### Expected Outputs
- LLM-generated narrative explanations
- Comparison of structured vs. narrative explanations
- Example explanations for different risk levels

### Notes
- LLM is optional - system works without it
- LLM explanations are slower than structured explanations
- Quality depends on LLM model and prompt engineering

---

## General Tips for All Tutorials

### Running Notebooks
1. **Execute cells sequentially**: Some cells depend on previous outputs
2. **Check outputs**: Verify each cell completes successfully
3. **Review warnings**: Address any warnings or errors
4. **Save frequently**: Save notebook after completing major sections

### Troubleshooting
- **Memory errors**: Reduce sample sizes or use smaller models
- **Import errors**: Ensure all dependencies are installed
- **File not found**: Check file paths and ensure data files exist
- **Model loading errors**: Verify model files are in correct locations

### Best Practices
1. **Read documentation**: Understand what each cell does before running
2. **Modify parameters**: Experiment with different hyperparameters
3. **Save results**: Keep track of experiment results
4. **Document findings**: Note interesting insights or issues

### Expected Runtime
- **EDA**: 5-10 minutes
- **Traditional ML Training**: 10-30 minutes (depending on hyperparameter search)
- **Neural Network Training**: 30-60 minutes (depending on model complexity)
- **Explainability**: 15-30 minutes
- **Advanced Explainability**: 20-40 minutes
- **LLM Explainability**: 5-15 minutes (if LLM available)

---

## Quick Reference Commands

### Start Jupyter Notebook
```bash
jupyter notebook
```

### Run specific notebook
```bash
jupyter notebook src/01_eda.ipynb
```

### Convert notebook to script (optional)
```bash
jupyter nbconvert --to script src/01_eda.ipynb
```

### Check notebook execution status
- Green cell number: Successfully executed
- Yellow cell number: Currently executing
- Red cell number: Error occurred

---

For more detailed information about specific modules, refer to the code comments and docstrings in each notebook.

