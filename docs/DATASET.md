# Dataset Information

## Dataset Overview

The credit risk dataset contains information about loan applicants and their loan outcomes. This dataset is used to train and evaluate credit risk prediction models.

## Dataset Details

- **Name**: Credit Risk Dataset
- **Format**: CSV (Comma-Separated Values)
- **Location**: `dataset/credit_risk_dataset.csv`
- **Size**: ~32,581 samples
- **Features**: 11 input features + 1 target variable

## Feature Schema

### Numeric Features

1. **person_age** (int)
   - Range: 18-100
   - Description: Age of the loan applicant
   - Unit: Years

2. **person_income** (int)
   - Range: >= 1000
   - Description: Annual income of the applicant
   - Unit: Currency units

3. **person_emp_length** (float)
   - Range: 0-60
   - Description: Employment length in years
   - Unit: Years
   - Note: May contain missing values

4. **loan_amnt** (int)
   - Range: 500-1,000,000
   - Description: Loan amount requested
   - Unit: Currency units

5. **loan_int_rate** (float)
   - Range: 0-100
   - Description: Interest rate on the loan
   - Unit: Percentage

6. **loan_percent_income** (float)
   - Range: 0-1
   - Description: Loan amount as percentage of income
   - Unit: Ratio (0.0 to 1.0)

7. **cb_person_cred_hist_length** (int)
   - Range: 0-50
   - Description: Length of credit history
   - Unit: Years

### Categorical Features

8. **person_home_ownership** (str)
   - Values: `RENT`, `OWN`, `MORTGAGE`, `OTHER`
   - Description: Home ownership status

9. **loan_intent** (str)
   - Description: Purpose of the loan
   - Examples: `VENTURE`, `EDUCATION`, `MEDICAL`, `PERSONAL`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`

10. **loan_grade** (str)
    - Values: `A`, `B`, `C`, `D`, `E`, `F`, `G`
    - Description: Loan grade assigned by lender (ordinal: A is best, G is worst)

11. **cb_person_default_on_file** (str)
    - Values: `Y` (Yes), `N` (No)
    - Description: Whether person has defaulted before

### Target Variable

12. **loan_status** (int)
    - Values: `0` (No default), `1` (Default)
    - Description: Binary classification target
    - `0`: Loan repaid successfully
    - `1`: Loan defaulted

## Data Preprocessing

The dataset undergoes the following preprocessing steps:

1. **Missing Value Imputation**:
   - KNN Imputer for numeric features (k=5 neighbors)
   - Uses relationships between features to impute missing values

2. **Feature Scaling**:
   - RobustScaler for numeric features (median-based, outlier-resistant)

3. **Categorical Encoding**:
   - **Ordinal**: `loan_grade` → LabelEncoder (preserves order)
   - **Nominal**: `person_home_ownership`, `loan_intent` → OneHotEncoder
   - **Binary**: `cb_person_default_on_file` → Simple mapping (Y→1, N→0)

4. **Feature Alignment**:
   - Features are aligned with training data feature order
   - Missing features are filled with zeros

## Dataset Files

The project uses several processed dataset files:

- `credit_risk_dataset.csv`: Raw dataset
- `X_train.pkl`: Preprocessed training features (pandas DataFrame)
- `X_test.pkl`: Preprocessed test features (pandas DataFrame)
- `y_train.pkl`: Training target labels
- `y_test.pkl`: Test target labels

## Data Distribution

- **Total Samples**: 32,581
- **Training Set**: ~26,065 (80%)
- **Test Set**: ~6,516 (20%)
- **Class Distribution**: Imbalanced (more non-defaults than defaults)

## Download Links

If you need to download or access the dataset:

1. **Primary Dataset**: 
   - **Kaggle**: https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data
   - **Local Path**: `dataset/credit_risk_dataset.csv` (if already downloaded)
   - **Note**: You'll need a Kaggle account to download. After downloading, place the CSV file in the `dataset/` folder.

2. **Processed Data**: Generated automatically by running `src/02_preprocessing_feature_engineering.ipynb`
   - The preprocessing notebook will create `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, and `y_test.pkl` in the `dataset/` folder

## Usage in Training

The dataset is used in the following notebooks:

1. **01_eda.ipynb**: Exploratory data analysis
2. **02_preprocessing_feature_engineering.ipynb**: Preprocessing and feature engineering
3. **03_traditional_machineLearning.ipynb**: Traditional ML model training
4. **03_2_neuralNetworkModel.ipynb**: Neural network model training
5. **04_explainability.ipynb**: Explainability analysis
6. **04b_advanced_explainability.ipynb**: Advanced explainability
7. **05_llm_explainability.ipynb**: LLM-based explanations

## Data Quality

- **Missing Values**: Present in `person_emp_length` (handled via imputation)
- **Outliers**: Present in numeric features (handled via clipping and robust scaling)
- **Class Imbalance**: Addressed through appropriate evaluation metrics (ROC-AUC, PR-AUC)

## Privacy and Ethics

- This dataset contains synthetic or anonymized financial data
- No personally identifiable information (PII) should be present
- Use responsibly and in accordance with data privacy regulations

## Citation

If you use this dataset, please cite appropriately based on the original source.
https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data
