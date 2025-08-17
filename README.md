# Churn Prediction Project

## Overview
This project predicts customer churn for a telecom company using machine learning models. It identifies which customers are likely to leave (churn) and highlights key factors driving churn. The project demonstrates **data cleaning, exploratory analysis, predictive modeling, and model interpretation** using SHAP.

---

## Dataset
- Source: [Telco Customer Churn – Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Records: 7,043 customers
- Features: Demographics, services, account information, and usage metrics

---

## Tools & Libraries
- **Python**: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn
- Optional: Tableau/Power BI for visualization

---

## Steps

### 1. Data Loading & Cleaning
- Loaded CSV and checked column types
- Converted `TotalCharges` to numeric
- Handled missing values
- Dropped `customerID` (not needed for modeling)

### 2. Feature Encoding
- Binary features (Yes/No, Male/Female) → 0/1
- Multi-class categorical features → One-hot encoding
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`

### 3. Exploratory Data Analysis (EDA)
- Visualized churn distribution by tenure, contract type, and other features
- Created tenure buckets for better insights

### 4. Model Training
- Split dataset into 80% train, 20% test
- Trained models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluated using **accuracy, precision, recall, F1-score**

### 5. Model Interpretation
- Used **SHAP** to explain predictions
- Generated:
  - Summary plot (overall feature importance)
  - Bar plot (average feature importance)
  - Dependence plots (feature impact on churn probability)

---

## Key Insights
- Short tenure, month-to-month contracts, and high monthly charges increase churn probability
- Customers with longer tenure and annual contracts are less likely to churn
- Actionable strategies:
  - Offer retention plans to high-risk customers
  - Provide incentives for month-to-month customers
  - Monitor customers with high monthly charges closely

---

## Results
- XGBoost Accuracy: ~79%
- Random Forest Acuuracy - 80%
- Logistic Regression Accuracy - 82%
- Identified top churn drivers using SHAP
- Visualizations provide clear understanding for decision-making

---

## How to Run
1. Clone the repository
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
