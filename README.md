# 🏦 End-to-End Bank Customer Churn Prediction Pipeline & Streamlit App

A full-stack, machine learning solution designed to predict customer churn risk using advanced gradient-boosted decision trees. This repository covers data cleaning, target leakage resolution, feature engineering, automated hyperparameter optimization focused on class imbalance, and an interactive frontend dashboard deployment.

## 🚀 Key Features & Architectural Pipeline

1. **Target Leakage Remediation**: Identified and eliminated data leakage issues stemming from raw historical complaint logs, successfully stabilizing the validation metrics back to real-world production levels.
2. **Feature Engineering & Dimensionality Reduction**: 
   - Created robust engineered signals: `Balance_Per_Product`, `Is_Zero_Balance`, `Age_Group`, `Germany_Female`, and `Products_Active`.
   - Dropped the raw continuous columns `Age` and `Balance` to force the model to rely on generalized engineered intervals, mitigating overfitting while training tree-based ensembles.
3. **Imbalance Handling**: Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively within the training folds to account for the minority churn class.
4. **Bayesian Hyperparameter Optimization**: Implemented **Optuna** to maximize the positive class **F1-Score** directly, adjusting constraints like leaves, learning rates, and depth across 40 distinct trials.
5. **Native Early Stopping**: Leveraged LightGBM's native evaluation callback systems to prune stale branches automatically during tree generation once validation log-loss plateaued for 30 consecutive rounds.
6. **Experiment Tracking**: Managed parameters, metrics, and trial thresholds smoothly through **MLflow**.
7. **Frontend Dashboard Deployment**: Wrapped the serialized model (`.pkl`) and isolated scaler transforms into a responsive **Streamlit** user interface.

---

## 💻 Tech Stack & Dependencies

- **Core Frameworks**: Python 3.x, Pandas, NumPy, Scikit-Learn
- **Boosting Architecture**: LightGBM
- **Optimization Tools**: Optuna, Imbalanced-Learn (SMOTE)
- **M LOps & Serialization**: MLflow, Joblib
- **Deployment Platform**: Streamlit

---

## 📊 Project Repository Structure

```text
├── .gitignore                      # Standard Python, MLflow, and workspace ignore file
├── Bank_Churn.ipynb                 # Core Jupyter Notebook pipeline (EDA, training, tuning)
├── Customer-Churn-Records.csv       # Raw input dataset
├── Customer_Churn_Cleaned.csv       # Preprocessed and validated data output
├── app.py                          # Interactive Streamlit frontend script
├── lightgbm_churn_model.pkl         # Serialized, Optuna-optimized LightGBM weights
├── scaler.pkl                      # Fitted standard scaler for model features
└── requirements.txt                # Production dependency manifest
