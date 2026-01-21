# 🏥 Diabetes Prediction: A Comparative Analysis

**A Data Science project demonstrating how advanced preprocessing and model selection drive predictive performance in medical diagnostics.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Overview
Diagnosing diabetes requires analyzing complex medical metrics. In this project, I built a machine learning pipeline to predict whether a patient has diabetes based on the **Pima Indians Diabetes Database**.

**Key Objective:** To demonstrate that **Data Quality** (cleaning & imputation) and **Feature Scaling** are just as important as the choice of algorithm.

## 🚀 Key Features
* **Advanced Data Cleaning:** Identified and handled biologically impossible values (e.g., 0 Glucose/Insulin) that act as "silent" missing data.
* **Grouped Imputation:** Filled missing values using median statistics grouped by the target class (`Outcome`), preserving the distinction between healthy and diabetic distributions.
* **Feature Scaling:** Applied `StandardScaler` to distance-based models (KNN, SVM) to prevent high-magnitude features (like Insulin) from dominating the prediction.
* **Model Comparison:** Benchmarked 5 different algorithms, from linear baselines to state-of-the-art boosting.

## 🏆 Model Performance
I compared five models on a stratified test set (20% of data). **XGBoost** achieved the highest accuracy, followed closely by Random Forest.

| Model | Accuracy | F1 Score | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **93.51%** | **90.00%** | **Best Performer** (Handles non-linearity well) |
| Random Forest | 90.91% | 86.79% | Very robust to outliers |
| KNN | 90.26% | 85.44% | **High Score** (Proves Scaling worked!) |
| SVM | 89.61% | 85.19% | Solid performance on small data |
| Logistic Regression | 78.57% | 64.52% | Baseline (Linear boundary insufficient) |

> **Visual Comparison:**
> ![Results Chart](./results_chart.png)
> *(Make sure to save the bar chart from your notebook as `results_chart.png` and upload it to your repo!)*

## 🧠 Technical Approach

### 1. The "Zero" Trap
Many datasets contain `0` as a placeholder for missing data.
* **Problem:** A Glucose level of 0 is biologically impossible (coma/death).
* **Solution:** I replaced zeros with `NaN` in specific columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`).

### 2. Grouped Imputation
Instead of a simple mean/median fill:
```python
# Imputing Insulin based on whether the patient is Healthy or Diabetic
data['Insulin'] = data['Insulin'].fillna(data.groupby('Outcome')['Insulin'].transform('median'))