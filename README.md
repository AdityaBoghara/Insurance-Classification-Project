# 🚗 Car Insurance Cross-Sell Prediction (XGBoost)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model%20Evaluation-yellow?logo=scikit-learn)
![License](https://img.shields.io/badge/License-Academic-blue)

---

## 📘 Project Overview

This repository contains the final course project for **BZAN 6357 (Machine Learning & Data Mining)** at the University of Houston.

The goal is to predict whether a customer will **purchase a car insurance policy** when contacted by a sales representative, using supervised machine learning techniques.

---

## 🎯 Objective

Develop and evaluate machine learning models that classify customers as likely to **purchase (1)** or **not purchase (0)** car insurance based on their demographic and behavioral data.

Model performance is evaluated primarily using **ROC-AUC** and **F1-score**, ensuring a balance between recall (catching potential buyers) and precision (avoiding false positives).

---

## 🧩 Dataset Description

### Training Dataset: `bzan6357_insurance_3_TRAINING.csv`

### Scoring Dataset: `bzan6357_insurance_3_SCORE.csv`

| Column            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `id_new`          | Unique customer ID                                |
| `buy`             | Target variable (1 = purchased, 0 = no purchase)  |
| `age`             | Customer's age                                    |
| `gender`          | Gender of the policyholder (male/female)          |
| `tenure`          | Days since start of medical-insurance policy      |
| `region`          | Numeric regional code (many discrete values)      |
| `dl`              | Has valid driver license (1 = yes, 0 = no)        |
| `has_v_insurance` | Already has valid car insurance (1 = yes, 0 = no) |
| `v_age`           | Vehicle age: "< 1 year", "1–2 years", "> 2 years" |
| `v_accident`      | Had accident before (yes/no)                      |
| `v_prem_quote`    | Annual premium quote (local currency)             |
| `cs_rep`          | Unique ID of customer service representative      |

---

## ⚙️ Data Preparation & EDA

* Inspected and cleaned the dataset; confirmed no missing values.
* Handled class imbalance (~82% no-purchase vs 18% purchase).
* Encoded categorical features (`gender`, `v_age`, `v_accident`) with Label Encoding.
* Converted high-cardinality features (`region`, `cs_rep`) to numeric form.
* Visualized feature distributions and correlation heatmap.
* Split data into **training** and **validation** sets using **Stratified K-Fold CV**.

---

## 🤖 Model Development

* Implemented **XGBoostClassifier** inside a pipeline with **SMOTE** for oversampling.
* Tuned hyperparameters via **RandomizedSearchCV**:

  ```python
  {
      'n_estimators': [200, 300, 400, 500],
      'max_depth': [4, 5, 6, 7],
      'learning_rate': [0.01, 0.03, 0.05, 0.1],
      'subsample': [0.7, 0.8, 0.9, 1.0],
      'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
      'scale_pos_weight': [imbalance_ratio]
  }
  ```
* Evaluated models using metrics:

  * **ROC-AUC** (primary)
  * **F1-score** (secondary)
  * **Precision**, **Recall**, and **Accuracy**
* Final model selected based on balanced performance.

---

## 📈 Results

| Metric       | Validation Score |
| ------------ | ---------------- |
| **ROC-AUC**  | ~0.89            |
| **F1-score** | ~0.88            |
| **Accuracy** | ~88%             |

### 🔑 Key Influential Features

* **Gender**: Females slightly more likely to purchase.
* **Vehicle Age**: 1–2-year-old vehicles show higher conversion.
* **Premium Quote**: Higher quotes correlate with stronger intent.
* **Tenure**: Long-term policyholders are more responsive.
* **Driver License**: Valid licenses strongly linked to purchases.

---

## 🧠 Insights & Business Recommendations

* Prioritize **medium-tenure customers** with **higher premium quotes** for marketing campaigns.
* Offer loyalty-based **discounts** to long-term customers.
* Optimize training for **customer service representatives (cs_rep)** with below-average conversion.
* Extend to **ensemble stacking** or **SHAP explainability** for further model transparency.

---

## 🧮 Deliverables

| File                                             | Description                                                  |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `Project_XGBoost.ipynb` / `Project_XGBoost.html` | Full notebook with code, outputs, and markdown explanations. |
| `Scored_Dataset.csv`                             | Contains only `id_new`, `probability`, and `classification`. |

---

## 🛠️ Tech Stack

**Languages & Libraries**
Python 3 · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · XGBoost · Imbalanced-learn

**Environment**
Jupyter Notebook

---

## 🏁 Conclusion

The tuned **XGBoost model** achieved a strong and balanced trade-off between **precision** and **recall**, effectively identifying potential car insurance buyers.
This approach demonstrates how machine learning can optimize cross-selling strategies and improve campaign targeting efficiency.

---

## 👩‍💻 Author

**Aditya Boghara**
Master of Science in Management Information Systems
University of Houston, C.T. Bauer College of Business
📧 [arboghara@uh.edu](mailto:arboghara@uh.edu)

---

> *This project was developed for academic purposes as part of the BZAN 6357 course under Dr. Xiao Ma. All data used is anonymized and proprietary to the course.*
