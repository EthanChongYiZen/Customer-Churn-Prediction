# 📊 Customer Churn Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A complete end-to-end machine learning project that predicts whether a customer will churn (leave a service) using supervised classification. This project covers the full data science workflow — from business problem definition and exploratory data analysis all the way through model training, hyperparameter tuning, and actionable business recommendations.

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Workflow Overview](#-workflow-overview)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [Business Recommendations](#-business-recommendations)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)
- [Future Work](#-future-work)

---

## 🏢 Business Problem

Customer churn — when a customer stops doing business with a company — is one of the most costly problems in subscription-based industries (telecom, SaaS, banking, streaming). Acquiring a new customer typically costs **5–25× more** than retaining an existing one.

**Goal:** Build a binary classifier to predict whether a customer will churn (`Yes` / `No`) using demographic and account data, enabling the business to intervene before the customer leaves.

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| File | `customer_churn_data.csv` |
| Rows | 1,000 customers |
| Features | 10 (9 input + 1 target) |
| Target | `Churn` (Yes / No) |
| Missing Values | `InternetService` (29.7%) |
| Class Distribution | ~88.3% Churn, ~11.7% No Churn |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `CustomerID` | ID | Unique customer identifier (dropped in modelling) |
| `Age` | Numeric | Customer age in years |
| `Gender` | Categorical | Male / Female |
| `Tenure` | Numeric | Number of months the customer has been with the company |
| `MonthlyCharges` | Numeric | Monthly bill amount in USD |
| `ContractType` | Categorical | Month-to-Month / One-Year / Two-Year |
| `InternetService` | Categorical | DSL / Fiber Optic / None |
| `TotalCharges` | Numeric | Cumulative charges over customer lifetime (USD) |
| `TechSupport` | Categorical | Yes / No |
| `Churn` | **Target** | Yes (churned) / No (retained) |

> **Note:** Only `Age`, `Gender`, `Tenure`, and `MonthlyCharges` were used as features in the final models. Adding more features is listed under [Future Work](#-future-work).

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction.ipynb   # Main Jupyter Notebook (all sections)
├── customer_churn_data.csv           # Raw dataset
├── model.pkl                         # Saved best model (Random Forest)
├── scaler.pkl                        # Saved StandardScaler
└── README.md                         # This file
```

---

## 🔬 Workflow Overview

```
1. Business Problem Definition
         ↓
2. Exploratory Data Analysis (EDA)
   ├── Dataset overview & data dictionary
   ├── Missing value analysis
   ├── Statistical summaries
   └── Churn rate & class distribution
         ↓
3. Data Preprocessing & Cleaning
   ├── Handle missing values (InternetService → empty string)
   ├── Check for duplicates (none found)
   └── Statistical summary with .describe()
         ↓
4. Data Visualization
   ├── Churn distribution (pie chart)
   ├── Average MonthlyCharges by Churn & Gender
   ├── Average Tenure by Churn
   ├── Contract Type vs Average Price (bar chart)
   └── Histograms of MonthlyCharges & Tenure
         ↓
5. Feature Engineering
   ├── Selected features: Age, Gender, Tenure, MonthlyCharges
   ├── Encoded Gender using lambda → (Female=1, Male=0)
   └── Encoded Churn using lambda → (Yes=1, No=0)
         ↓
6. Train-Test Split & Scaling
   ├── 80/20 split (800 train / 200 test)
   ├── StandardScaler applied (fit on train, transform on test)
   └── Scaler saved → scaler.pkl
         ↓
7–10. Model Training & Hyperparameter Tuning (GridSearchCV, cv=5)
   ├── Logistic Regression
   ├── K-Nearest Neighbors
   ├── Support Vector Machine (SVC)
   ├── Decision Tree Classifier
   └── Random Forest Classifier
         ↓
11. Model Evaluation & Comparison
    └── Accuracy scores compared across all models
         ↓
12. Best Model Exported
    └── Random Forest → model.pkl
         ↓
13. Business Insights & Recommendations
```

---

## 📈 Models & Results

All models were tuned using **GridSearchCV with 5-fold cross-validation**.

| # | Model | Best Parameters | Test Accuracy |
|---|-------|----------------|:-------------:|
| 1 | Logistic Regression | Default | 87.0% |
| 2 | K-Nearest Neighbors | `n_neighbors=9`, `weights=uniform` | 86.0% |
| 3 | Support Vector Machine | `C=0.01`, `kernel=linear` | 87.5% |
| 4 | Decision Tree | `criterion=entropy`, `max_depth=10`, `min_samples_split=10` | 84.5% |
| 5 | **Random Forest** ⭐ | `n_estimators=256`, `max_features=2`, `bootstrap=True` | **88.0%** |

### 🏆 Best Model: Random Forest Classifier — 88.0% Accuracy

Random Forest outperformed all other models by combining the predictions of 256 decision trees. It was saved to `model.pkl` for future use.

---

## 🔍 Key Findings

| Finding | Detail |
|---------|--------|
| **Churners pay more monthly** | Churned customers average ~$76/month vs ~$63/month for retained customers |
| **Churners have shorter tenure** | Churned customers stay ~17 months on average vs ~30 months for retained customers |
| **Age is not a strong predictor** | Both groups average ~44–45 years old — age alone does not predict churn |
| **High overall churn rate** | 88.3% of the dataset churned — a severe retention problem requiring immediate action |

---

## 💡 Business Recommendations

**1. Prioritise Early Retention (First 6–18 Months)**
Customers who churn tend to leave early in their lifecycle. Invest in onboarding experiences, welcome offers, and proactive check-in calls during the first year to build loyalty.

**2. Review Pricing for High-Paying Customers**
Churned customers pay significantly more per month. Customers who feel they aren't receiving value for money are more likely to leave. Consider loyalty discounts or added perks for higher billing tiers.

**3. Deploy the Model in a CRM System**
The trained model (`model.pkl`) can be integrated into a CRM to automatically flag high-risk customers each week. Customer success teams can then reach out with targeted retention offers before churn occurs.

**4. Collect More Customer Features**
The current model uses only 4 features. Adding contract type, payment method, number of support tickets, and product usage data would significantly improve prediction accuracy.

**5. Address Class Imbalance**
With 88% of customers labelled as churned, the dataset is highly imbalanced. Future models should apply SMOTE (Synthetic Minority Oversampling) or use `class_weight='balanced'` to improve detection of non-churners.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```

### 4. Run all cells
Use **Kernel → Restart & Run All** to execute the full pipeline from scratch.

### 5. Load the saved model (optional)
```python
import joblib

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Example prediction on new data
# new_data = [[age, gender, tenure, monthly_charges]]
# scaled   = scaler.transform(new_data)
# prediction = model.predict(scaled)
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

---

## 🔮 Future Work

- [ ] Apply SMOTE or `class_weight='balanced'` to handle the 88:12 class imbalance
- [ ] Add more features: ContractType, InternetService, TechSupport, TotalCharges
- [ ] Try additional models: XGBoost, LightGBM, CatBoost
- [ ] Build a Streamlit web app for real-time churn scoring
- [ ] Add Precision, Recall, F1-Score, and ROC-AUC to the evaluation metrics
- [ ] Retrain the model periodically with fresh data to maintain accuracy

---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ using Python, Scikit-learn, and Matplotlib.*
