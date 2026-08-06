<div align="center">

# 🧾 Invoice Intelligence

### ML-powered freight cost prediction & invoice fraud detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Two production-style ML pipelines built on real vendor invoice and purchase order data —
one predicts freight cost, the other flags invoices that look fraudulent or mishandled.

</div>

---

## 📌 Overview

Vendor invoice processing generates two recurring problems for procurement teams:

1. **How much should freight cost?** — useful for budgeting and catching invoices where freight looks inflated.
2. **Which invoices need a second look?** — dollar amounts that don't match what was actually purchased, or shipments that arrived far later than expected, are early signs of billing errors or fraud.

This project builds an end-to-end pipeline for both problems: raw SQL data → feature engineering → model training & comparison → saved model artifacts → inference scripts you can run on new invoices.

---

## 🏗️ Architecture

```
                         ┌─────────────────────┐
                         │   inventory.db       │
                         │   (SQLite)           │
                         │  Vendor_Invoice       │
                         │  Purchases            │
                         │  Purchase_Prices      │
                         │  Begin/End_Inventory  │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                      │
     ┌───────────▼────────────┐          ┌─────────────▼─────────────┐
     │  freight_cost_prediction │          │      invoice_flagging      │
     │  ─────────────────────  │          │      ─────────────────      │
     │  Regression: predicts    │          │  Classification: flags     │
     │  Freight_Cost from        │          │  risky/fraudulent invoices │
     │  invoice Dollars          │          │  from dollar mismatches &  │
     │                           │          │  receiving delays          │
     └───────────┬───────────┘          └─────────────┬─────────────┘
                 │                                      │
                 └──────────────────┬───────────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │     inferencing/       │
                         │  freight_cost.py        │
                         │  predict_invoice_flag.py│
                         └───────────────────────┘
```

---

## 🎯 What each model does

### 1. Freight Cost Prediction
Predicts `Freight_Cost` from the invoice amount (`Dollars`) using regression, so unusually high freight charges stand out against what's expected.

### 2. Invoice Fraud/Risk Flagging
Classifies each invoice as **normal (0)** or **flagged (1)** based on two real risk signals:
- The invoice total doesn't match the actual purchased item total by more than **$350**
- The average delay between PO date and receiving date exceeds **10 days**

---

## 🧠 Feature Engineering — the part that matters most

The invoice-flagging label is defined by a *relationship between two columns*, not a single raw value:

```python
abs(Invoice_Dollars - Total_Item_Dollars) > 350   OR   Avg_Receiving_Delay > 10
```

Tree-based models can't natively represent "the difference between two features" — they split on one feature at a time. So this project explicitly engineers that relationship as its own column:

```python
df["Dollar_Gap"] = (df["Invoice_Dollars"] - df["Total_Item_Dollars"]).abs()
```

This single change took the classifier's recall on flagged invoices from **17% → 99%** on the same dataset. It's the difference between a model that mostly misses fraud and one that reliably catches it.

---

## 📊 Results

### Invoice Flagging — Random Forest (tuned)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Normal (0) | 1.00 | 1.00 | 1.00 |
| **Flagged (1)** | **1.00** | **0.99** | **0.99** |

*4,878 invoices · 14% flagged · stratified 80/20 split*

### Freight Cost Prediction — model comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 7.81 | 10.53 | 96.86% |
| **Decision Tree (depth=5)** | **7.32** | **10.10** | **97.12%** |
| Random Forest (depth=3) | 9.00 | 11.85 | 96.03% |

The freight-cost relationship turns out to be strongly linear/near-linear, so a shallow, simple model performs best — a good example of not over-engineering when the data doesn't need it.

---

## 📁 Project Structure

```
Invoice-Intelligence-ML-Project/
├── data/
│   └── inventory.db                    # SQLite source database
├── freight_cost_prediction/
│   ├── data_preprocessing.py           # Load + prep data for regression
│   ├── model_evaluation.py             # Train & evaluate regressors
│   ├── train.py                        # Pipeline entry point
│   └── models/
│       └── predict_freight_model.pkl
├── invoice_flagging/
│   ├── data_preprocessing.py           # Load, label, feature-engineer, scale
│   ├── modeling_evaluation.py          # Train, tune & evaluate classifiers
│   ├── train.py                        # Pipeline entry point
│   └── models/
│       ├── random_forest.pkl
│       └── scaler.pkl
├── inferencing/
│   ├── freight_cost.py                 # Run freight predictions on new data
│   └── predict_invoice_flag.py         # Run fraud/risk flags on new invoices
├── notebooks/
│   └── Predicting Freight Cost.ipynb   # Exploratory analysis
├── LICENSE
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/Invoice-Intelligence-ML-Project.git
cd Invoice-Intelligence-ML-Project
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code></summary>

```
pandas
scikit-learn
joblib
```
</details>

---

## 🚀 Usage

### Train the models

```bash
cd invoice_flagging
python train.py

cd ../freight_cost_prediction
python train.py
```

Each run prints a full model comparison (Logistic Regression / Decision Tree / Random Forest / Tuned Random Forest for flagging; Linear / Decision Tree / Random Forest for freight) and saves the best-performing model to its `models/` folder.

### Run predictions on new invoices

```bash
cd inferencing
python predict_invoice_flag.py   # flags risky invoices
python freight_cost.py           # predicts freight cost
```

Example output:

```
   Invoice_Quantity  Invoice_Dollars  ...  Dollar_Gap  predicted_flag
0               120             9500  ...        5020               1
1                80             6200  ...         700               1
2                50             2000  ...          10               0
```

---

## 🔍 Modeling Approach

- **Data**: SQL joins across `Vendor_Invoice` and `Purchases` (aggregated per PO) via SQLite/pandas
- **Preprocessing**: `StandardScaler` on all numeric features, saved alongside the model so inference always scales inputs identically to training
- **Model selection**: multiple algorithms trained and compared side-by-side rather than committing to one upfront
- **Tuning**: `RandomizedSearchCV` over Random Forest hyperparameters (`n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features`, `class_weight`), optimized on F1 score to account for class imbalance
- **Validation**: stratified train/test split to preserve the real-world ratio of flagged vs. normal invoices in evaluation

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `SQLite` · `joblib`

---

## 🔮 Future Improvements

- [ ] Add SHAP-based explainability so flagged invoices come with a human-readable reason
- [ ] Automate retraining on a schedule as new invoice data comes in

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

<div align="center">

Built as a hands-on exploration of end-to-end ML pipelines — from raw SQL to deployable predictions.

</div>