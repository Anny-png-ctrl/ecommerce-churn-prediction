# E-Commerce Customer Churn Prediction

A business analytics ML project that predicts whether an e-commerce customer
will churn, and gives a business recommendation for retaining at-risk
customers. Built as a Streamlit web application backed by a trained
scikit-learn model.

**Live app:** https://ecommerce-churn-prediction-vym2kejtfbsm8h8we6tmtc.streamlit.app
**GitHub repo:** https://github.com/Anny-png-ctrl/ecommerce-churn-prediction
**Project report:** [docs/Project_Report.docx](docs/Project_Report.docx)
**Presentation:** [docs/Presentation.pptx](docs/Presentation.pptx)

## 1. Business problem

Customer churn — customers who stop purchasing — directly erodes recurring
revenue for an e-commerce business. Acquiring a new customer typically costs
far more than retaining an existing one, so identifying at-risk customers
*before* they leave lets marketing and customer-success teams intervene with
targeted offers, support outreach, or loyalty incentives.

This project builds a classification model that predicts churn risk for an
individual customer from their profile and recent activity, and pairs each
prediction with a concrete retention recommendation.

## 2. Dataset

- **Name:** E-Commerce Customer Churn Dataset
- **Size:** 5,630 customers, 20 columns (19 features + target)
- **Target:** `Churn` (1 = churned, 0 = retained); base churn rate ≈ 16.8%
- **Features:** tenure, city tier, distance from warehouse, app usage hours,
  devices registered, satisfaction score, number of addresses, complaint
  history, order growth vs. last year, coupons used, order count, days since
  last order, cashback amount, plus categorical fields (login device, payment
  mode, gender, preferred order category, marital status).
- **File:** [`data/ecommerce_customer_data.xlsx`](data/ecommerce_customer_data.xlsx)
  (sheet `E Comm`), sourced from the public Kaggle "E Commerce Dataset".

## 3. Methodology

1. **Data cleaning** — dropped the ID column, merged duplicate category
   labels (e.g. `"Phone"` → `"Mobile Phone"`).
2. **EDA** — churn distribution, churn rate by complaint status and by
   preferred order category, tenure/satisfaction/recency vs. churn.
3. **Preprocessing pipeline** (`sklearn.pipeline.Pipeline` +
   `ColumnTransformer`):
   - Numeric features: median imputation + standard scaling.
   - Categorical features: most-frequent imputation + one-hot encoding.
4. **Model** — `RandomForestClassifier` (300 trees, max depth 12,
   `class_weight="balanced"` to handle the ~17% minority churn class).
5. **Evaluation** — held-out 20% test split, stratified by churn.
6. **Deployment** — pipeline serialized with `joblib`, loaded by a Streamlit
   app that takes user input, predicts churn probability, and shows a
   business recommendation.

See the full walkthrough in
[`Ecommerce_Churn_Prediction.ipynb`](Ecommerce_Churn_Prediction.ipynb).

## 4. Model performance

| Metric | Score |
|---|---|
| Accuracy | 95.9% |
| Precision (churn class) | 87.9% |
| Recall (churn class) | 87.9% |
| F1-score (churn class) | 87.9% |
| ROC-AUC | 0.991 |

(See [`model/metrics.json`](model/metrics.json) for the exact numbers from
the last training run — re-run `train_model.py` to regenerate.)

## 5. Business insights & recommendations

- **Recency and tenure are the strongest churn signals** — customers with
  fewer days on the platform or a long gap since their last order are far
  more likely to churn. Trigger a win-back email/coupon after ~2 weeks of
  inactivity.
- **Complaints strongly predict churn** — customers who filed a complaint
  churn at a much higher rate. Fast, prioritized complaint resolution is a
  high-leverage retention lever.
- **Low satisfaction scores and order category** also matter — segment-level
  promotions (e.g. targeted cashback for low-satisfaction customers in a
  specific category) can be more effective than blanket discounts.
- The model's high recall means it reliably surfaces the customers most
  worth spending retention budget on, rather than blasting offers to
  everyone.

## 6. Project structure

```
ecommerce-churn-app/
├── app.py                              # Streamlit application
├── train_model.py                      # Data prep + model training script
├── Ecommerce_Churn_Prediction.ipynb    # Full EDA + modelling notebook
├── requirements.txt
├── data/
│   └── ecommerce_customer_data.xlsx    # Source dataset
├── model/
│   ├── churn_model.pkl                 # Trained pipeline (preprocessing + model)
│   └── metrics.json                    # Held-out test metrics
└── .streamlit/
    └── config.toml
```

## 7. Running locally

```bash
pip install -r requirements.txt

# (optional) retrain the model from scratch
python train_model.py

# launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Fill in the customer profile and
click **Predict churn risk** to see the churn probability and a business
recommendation.

## 8. Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select this repository/branch, and set the main file
   path to `ecommerce-churn-app/app.py` (or `app.py` if this folder is the
   repo root).
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and starts
   the app automatically; you'll get a public `*.streamlit.app` URL.
5. Add that URL to the top of this README once deployed.

## 9. Team / academic context

Business Analytics group assignment: build and deploy a machine learning
model as an interactive Streamlit application. Topic: predicting e-commerce
customer churn using classification.
