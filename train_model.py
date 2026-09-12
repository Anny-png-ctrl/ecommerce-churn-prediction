"""
Train a customer churn classifier for an e-commerce business.

Dataset: E-Commerce Customer Churn Dataset (5,630 customers, 20 features)
Source: Kaggle - "E Commerce Dataset" (also mirrored on GitHub, see README)

Run:
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ecommerce_customer_data.xlsx"
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

TARGET = "Churn"
NUMERIC_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]
CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="E Comm")
    df = df.drop(columns=["CustomerID"])

    # Collapse duplicate category spellings found in the raw data
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(
        {"Phone": "Mobile Phone"}
    )
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
        {"COD": "Cash on Delivery", "CC": "Credit Card"}
    )
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(
        {"Mobile": "Mobile Phone"}
    )
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def main() -> None:
    df = load_data(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate": round(float(y.mean()), 4),
    }

    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Metrics:", json.dumps(metrics, indent=2))

    joblib.dump(
        {
            "pipeline": pipeline,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "features": FEATURES,
        },
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
