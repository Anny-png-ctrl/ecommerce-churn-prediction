"""
Streamlit app: E-Commerce Customer Churn Predictor

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="centered",
)


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["features"]


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


pipeline, features = load_model()
metrics = load_metrics()

st.title("🛒 E-Commerce Customer Churn Predictor")
st.write(
    "Enter a customer's profile and activity data to predict the likelihood "
    "that they will churn (stop shopping with the platform), and get a "
    "business recommendation for retaining them."
)

with st.sidebar:
    st.header("Model performance")
    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("Recall (catches churners)", f"{metrics['recall']*100:.1f}%")
    st.caption(
        f"Trained on {metrics['n_train']} customers, tested on "
        f"{metrics['n_test']}. Historical churn rate: "
        f"{metrics['churn_rate']*100:.1f}%."
    )
    st.markdown("---")
    st.caption(
        "Model: Random Forest Classifier trained on the E-Commerce "
        "Customer Churn dataset (5,630 customers, 20 features)."
    )

st.subheader("Customer profile")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months with the platform)", min_value=0, max_value=120, value=12
    )
    city_tier = st.selectbox("City tier", options=[1, 2, 3], index=0)
    warehouse_to_home = st.number_input(
        "Distance: warehouse to home (km)", min_value=0, max_value=200, value=15
    )
    hours_on_app = st.number_input(
        "Avg. hours spent on app per day", min_value=0.0, max_value=10.0, value=3.0, step=0.5
    )
    devices_registered = st.number_input(
        "Number of devices registered", min_value=1, max_value=10, value=3
    )
    satisfaction_score = st.slider("Satisfaction score (1-5)", 1, 5, 3)
    num_address = st.number_input(
        "Number of addresses saved", min_value=1, max_value=20, value=2
    )

with col2:
    complain = st.selectbox(
        "Raised a complaint recently?", options=["No", "Yes"], index=0
    )
    order_hike = st.number_input(
        "Order amount hike vs last year (%)", min_value=0.0, max_value=100.0, value=15.0
    )
    coupon_used = st.number_input(
        "Coupons used (last month)", min_value=0, max_value=20, value=1
    )
    order_count = st.number_input(
        "Order count (last month)", min_value=0, max_value=50, value=2
    )
    days_since_last_order = st.number_input(
        "Days since last order", min_value=0, max_value=365, value=5
    )
    cashback_amount = st.number_input(
        "Avg. cashback amount received", min_value=0.0, max_value=1000.0, value=150.0
    )

st.subheader("Customer segment")
col3, col4, col5 = st.columns(3)

with col3:
    login_device = st.selectbox(
        "Preferred login device",
        options=["Mobile Phone", "Computer"],
    )
    gender = st.selectbox("Gender", options=["Male", "Female"])

with col4:
    payment_mode = st.selectbox(
        "Preferred payment mode",
        options=[
            "Debit Card",
            "Credit Card",
            "E wallet",
            "UPI",
            "Cash on Delivery",
        ],
    )
    marital_status = st.selectbox(
        "Marital status", options=["Single", "Married", "Divorced"]
    )

with col5:
    order_category = st.selectbox(
        "Preferred order category",
        options=[
            "Laptop & Accessory",
            "Mobile Phone",
            "Fashion",
            "Grocery",
            "Others",
        ],
    )

if st.button("Predict churn risk", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "Tenure": tenure,
                "CityTier": city_tier,
                "WarehouseToHome": warehouse_to_home,
                "HourSpendOnApp": hours_on_app,
                "NumberOfDeviceRegistered": devices_registered,
                "SatisfactionScore": satisfaction_score,
                "NumberOfAddress": num_address,
                "Complain": 1 if complain == "Yes" else 0,
                "OrderAmountHikeFromlastYear": order_hike,
                "CouponUsed": coupon_used,
                "OrderCount": order_count,
                "DaySinceLastOrder": days_since_last_order,
                "CashbackAmount": cashback_amount,
                "PreferredLoginDevice": login_device,
                "PreferredPaymentMode": payment_mode,
                "Gender": gender,
                "PreferedOrderCat": order_category,
                "MaritalStatus": marital_status,
            }
        ]
    )[features]

    proba = pipeline.predict_proba(input_df)[0, 1]
    prediction = pipeline.predict(input_df)[0]

    st.markdown("---")
    st.subheader("Prediction")

    risk_pct = proba * 100
    if prediction == 1:
        st.error(f"⚠️ High churn risk: **{risk_pct:.1f}%** probability of churn")
    else:
        st.success(f"✅ Low churn risk: **{risk_pct:.1f}%** probability of churn")

    st.progress(min(int(risk_pct), 100))

    st.subheader("Business recommendation")
    if risk_pct >= 70:
        st.write(
            "- **Immediate retention action recommended.** This customer shows "
            "strong churn signals.\n"
            "- Offer a personalized discount or cashback incentive tied to "
            "their preferred order category.\n"
            "- If they raised a complaint, prioritize resolving it via "
            "customer support outreach within 48 hours.\n"
            "- Consider a loyalty check-in call — low satisfaction score and "
            "long gaps since last order are strong churn predictors."
        )
    elif risk_pct >= 40:
        st.write(
            "- **Moderate risk — proactive engagement recommended.**\n"
            "- Send a targeted re-engagement email or push notification with "
            "a limited-time coupon.\n"
            "- Highlight new arrivals in their preferred category to "
            "re-activate interest.\n"
            "- Monitor this customer's activity over the next billing cycle."
        )
    else:
        st.write(
            "- **Low risk — maintain standard engagement.**\n"
            "- Continue regular loyalty and cashback programs.\n"
            "- Consider upsell opportunities (bundles, premium membership) "
            "given their stable engagement."
        )

    with st.expander("See input data sent to the model"):
        display_df = input_df.T.rename(columns={0: "value"})
        display_df["value"] = display_df["value"].astype(str)
        st.dataframe(display_df)

st.markdown("---")
st.caption(
    "Business Analytics ML Assignment — Predicting e-commerce customer churn "
    "using a Random Forest classifier. For educational use."
)
