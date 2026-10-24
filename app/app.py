import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap

# ── Churn AI visual system ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn AI | Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#070B16; --panel:#0D1426; --line:rgba(104,145,210,.20); --cyan:#4DE2FF; --blue:#638BFF; --muted:#94A7C7; }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(35,83,166,.20), transparent 35%), radial-gradient(ellipse at 90% 5%, rgba(0,201,220,.09), transparent 28%), var(--bg); color:#EAF2FF; font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
h1 { color:#F3F7FF !important; font-size:2.35rem !important; }
h2,h3 { color:#DCE8FF !important; }
p, label, [data-testid="stMarkdownContainer"] { color:#C5D2EA; }
[data-testid="stTabs"] button { color:#91A5C8; font-weight:600; border-radius:10px 10px 0 0; }
[data-testid="stTabs"] button[aria-selected="true"] { color:var(--cyan); border-bottom:2px solid var(--cyan); }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], [data-testid="stExpander"] { background:linear-gradient(145deg,rgba(17,29,53,.88),rgba(10,17,33,.92)); border:1px solid var(--line); border-radius:16px; padding:14px; }
[data-testid="stMetricLabel"] { color:#91A5C8 !important; }
[data-testid="stMetricValue"] { color:#F2F7FF !important; font-family:'Space Grotesk',sans-serif; }
.stButton > button { background:linear-gradient(105deg,#3478F6,#14B8D4); color:#04111F; border:0; border-radius:10px; padding:.72rem 1.1rem; font-weight:700; box-shadow:0 8px 28px rgba(37,147,255,.18); transition:transform .15s ease,filter .15s ease; }
.stButton > button:hover { filter:brightness(1.12); transform:translateY(-1px); color:#04111F; border:0; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div { background:#0B1427; border-color:#263957; border-radius:10px; }
input, textarea { color:#EAF2FF !important; }
[data-testid="stSlider"] [role="slider"] { background:var(--cyan); }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#3978FF,#4DE2FF); }
hr { border-color:var(--line); }
.churn-hero { padding:26px 30px; margin:0 0 22px; border:1px solid rgba(77,226,255,.22); border-radius:20px; background:linear-gradient(115deg,rgba(19,39,75,.88),rgba(10,21,42,.86)); box-shadow:0 18px 55px rgba(0,0,0,.15); }
.churn-eyebrow { color:#4DE2FF; font-size:.75rem; letter-spacing:.18em; font-weight:700; text-transform:uppercase; }
.churn-title { color:#F3F7FF; font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; margin:.45rem 0; }
.churn-subtitle { color:#9DB0D1; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

logistic_model = load("models/logistic_model.pkl")
rf_model = load("models/rf_model.pkl")
xgb_model = load("models/xgb_model.pkl")

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])
preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

with tab1:

    st.markdown("""
    <div class="churn-hero">
      <div class="churn-eyebrow">✦ CUSTOMER INTELLIGENCE PLATFORM</div>
      <div class="churn-title">Churn AI <span style="color:#4DE2FF">/</span> Prediction Studio</div>
      <div class="churn-subtitle">Predict customer attrition, understand the signals, and turn risk into retention action.</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Configure a customer profile below, then run a prediction to see churn probability and model explanation.")


    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )


    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])

    tenure = st.slider("Tenure (months)",0,72)

    phoneservice = st.selectbox("Phone Service", ["Yes","No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    onlinesecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    streamingtv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    )

    monthly = st.number_input("Monthly Charges",0.0,200.0)
    total = st.number_input("Total Charges",0.0,10000.0)


    data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phoneservice],
    "MultipleLines":[multiplelines],
    "InternetService":[internet],
    "OnlineSecurity":[onlinesecurity],
    "OnlineBackup":[onlinebackup],
    "DeviceProtection":[deviceprotection],
    "TechSupport":[techsupport],
    "StreamingTV":[streamingtv],
    "StreamingMovies":[streamingmovies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
    })

    X_transformed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )
    explainer = shap.TreeExplainer(rf_classifier)


    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    if st.button("Predict Churn"):

        prob = model.predict_proba(data)[0][1]

        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

        if prob > 0.6:
            st.error("High Risk Customer")
        elif prob > 0.3:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")
        st.write(f"Model Used: **{model_choice}**")
        st.subheader("Prediction Explanation (SHAP)")

        X_transformed = preprocessor.transform(data)

        explainer = shap.Explainer(rf_classifier)
        shap_values = explainer(X_transformed_df)

        fig = plt.figure()

        shap.plots.waterfall(
            shap_values[0, :, 1],
            show=False
        )
        st.pyplot(fig)

with tab2:
    preprocessor = rf_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    rf_classifier = rf_model.named_steps["model"]
    feature_importance = rf_classifier.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    top_features = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Drivers of Customer Churn")
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")

    sample_data = pd.concat([data]*50, ignore_index=True)

    sample_data["tenure"] = np.random.randint(0, 72, size=50)
    sample_data["MonthlyCharges"] = np.random.uniform(20, 120, size=50)
    sample_data["TotalCharges"] = np.random.uniform(100, 5000, size=50)

    X_sample_transformed = preprocessor.transform(sample_data)
    feature_names = preprocessor.get_feature_names_out()

    X_sample_df = pd.DataFrame(
        X_sample_transformed,
        columns=feature_names
    )
    explainer = shap.Explainer(rf_classifier)
    shap_values = explainer(X_sample_df)

    fig = plt.figure()
    shap.plots.beeswarm(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)

    st.subheader("SHAP Feature Importance")

    fig = plt.figure()
    shap.plots.bar(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)

    performance_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "ROC AUC": [0.86, 0.85, 0.85],
    "F1 Score": [0.64, 0.65, 0.63],
    "Precision": [0.52, 0.56, 0.55],
    "Recall": [0.84, 0.78, 0.75]
})

    st.subheader("Model Performance")
    st.dataframe(performance_df)

    st.subheader("Business Insights")
    st.markdown("""
### Key Drivers of Customer Churn

Based on the model analysis and feature importance results, several factors significantly influence customer churn:

**1️⃣ Customer Tenure**
- Customers with shorter tenure are much more likely to churn.
- New customers have a higher probability of leaving compared to long-term subscribers.

**2️⃣ Contract Type**
- Customers on **month-to-month contracts** show the highest churn risk.
- Long-term contracts such as **one-year or two-year agreements significantly reduce churn**.

**3️⃣ Monthly Charges**
- Higher monthly charges correlate with increased churn probability.
- Customers paying more are more likely to switch providers if they perceive better value elsewhere.

**4️⃣ Internet Service Type**
- Customers using **fiber optic internet services** show relatively higher churn rates compared to DSL users.

**5️⃣ Lack of Value-Added Services**
- Customers without services like **online security, tech support, or device protection** are more likely to churn.

---

### Business Recommendations

• Encourage **long-term contracts** through discounts or loyalty rewards.  
• Offer **bundled services (security, tech support)** to increase customer retention.  
• Provide **special retention offers for high-charge customers** to reduce churn risk.  
• Focus retention campaigns on **new customers with low tenure**.
""")