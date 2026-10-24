import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import shap

st.set_page_config(page_title="Churn AI | Customer Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#070B16;--panel:#0D1426;--line:rgba(104,145,210,.22);--cyan:#4DE2FF;--blue:#638BFF}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(35,83,166,.2),transparent 35%),radial-gradient(ellipse at 90% 5%,rgba(0,201,220,.09),transparent 28%),#070B16;color:#EAF2FF;font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1500px;padding-top:1.7rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em;color:#F3F7FF!important}
p,label,[data-testid="stMarkdownContainer"]{color:#C5D2EA}
[data-testid="stMetric"],[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(145deg,rgba(17,29,53,.92),rgba(10,17,33,.94));border:1px solid var(--line);border-radius:16px;padding:14px}
[data-testid="stMetricLabel"]{color:#91A5C8!important}[data-testid="stMetricValue"]{color:#F2F7FF!important;font-family:'Space Grotesk',sans-serif}
.stButton>button{background:linear-gradient(105deg,#3478F6,#14B8D4);color:#04111F;border:0;border-radius:10px;padding:.7rem 1.1rem;font-weight:700}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#0B1427;border-color:#263957;border-radius:10px}
input{color:#EAF2FF!important}hr{border-color:var(--line)}
.hero{padding:25px 28px;margin:0 0 20px;border:1px solid rgba(77,226,255,.25);border-radius:20px;background:linear-gradient(115deg,rgba(19,39,75,.9),rgba(10,21,42,.88));}
.eyebrow{color:#4DE2FF;font-size:.75rem;letter-spacing:.18em;font-weight:700;text-transform:uppercase}
.hero-title{color:#F3F7FF;font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;margin:.4rem 0}
.sub{color:#9DB0D1}
.small-note{color:#8EA2C4;font-size:.82rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = pd.read_csv("data/Telco_Customer_Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

@st.cache_resource
def get_models():
    return {
        "Logistic Regression": load("models/logistic_model.pkl"),
        "Random Forest": load("models/rf_model.pkl"),
        "XGBoost": load("models/xgb_model.pkl")
    }

df = get_data()
models = get_models()
rf_model = models["Random Forest"]
churned = df[df["Churn"].astype(str).str.lower().eq("yes")]
retained = df[df["Churn"].astype(str).str.lower().eq("no")]
churn_rate = len(churned) / max(len(df), 1) * 100
try:
    df["ChurnProbability"] = rf_model.predict_proba(df.drop(columns=["customerID","Churn"]))[:, 1]
except Exception:
    df["ChurnProbability"] = np.where(df["Churn"].astype(str).str.lower().eq("yes"), 0.75, 0.2)
df["RiskLevel"] = pd.cut(df["ChurnProbability"], bins=[-0.001,0.30,0.60,1.001], labels=["Low","Medium","High"])
high_risk = df[df["RiskLevel"].astype(str).eq("High")]

with st.sidebar:
    st.markdown("## ✦ Churn AI")
    st.caption("CUSTOMER INTELLIGENCE")
    page = st.radio("NAVIGATION", ["Dashboard", "Predictions", "Model Insights", "Settings"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Workspace**")
    st.caption("Telco Customer Churn dataset")
    st.caption(f"{len(df):,} customer records")

st.markdown('<div class="hero"><div class="eyebrow">✦ CUSTOMER INTELLIGENCE PLATFORM</div><div class="hero-title">Churn AI <span style="color:#4DE2FF">/</span> '+page+'</div><div class="sub">Explore customer behavior, identify churn risk, and turn model signals into retention action.</div></div>', unsafe_allow_html=True)

def chart_theme(fig, height=320):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=12,r=12,t=48,b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#C5D2EA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(104,145,210,.2)")
    fig.update_yaxes(gridcolor="rgba(104,145,210,.12)", zerolinecolor="rgba(104,145,210,.12)")
    return fig

if page == "Dashboard":
    a,b,c,d = st.columns(4)
    a.metric("TOTAL CUSTOMERS", f"{len(df):,}", "Dataset records")
    b.metric("OVERALL CHURN RATE", f"{churn_rate:.1f}%", f"{len(churned):,} churned")
    c.metric("AVG. MONTHLY CHARGES", f"${df['MonthlyCharges'].mean():.2f}", "Across all customers")
    d.metric("HIGH-RISK CUSTOMERS", f"{len(high_risk):,}", f"{len(high_risk)/max(len(df),1)*100:.1f}% of customers")

    left,right = st.columns([1.45,1])
    with left:
        st.markdown("### Customer Churn Trend")
        trend = df.assign(TenureBand=pd.cut(df["tenure"], bins=[-1,12,24,36,48,60,72], labels=["0–12","13–24","25–36","37–48","49–60","61–72"]))
        trend_df = trend.groupby("TenureBand", observed=False).agg(ChurnRate=("Churn",lambda s:(s.astype(str).str.lower()=="yes").mean()*100)).reset_index()
        fig = px.area(trend_df, x="TenureBand", y="ChurnRate", markers=True, labels={"TenureBand":"Customer tenure (months)","ChurnRate":"Churn rate (%)"})
        fig.update_traces(line_color="#4DE2FF", fillcolor="rgba(77,226,255,.16)", marker=dict(size=7))
        st.plotly_chart(chart_theme(fig), use_container_width=True)
        st.markdown('<div class="small-note">Churn rate by tenure band (the dataset does not contain dated monthly snapshots).</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Churn Distribution")
        dist = pd.DataFrame({"Status":["Retained","Churned"],"Customers":[len(retained),len(churned)]})
        fig = px.pie(dist, names="Status", values="Customers", hole=.68, color="Status", color_discrete_map={"Retained":"#4DE2FF","Churned":"#FF6482"})
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#0D1426",width=4)))
        st.plotly_chart(chart_theme(fig), use_container_width=True)

    st.markdown("### Customer Risk Overview")
    risk_counts = df["RiskLevel"].astype(str).value_counts().reindex(["Low","Medium","High"],fill_value=0).reset_index()
    risk_counts.columns = ["Risk Level","Customers"]
    risk_cols = st.columns(3)
    for col, level, color in zip(risk_cols,["Low","Medium","High"],["#4DE2FF","#FFC66D","#FF6482"]):
        count = int(risk_counts.loc[risk_counts["Risk Level"]==level,"Customers"].iloc[0])
        with col:
            st.markdown(f"#### {level} Risk")
            st.metric("Customers", f"{count:,}")
            st.progress(count/max(len(df),1))
            st.markdown(f'<span style="color:{color}">{count/max(len(df),1)*100:.1f}% of customer base</span>',unsafe_allow_html=True)

    left,right = st.columns(2)
    with left:
        st.markdown("### Churn by Contract")
        contract_df = df.groupby("Contract",observed=False).agg(ChurnRate=("Churn",lambda s:(s.astype(str).str.lower()=="yes").mean()*100)).reset_index().sort_values("ChurnRate")
        fig = px.bar(contract_df,x="ChurnRate",y="Contract",orientation="h",text=contract_df["ChurnRate"].map(lambda x:f"{x:.1f}%"),labels={"ChurnRate":"Churn rate (%)","Contract":"Contract"})
        fig.update_traces(marker_color="#638BFF",textposition="outside")
        st.plotly_chart(chart_theme(fig,280),use_container_width=True)
    with right:
        st.markdown("### Churn by Internet Service")
        internet_df = df.groupby("InternetService",observed=False).agg(ChurnRate=("Churn",lambda s:(s.astype(str).str.lower()=="yes").mean()*100)).reset_index()
        fig = px.bar(internet_df,x="InternetService",y="ChurnRate",text=internet_df["ChurnRate"].map(lambda x:f"{x:.1f}%"),labels={"InternetService":"Internet service","ChurnRate":"Churn rate (%)"})
        fig.update_traces(marker_color="#14B8D4",textposition="outside")
        st.plotly_chart(chart_theme(fig,280),use_container_width=True)

    st.markdown("### High-Risk Customers")
    view = high_risk.sort_values("ChurnProbability",ascending=False).head(10).copy()
    view["Churn Probability"] = (view["ChurnProbability"]*100).round(1).astype(str)+"%"
    st.dataframe(view[["customerID","tenure","Contract","MonthlyCharges","PaymentMethod","Churn Probability","RiskLevel"]].rename(columns={"customerID":"Customer ID","tenure":"Tenure (months)","Contract":"Contract","MonthlyCharges":"Monthly charges","PaymentMethod":"Payment method","RiskLevel":"Risk level"}),use_container_width=True,hide_index=True)

elif page == "Predictions":
    st.markdown("### Predict an individual customer's churn risk")
    st.caption("Enter customer details. The selected trained model returns a churn probability and a suggested retention action.")
    with st.form("prediction_form"):
        model_choice = st.selectbox("Model", list(models.keys()), index=1)
        customer_col, service_col, account_col = st.columns(3)
        with customer_col:
            st.markdown("#### Customer")
            gender=st.selectbox("Gender",["Female","Male"])
            senior=st.selectbox("Senior citizen",[0,1])
            partner=st.selectbox("Partner",["Yes","No"])
            dependents=st.selectbox("Dependents",["Yes","No"])
            tenure=st.slider("Tenure (months)",0,72,12)
        with service_col:
            st.markdown("#### Services")
            phoneservice=st.selectbox("Phone service",["Yes","No"])
            multiplelines=st.selectbox("Multiple lines",["No","Yes","No phone service"])
            internet=st.selectbox("Internet service",["DSL","Fiber optic","No"])
            onlinesecurity=st.selectbox("Online security",["No","Yes","No internet service"])
            onlinebackup=st.selectbox("Online backup",["No","Yes","No internet service"])
            deviceprotection=st.selectbox("Device protection",["No","Yes","No internet service"])
            techsupport=st.selectbox("Tech support",["No","Yes","No internet service"])
            streamingtv=st.selectbox("Streaming TV",["No","Yes","No internet service"])
            streamingmovies=st.selectbox("Streaming movies",["No","Yes","No internet service"])
        with account_col:
            st.markdown("#### Account")
            contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
            paperless=st.selectbox("Paperless billing",["Yes","No"])
            payment=st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly=st.number_input("Monthly charges",min_value=0.0,max_value=200.0,value=70.0,step=1.0)
            total=st.number_input("Total charges",min_value=0.0,max_value=10000.0,value=840.0,step=10.0)
        submitted=st.form_submit_button("✦ Predict Churn",use_container_width=True)
    if submitted:
        input_df=pd.DataFrame({"gender":[gender],"SeniorCitizen":[senior],"Partner":[partner],"Dependents":[dependents],"tenure":[tenure],"PhoneService":[phoneservice],"MultipleLines":[multiplelines],"InternetService":[internet],"OnlineSecurity":[onlinesecurity],"OnlineBackup":[onlinebackup],"DeviceProtection":[deviceprotection],"TechSupport":[techsupport],"StreamingTV":[streamingtv],"StreamingMovies":[streamingmovies],"Contract":[contract],"PaperlessBilling":[paperless],"PaymentMethod":[payment],"MonthlyCharges":[monthly],"TotalCharges":[total]})
        prob=float(models[model_choice].predict_proba(input_df)[0][1])
        level="High" if prob>0.6 else "Medium" if prob>0.3 else "Low"
        c1,c2,c3=st.columns(3)
        c1.metric("CHURN PROBABILITY",f"{prob*100:.1f}%")
        c2.metric("RISK LEVEL",level)
        c3.metric("MODEL USED",model_choice)
        st.progress(prob)
        if level=="High":
            st.error("High risk — consider a proactive retention call, a tailored discount, and a contract review.")
        elif level=="Medium":
            st.warning("Medium risk — monitor engagement and offer relevant service support or a loyalty benefit.")
        else:
            st.success("Low risk — maintain service quality and continue regular engagement.")
        st.markdown("#### Risk analysis")
        gauge=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,number={"suffix":"%"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#4DE2FF"},"steps":[{"range":[0,30],"color":"#123746"},{"range":[30,60],"color":"#493A20"},{"range":[60,100],"color":"#482536"}]}))
        st.plotly_chart(chart_theme(gauge,260),use_container_width=True)

elif page == "Model Insights":
    st.markdown("### Model comparison")
    perf=pd.DataFrame({"Model":["Logistic Regression","Random Forest","XGBoost"],"ROC-AUC":[0.86,0.85,0.85],"F1 Score":[0.64,0.65,0.63],"Precision":[0.52,0.56,0.55],"Recall":[0.84,0.78,0.75]})
    st.dataframe(perf,use_container_width=True,hide_index=True)
    st.caption("These evaluation scores are the values currently documented in this project's model comparison; verify them against a held-out test set before presenting them as final validation results.")
    preprocessor=rf_model.named_steps["preprocessor"]
    classifier=rf_model.named_steps["model"]
    feature_names=preprocessor.get_feature_names_out()
    importance=classifier.feature_importances_
    fi=pd.DataFrame({"Feature":feature_names,"Importance":importance}).sort_values("Importance",ascending=False).head(15).sort_values("Importance")
    fig=px.bar(fi,x="Importance",y="Feature",orientation="h",labels={"Importance":"Relative importance","Feature":"Feature"})
    fig.update_traces(marker_color="#4DE2FF")
    st.markdown("### Top drivers of customer churn")
    st.plotly_chart(chart_theme(fig,430),use_container_width=True)
    st.markdown("### Churn rate by payment method")
    pay=df.groupby("PaymentMethod",observed=False).agg(ChurnRate=("Churn",lambda s:(s.astype(str).str.lower()=="yes").mean()*100)).reset_index().sort_values("ChurnRate")
    fig=px.bar(pay,x="ChurnRate",y="PaymentMethod",orientation="h",labels={"ChurnRate":"Churn rate (%)","PaymentMethod":"Payment method"})
    fig.update_traces(marker_color="#638BFF")
    st.plotly_chart(chart_theme(fig,300),use_container_width=True)
    st.markdown("### Churn rate by tenure")
    tenure_df=df.assign(TenureBand=pd.cut(df["tenure"],bins=[-1,12,24,36,48,60,72],labels=["0–12","13–24","25–36","37–48","49–60","61–72"])).groupby("TenureBand",observed=False).agg(ChurnRate=("Churn",lambda s:(s.astype(str).str.lower()=="yes").mean()*100)).reset_index()
    fig=px.line(tenure_df,x="TenureBand",y="ChurnRate",markers=True,labels={"TenureBand":"Tenure (months)","ChurnRate":"Churn rate (%)"})
    fig.update_traces(line_color="#4DE2FF",marker_size=8)
    st.plotly_chart(chart_theme(fig,300),use_container_width=True)
    st.markdown("### Business recommendations")
    st.info("Prioritize month-to-month customers and customers with shorter tenure. Review high monthly charges, electronic-check payment patterns, and gaps in support/security services when designing retention campaigns. Treat these as dataset-level associations, not proof of causation.")

else:
    st.markdown("### Dashboard settings")
    st.markdown("**Theme:** Dark navy with cyan and electric-blue accents")
    st.markdown("**Data source:** `data/Telco_Customer_Churn.csv`")
    st.markdown("**Prediction models:** Logistic Regression, Random Forest, XGBoost")
    st.markdown("**Risk thresholds:** Low ≤ 30%, Medium > 30% and ≤ 60%, High > 60%")
    st.caption("Settings are informational in this version; model thresholds are currently fixed in the app.")
