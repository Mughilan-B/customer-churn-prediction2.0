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
except Exception as prediction_error:
    df["ChurnProbability"] = np.nan
    st.warning(f"Dataset-level model scoring is unavailable: {prediction_error}. Risk scores are not being fabricated.")
df["RiskLevel"] = pd.cut(df["ChurnProbability"], bins=[-0.001,0.30,0.60,1.001], labels=["Low","Medium","High"])
high_risk = df[df["RiskLevel"].astype(str).eq("High")]

with st.sidebar:
    st.markdown("## ✦ Churn AI")
    st.caption("CUSTOMER INTELLIGENCE")
    page = st.radio("NAVIGATION", ["Dashboard", "Predictions", "Customer Explorer", "Bulk Risk Scanner", "What-if Simulator", "Model Insights", "Settings"], label_visibility="visible")
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


elif page == "Customer Explorer":
    st.markdown("### Customer 360 Explorer")
    st.caption("Search and filter customer records, inspect a profile, and download the filtered view.")
    c1, c2, c3 = st.columns(3)
    with c1:
        search_id = st.text_input("Customer ID contains")
    with c2:
        risks = st.multiselect("Risk levels", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    with c3:
        contracts = st.multiselect("Contract types", sorted(df["Contract"].dropna().unique()), default=sorted(df["Contract"].dropna().unique()))
    filtered = df.copy()
    if search_id.strip():
        filtered = filtered[filtered["customerID"].astype(str).str.contains(search_id.strip(), case=False, na=False)]
    filtered = filtered[filtered["RiskLevel"].astype(str).isin(risks) & filtered["Contract"].isin(contracts)]
    st.metric("Matching customers", f"{len(filtered):,}")
    if not filtered.empty:
        customer_id = st.selectbox("Select customer profile", filtered["customerID"].astype(str).tolist())
        row = filtered[filtered["customerID"].astype(str) == customer_id].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Risk level", str(row["RiskLevel"]))
        b.metric("Tenure", f"{int(row['tenure'])} months")
        c.metric("Monthly charges", f"{float(row['MonthlyCharges']):.2f} USD")
        d.metric("Churn probability", "Unavailable" if pd.isna(row["ChurnProbability"]) else f"{row['ChurnProbability']*100:.1f}%")
        st.dataframe(pd.DataFrame({"Attribute": row.index, "Value": row.astype(str).values}), use_container_width=True, hide_index=True)
        st.markdown("#### Retention suggestions")
        if row["Contract"] == "Month-to-month":
            st.info("Discuss the benefits and total cost of longer-term plans without pressuring the customer.")
        if row["tenure"] <= 12:
            st.info("Early-tenure customer: offer onboarding help and check whether the service meets expectations.")
        if row["MonthlyCharges"] >= df["MonthlyCharges"].quantile(.75):
            st.info("Review plan fit and billing clarity for this higher-spend customer.")
        if str(row.get("TechSupport", "")) == "No":
            st.info("Consider offering a technical-support check-in.")
    else:
        st.info("No customers match these filters.")
    st.download_button("Download filtered customers", filtered.to_csv(index=False).encode("utf-8"), "customer_explorer.csv", "text/csv", use_container_width=True)

elif page == "Bulk Risk Scanner":
    st.markdown("### Bulk Customer Risk Scanner")
    st.caption("Upload a CSV of customer records to score many customers at once, then download the results.")
    st.info("Required columns must match the training dataset feature names. customerID and Churn are optional; Churn is never used as a prediction input.")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            if "TotalCharges" in batch.columns:
                batch["TotalCharges"] = pd.to_numeric(batch["TotalCharges"], errors="coerce").fillna(0)
            feature_cols = [col for col in df.columns if col not in ["customerID", "Churn", "ChurnProbability", "RiskLevel"]]
            missing = [col for col in feature_cols if col not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
                template = df.drop(columns=["Churn", "ChurnProbability", "RiskLevel"], errors="ignore").head(0)
                st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"), "churn_upload_template.csv", "text/csv")
            else:
                model_choice = st.selectbox("Model for batch scoring", list(models.keys()), key="batch_model")
                features = batch[feature_cols].copy()
                for col in features.select_dtypes(include=["object"]).columns:
                    features[col] = features[col].astype(str)
                probs = models[model_choice].predict_proba(features)[:, 1]
                results = batch.copy()
                results["ChurnProbability"] = probs
                results["RiskLevel"] = pd.cut(probs, [-.001, .30, .60, 1.001], labels=["Low", "Medium", "High"]).astype(str)
                action_map = {"Low": "Maintain service quality", "Medium": "Proactive check-in and relevant loyalty benefit", "High": "Prioritize human retention review and tailored offer"}
                results["RecommendedAction"] = results["RiskLevel"].map(action_map)
                high = int((results["RiskLevel"] == "High").sum())
                a, b, c = st.columns(3)
                a.metric("Scanned", f"{len(results):,}")
                b.metric("High risk", f"{high:,}")
                c.metric("High-risk share", f"{100*high/max(len(results),1):.1f}%")
                st.dataframe(results.sort_values("ChurnProbability", ascending=False), use_container_width=True, hide_index=True)
                st.download_button("Download scored CSV", results.to_csv(index=False).encode("utf-8"), "churn_risk_scan.csv", "text/csv", use_container_width=True)
                st.caption("Model scores are estimates, not certainties. Review context and treat customers fairly before acting.")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

elif page == "What-if Simulator":
    st.markdown("### Churn What-if Simulator")
    st.caption("Compare model scores across contract options for one hypothetical customer. This is a scenario comparison, not proof of causation.")
    model_choice = st.selectbox("Model", list(models.keys()), index=1, key="whatif_model")
    a, b, c = st.columns(3)
    with a:
        gender = st.selectbox("Gender", ["Female", "Male"], key="w_gender")
        senior = st.selectbox("Senior citizen", [0, 1], key="w_senior")
        partner = st.selectbox("Partner", ["Yes", "No"], key="w_partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="w_dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="w_tenure")
        phone = st.selectbox("Phone service", ["Yes", "No"], key="w_phone")
        lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"], key="w_lines")
    with b:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="w_internet")
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"], key="w_security")
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"], key="w_backup")
        device = st.selectbox("Device protection", ["No", "Yes", "No internet service"], key="w_device")
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"], key="w_support")
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="w_tv")
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"], key="w_movies")
    with c:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="w_contract")
        paperless = st.selectbox("Paperless billing", ["Yes", "No"], key="w_paperless")
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="w_payment")
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, 1.0, key="w_monthly")
        total = st.number_input("Total charges", 0.0, 10000.0, 840.0, 10.0, key="w_total")
    base = {"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,"PhoneService":phone,"MultipleLines":lines,"InternetService":internet,"OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}
    scenarios = []
    try:
        for contract_option in ["Month-to-month", "One year", "Two year"]:
            candidate = pd.DataFrame([{**base, "Contract": contract_option}])
            probability = float(models[model_choice].predict_proba(candidate)[0, 1])
            scenarios.append({"Contract option": contract_option, "Estimated churn probability (%)": round(probability*100, 2)})
        st.metric("Month-to-month scenario score", f"{scenarios[0]['Estimated churn probability (%)']:.1f}%")
        scenario_df = pd.DataFrame(scenarios)
        fig = px.bar(scenario_df, x="Contract option", y="Estimated churn probability (%)", text=scenario_df["Estimated churn probability (%)"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(marker_color="#4DE2FF", textposition="outside")
        st.plotly_chart(chart_theme(fig, 330), use_container_width=True)
        st.caption("Scores reflect patterns learned by the model. Contract changes may not be suitable for every customer; discuss options transparently.")
    except Exception as exc:
        st.error(f"Scenario prediction failed: {exc}")

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
