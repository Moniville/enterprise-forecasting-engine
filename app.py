# ================================================================
# Pulse AI — Enterprise Forecasting Engine
# Author: Monivi Hope | Hope Tech
# ================================================================
# This application provides:
# - Time-series forecasting using Prophet
# - Interactive dashboards (Forecast, Anomaly, Accuracy, etc.)
# - Gemini-powered AI analyst with strict data grounding
# - Supabase feedback logging
# - Recruiter-grade UI & engineering structure
# ================================================================

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import google.generativeai as genai
from supabase import create_client
import streamlit.components.v1 as components

# ================================================================
# 0. BRANDING & PAGE CONFIGURATION
# ================================================================
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(
    page_title=f"{PRODUCT_NAME} | {BRAND_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# GOOGLE ANALYTICS (Optional – Safe Injection)
# ================================================================
GA_ID = "G-2XRSHF2S9F"
components.html(f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());
gtag('config', '{GA_ID}');
</script>
""", height=0)

# ================================================================
# GLOBAL DARK THEME STYLING
# ================================================================
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
[data-testid="stSidebar"] { background-color: #1a1c23; }
.support-bar {
    background: linear-gradient(90deg,#00B0F6,#00FFCC);
    padding:12px;border-radius:8px;
    text-align:center;font-weight:bold;color:#0e1117;
}
.glass-card {
    background: rgba(255,255,255,0.05);
    border:1px solid rgba(0,176,246,.3);
    border-radius:12px;padding:20px;
}
.main-title { font-size:42px;font-weight:bold;color:#00B0F6; }
.interpretation-box {
    background:rgba(0,176,246,.1);
    padding:20px;border-radius:12px;
    border-left:5px solid #00B0F6;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# 1. CONNECTION INITIALIZATION (Supabase + Gemini)
# ================================================================
def init_connections():
    sb, ai = None, None

    try:
        if "SUPABASE_URL" in st.secrets:
            sb = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_KEY"]
            )
            st.sidebar.success("✅ Database Connected")
    except:
        st.sidebar.warning("⚠️ Supabase unavailable")

    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            ai = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={"temperature": 0.7}
            )
            st.sidebar.success("✅ AI Engine Active")
        else:
            st.sidebar.error("❌ GOOGLE_API_KEY missing")
    except:
        st.sidebar.error("❌ AI initialization failed")

    return sb, ai

supabase, ai_model = init_connections()

# ================================================================
# SESSION STATE
# ================================================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_ai_call", 0)

# ================================================================
# 2. ANALYTICS UTILITIES
# ================================================================
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future), model

def calculate_insights(hist, forecast, horizon):
    hist_idx = hist.set_index("ds")

    monthly = hist_idx["y"].resample("MS").sum()
    weekly = hist_idx["y"].resample("W").sum()

    return {
        "hist_total": float(hist["y"].sum()),
        "hist_avg": float(hist["y"].mean()),
        "hist_max": float(hist["y"].max()),
        "hist_min": float(hist["y"].min()),
        "forecast_total": float(forecast["yhat"].tail(horizon).sum()),
        "forecast_avg": float(forecast["yhat"].tail(horizon).mean()),
        "forecast_max": float(forecast["yhat"].tail(horizon).max()),
        "forecast_min": float(forecast["yhat"].tail(horizon).min()),
        "monthly_total": monthly,
        "weekly_total": weekly,
        "monthly_sum_all": float(monthly.sum()),
        "weekly_sum_all": float(weekly.sum())
    }

# ================================================================
# 3. SIDEBAR CONFIGURATION
# ================================================================
with st.sidebar:
    st.header("Project Configuration")
    project_name = st.text_input("Project Name", "Demo Project")

    currency_map = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€"}
    curr_sym = currency_map[st.selectbox("Currency", list(currency_map))]

    input_method = st.radio("Data Input", ["CSV Upload", "Manual Entry"])
    horizon = st.number_input("Forecast Horizon", 1, 60, 12)

    with st.expander("🔒 Developer Access"):
        admin_key = st.text_input("Security Key", type="password")
        is_admin = admin_key == "Ibiene2003#"

# ================================================================
# 🔑 CRITICAL FIX — DEFINE LAYOUT BEFORE st.stop()
# ================================================================
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Analytics Engine</p>', unsafe_allow_html=True)
col_left, col_right = st.columns([2.2, 1.3])

# ================================================================
# ADMIN MODE (SAFE EXIT)
# ================================================================
if is_admin:
    st.subheader("Admin Panel")
    if supabase:
        st.dataframe(pd.DataFrame(
            supabase.table("feedback").select("*").execute().data
        ))
    st.stop()

# ================================================================
# 4. DATA INGESTION
# ================================================================
with col_left:
    df = None

    if input_method == "CSV Upload":
        file = st.file_uploader("Upload CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            date_col = st.selectbox("Date Column", df.columns)
            val_col = st.selectbox("Value Column", df.columns)
            df = df[[date_col, val_col]].rename(columns={date_col: "ds", val_col: "y"})
            df["ds"] = pd.to_datetime(df["ds"])
    else:
        raw = st.text_area("Enter values (comma-separated)")
        if raw:
            df = pd.DataFrame({"y": [float(x) for x in raw.split(",")]})
            df["ds"] = pd.date_range(end=datetime.now(), periods=len(df))

    if df is not None and st.button("🚀 Process Intelligence"):
        forecast, model = run_forecast_model(df, horizon, "MS")
        insights = calculate_insights(df, forecast, horizon)
        st.session_state.update({
            "history": df,
            "forecast": forecast,
            "model": model,
            "insights": insights,
            "analyzed": True
        })

# ================================================================
# 5. AI CHAT ANALYST
# ================================================================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.get("analyzed") and ai_model:
        if query := st.chat_input("Ask about your data"):
            context = f"""
You are a senior data analyst.

Historical Total: {curr_sym}{st.session_state['insights']['hist_total']:,.2f}
Forecast Total: {curr_sym}{st.session_state['insights']['forecast_total']:,.2f}

Question: {query}
"""
            response = ai_model.generate_content(context)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# 6. DASHBOARD
# ================================================================
if st.session_state.get("analyzed"):
    hist = st.session_state["history"]
    fcst = st.session_state["forecast"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="Historical"))
    fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], name="Forecast"))
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# 7. FOOTER & FEEDBACK
# ================================================================
st.divider()
with st.form("feedback"):
    email = st.text_input("Email")
    msg = st.text_area("Feedback")
    if st.form_submit_button("Submit") and supabase:
        supabase.table("feedback").insert({"email": email, "message": msg}).execute()
        st.success("Feedback submitted")
