import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import io
import time
import os
import google.generativeai as genai
from supabase import create_client, Client
import streamlit.components.v1 as components

# =================================================================
# 0. BRANDING & UI CONFIGURATION (Recruiter-Facing Design)
# =================================================================
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(
    page_title=f"{PRODUCT_NAME} | {BRAND_NAME}", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# GOOGLE ANALYTICS (NON-INTRUSIVE)
# =================================================================
GA_ID = "G-2XRSHF2S9F"
ga_injection = f"""
<script>
const script = window.parent.document.createElement('script');
script.async = true;
script.src = 'https://www.googletagmanager.com/gtag/js?id={GA_ID}';
window.parent.document.head.appendChild(script);
window.parent.dataLayer = window.parent.dataLayer || [];
function gtag(){{window.parent.dataLayer.push(arguments);}}
gtag('js', new Date());
gtag('config', '{GA_ID}', {{ 'page_path': window.parent.location.pathname }});
</script>
"""
components.html(ga_injection, height=0)

# =================================================================
# 1. SYSTEM INITIALIZATION
# =================================================================
def init_connections():
    """
    Initializes Supabase and Google Gemini connections safely.
    Failure here must NOT crash the app.
    """
    sb, ai = None, None

    # --- Supabase ---
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            st.sidebar.success("✅ Database Connected")
    except Exception as e:
        st.sidebar.warning("⚠️ Database unavailable")

    # --- Google Gemini ---
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            ai = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.4,  # lower temp = more factual answers
                    "top_p": 0.9,
                    "max_output_tokens": 900,
                }
            )
            st.sidebar.success("✅ AI Engine Active")
    except Exception:
        st.sidebar.error("❌ AI Engine Failed")

    return sb, ai

supabase, ai_model = init_connections()

# =================================================================
# SESSION STATE
# =================================================================
if "last_ai_call" not in st.session_state:
    st.session_state.last_ai_call = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

# =================================================================
# 2. ANALYTICS ENGINE (ALL MATH DONE IN PYTHON)
# =================================================================
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model


def calculate_insights(hist_data, forecast_data, horizon, curr_sym):
    """
    IMPORTANT DESIGN DECISION:
    - Python computes ALL numbers
    - AI only EXPLAINS numbers
    """

    hist_idx = hist_data.set_index("ds")

    monthly_totals = hist_idx["y"].resample("MS").sum()
    weekly_totals = hist_idx["y"].resample("W").sum()

    insights = {
        # Historical stats
        "hist_total": float(hist_data["y"].sum()),
        "hist_avg": float(hist_data["y"].mean()),
        "hist_max": float(hist_data["y"].max()),
        "hist_min": float(hist_data["y"].min()),

        # Forecast stats
        "forecast_total": float(forecast_data["yhat"].tail(horizon).sum()),
        "forecast_avg": float(forecast_data["yhat"].tail(horizon).mean()),
        "forecast_max": float(forecast_data["yhat"].tail(horizon).max()),
        "forecast_min": float(forecast_data["yhat"].tail(horizon).min()),

        # RAW series for AI reasoning (NOT formatted strings)
        "monthly_total": monthly_totals,
        "weekly_total": weekly_totals,

        # Precomputed totals (AI must NOT recalc)
        "monthly_sum_all": float(monthly_totals.sum()),
        "weekly_sum_all": float(weekly_totals.sum())
    }

    insights["growth_rate"] = (
        ((insights["forecast_total"] - insights["hist_total"]) / insights["hist_total"]) * 100
        if insights["hist_total"] > 0 else 0
    )

    return insights

# =================================================================
# 5. CHAT-STYLE AI ASSISTANT (FIXED & DATA-AWARE)
# =================================================================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.get("analyzed") and ai_model:
        if query := st.chat_input("Ask about your projections..."):

            # Cooldown protection
            if time.time() - st.session_state.last_ai_call < 2:
                st.warning("⏳ Please wait a moment before sending another message.")
            else:
                st.session_state.messages.append({"role": "user", "content": query})

                insights = st.session_state["insights"]

                # 🔑 KEY FIX:
                # Pass RAW historical monthly values so AI can reason correctly
                monthly_raw = {
                    str(k.date()): float(v)
                    for k, v in insights["monthly_total"].items()
                }

                context = f"""
You are a senior data analyst at {BRAND_NAME}.

STRICT RULES:
- Use HISTORICAL data unless forecast is explicitly requested
- Do NOT invent numbers
- Do NOT recompute totals
- Explain insights clearly and professionally

HISTORICAL MONTHLY SALES DATA:
{monthly_raw}

TOTAL HISTORICAL SALES (ALL MONTHS):
{curr_sym}{insights["monthly_sum_all"]:,.2f}

FORECAST SUMMARY:
- Forecast Total: {curr_sym}{insights["forecast_total"]:,.2f}
- Growth Rate: {insights["growth_rate"]:+.2f}%

USER QUESTION:
{query}
"""

                try:
                    response = ai_model.generate_content(context)
                    answer = response.text
                except Exception:
                    # Safe fallback (NO forecast hijacking)
                    answer = (
                        f"Based on the uploaded data, the total historical sales "
                        f"from January to December equal **{curr_sym}{insights['monthly_sum_all']:,.2f}**."
                    )

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_ai_call = time.time()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
