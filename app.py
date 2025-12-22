import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
import io
import requests
import google.generativeai as genai
from streamlit_lottie import st_lottie
from supabase import create_client, Client

# --- 0. BRANDING & UI CONFIG ---
PRODUCT_NAME = "Pulse AI" # <-- CHANGE THIS if you pick another name!
BRAND_NAME = "Hope Tech"

st.set_page_config(page_title=f"{PRODUCT_NAME} by {BRAND_NAME}", layout="wide")

# The "Lit" Professional UI CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05); border-radius: 12px;
        padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .main-title { font-size: 36px; font-weight: bold; background: -webkit-linear-gradient(#00B0F6, #00FFCC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CORE CONNECTIONS ---
def init_connections():
    sb, ai = None, None
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            ai = genai.GenerativeModel('gemini-1.5-flash')
    except: pass
    return sb, ai

supabase, ai_model = init_connections()

# --- 2. THE COMPLETE TOOLSET ---
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

def create_pdf_report(hist_total, avg_val, proj_total, status, growth_pct, freq_label, curr_sym, curr_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"{PRODUCT_NAME} Strategic Summary", ln=True, align='C')
    pdf.ln(10)
    display_curr = curr_sym if curr_sym not in ["GH₵", "₦", "د.إ", "﷼"] else curr_name.split(" ")[0]
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Business Performance Indicators:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"- Lifetime Revenue: {display_curr}{hist_total:,.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Avg per {freq_label}: {display_curr}{avg_val:,.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Horizon Projection: {display_curr}{proj_total:,.2f}", ln=True)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 3. SIDEBAR: HOPE TECH HUB ---
currency_lookup = {
    "USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£",
    "GHS (GH₵)": "GH₵", "AED (د.إ)": "DH ", "SAR (﷼)": "SR "
}

with st.sidebar:
    st.markdown(f"### {BRAND_NAME}")
    tabs = st.tabs(["🚀 Control", "👨‍💻 Engineer", "💖 Support", "🔐 Admin"])
    
    with tabs[0]:
        project_name = st.text_input("Project Name:", value="Alpha_Growth")
        selected_curr = st.selectbox("Currency:", options=list(currency_lookup.keys()))
        curr_sym = currency_lookup[selected_curr]
        ma_window = st.slider("Smoothing Window (Days):", 2, 90, 7)
        if st.button("🔄 Reset Engine"): st.session_state.clear(); st.rerun()

    with tabs[1]:
        st.write("**Monivi Hope**")
        st.caption("Data & Analytics Engineer")
        st.markdown(f"[🔗 Linktree Portfolio](https://linktr.ee/MoniviHope)")
        st.write(f"Email: moniviogi41@gmail.com")

    with tabs[2]:
        st.markdown("#### Support the Innovation")
        st.link_button("💝 Tip via Selar (Global)", "https://selar.com/showlove/hopetech", type="primary")

    with tabs[3]:
        pw = st.text_input("Admin Key", type="password")
        is_admin = (pw == "Ibiene2003#")

# --- 4. MAIN INTERFACE ---
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Analytics Engine</p>', unsafe_allow_html=True)
col_main, col_ai = st.columns([2.5, 1])

with col_main:
    input_method = st.radio("Intelligence Source:", ["CSV Upload", "Manual Entry"], horizontal=True)
    df_raw = None
    if input_method == "CSV Upload":
        file = st.file_uploader("Drop Enterprise Data", type="csv")
        if file:
            df_raw = pd.read_csv(file)
            c1, c2 = st.columns(2)
            d_col = c1.selectbox("Timeline:", df_raw.columns)
            v_col = c2.selectbox("Financials:", df_raw.columns)
    else:
        manual_data = st.text_area("Input Stream (comma separated):", placeholder="500, 1200, 800...")
        if manual_data:
            df_raw = pd.DataFrame({"y": [float(x.strip()) for x in manual_data.split(",")]})

    if df_raw is not None:
        col_f, col_h = st.columns(2)
        freq_label = col_f.selectbox("Interval:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        horizon = col_h.number_input(f"Horizon ({freq_label}):", min_value=1, value=12)

        if st.button("🚀 Execute Strategic Analysis", type="primary"):
            working_df = df_raw.copy()
            if input_method == "CSV Upload":
                working_df = working_df[[d_col, v_col]].rename(columns={d_col: 'ds', v_col: 'y'})
                working_df['ds'] = pd.to_datetime(working_df['ds'], errors='coerce')
            else:
                working_df['ds'] = pd.date_range(end=datetime.now(), periods=len(working_df), freq='D')
            
            working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()
            working_df['ma'] = working_df['y'].rolling(window=ma_window, min_periods=1).mean()
            
            with st.spinner(f"AI Modeling via {PRODUCT_NAME}..."):
                freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                st.session_state.update({'forecast': f_data, 'model': f_model, 'history': working_df, 'analyzed': True})

with col_ai:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50) # Cartoon Avatar
    st.subheader("AI Analyst Chat")
    if st.session_state.get('analyzed') and ai_model:
        query = st.text_input("Ask about insights or charts:")
        if query:
            hist_sum = st.session_state['history']['y'].sum()
            prompt = f"Product: {PRODUCT_NAME}. Data: Revenue {hist_sum}. User asks: {query}."
            st.info(ai_model.generate_content(prompt).text)
            if "chart" in query.lower() or "show" in query.lower():
                st.line_chart(st.session_state['history'].set_index('ds').tail(20))
    else: st.info("Run analysis to chat with AI.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. DASHBOARD VISUALS ---
if st.session_state.get('analyzed'):
    st.divider()
    hist, fcst = st.session_state['history'], st.session_state['forecast']
    future_only = fcst.tail(horizon)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Historical", f"{curr_sym}{hist['y'].sum():,.2f}")
    m2.metric(f"Mean ({freq_label})", f"{curr_sym}{hist['y'].mean():,.2f}")
    m3.metric("Projected Total", f"{curr_sym}{future_only['yhat'].sum():,.2f}")

    view = st.radio("Business Perspectives:", 
                    ["Strategic Forecast", "Anomaly Detector", "Performance Smoothing", "Annual Growth"], 
                    horizontal=True)
    
    fig = go.Figure()
    if view == "Strategic Forecast":
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name="Historical", line=dict(color="#00FFCC")))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], name="AI Prediction", line=dict(dash='dash', color="#00B0F6")))
    elif view == "Anomaly Detector":
        perf = fcst.set_index('ds')[['yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
        anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual'))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
    
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Export
    pdf_bytes = create_pdf_report(hist['y'].sum(), hist['y'].mean(), future_only['yhat'].sum(), "trend", 0, freq_label, curr_sym, selected_curr)
    st.download_button(f"📥 Export {PRODUCT_NAME} Report (PDF)", pdf_bytes, f"{PRODUCT_NAME}_Summary.pdf")

# Footer
st.markdown("---")
st.markdown(f"<center>{BRAND_NAME} | Simple, Smart & Productive</center>", unsafe_allow_html=True)
