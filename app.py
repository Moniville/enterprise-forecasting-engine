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
from supabase import create_client, Client

# --- 0. BRANDING & UI CONFIG ---
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(page_title=f"{PRODUCT_NAME}: Enterprise Forecasting", layout="wide")

# Professional SaaS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .support-bar {
        background: linear-gradient(90deg, #00B0F6, #00FFCC);
        padding: 12px; border-radius: 8px; text-align: center;
        margin-bottom: 25px; color: #0e1117; font-weight: bold;
        font-size: 16px; border: 1px solid rgba(255,255,255,0.2);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05); border-radius: 12px;
        padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .main-title { font-size: 42px; font-weight: bold; color: #00B0F6; margin-top: -10px; }
    .footer-section { padding: 40px; background: rgba(255,255,255,0.02); border-radius: 15px; margin-top: 50px; border: 1px solid rgba(255,255,255,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATABASE & AI CONNECTIONS ---
def init_connections():
    """Initializes Supabase and Google Gemini AI."""
    sb, ai = None, None
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            # Fixed model path to prevent NotFound error
            ai = genai.GenerativeModel('models/gemini-1.5-flash')
    except Exception as e:
        st.error(f"Connection Error: {e}")
    return sb, ai

supabase, ai_model = init_connections()

def save_forecast_to_db(project_name, forecast_df):
    """Saves the analyzed data back to Supabase for historical tracking."""
    if supabase:
        try:
            data_json = forecast_df.to_json(orient='records')
            supabase.table("forecast_history").insert({
                "project_name": project_name, 
                "forecast_data": data_json,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            st.sidebar.error(f"DB Write Error: {e}")

# --- 2. ANALYTICS & EXPORT ENGINES ---
@st.cache_resource
def run_forecast_model(df, periods, freq):
    """Core Prophet logic for time-series forecasting."""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

def create_pdf_report(hist_total, avg_val, proj_total, status, freq_label, curr_sym, curr_name):
    """Generates a downloadable PDF summary."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"{PRODUCT_NAME} Executive Summary", ln=True, align='C')
    pdf.ln(10)
    display_curr = curr_sym if curr_sym not in ["GH₵", "₦", "د.إ", "﷼"] else curr_name.split(" ")[0]
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Lifetime Revenue: {display_curr}{hist_total:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Projected {freq_label} Total: {display_curr}{proj_total:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Market Trajectory: {status.upper()}", ln=True)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 3. TOP LEVEL: SUPPORT BANNER ---
st.markdown(f"""
    <div class="support-bar">
        🌟 <b>Support Innovation:</b> Help {BRAND_NAME} maintain this tool. 
        <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">
        Click to Tip/Donate
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR: PROJECT CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1684/1684374.png", width=70)
    st.header("Project Settings")
    
    project_name = st.text_input("Project Name:", value="Alpha_Analysis")
    
    currency_map = {
        "USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£",
        "GHS (GH₵)": "GH₵", "AED (د.إ)": "DH ", "SAR (﷼)": "SR "
    }
    selected_curr = st.selectbox("Reporting Currency:", options=list(currency_map.keys()))
    curr_sym = currency_map[selected_curr]
    
    st.divider()
    ma_window = st.slider("Data Smoothing (Days):", 2, 90, 7)
    
    if st.button("🔄 Reset Engine"):
        st.session_state.clear()
        st.rerun()

    # HIDDEN ADMIN (No visible tab)
    st.write("")
    with st.expander("🔒 System"):
        admin_key = st.text_input("Admin Key", type="password")
        is_admin = (admin_key == "Ibiene2003#")

# --- 5. ADMIN DASHBOARD (CONDITIONAL) ---
if is_admin:
    st.title("🔐 Admin Intelligence Panel")
    if supabase:
        try:
            # Simple site analytics viewer
            res = supabase.table("site_analytics").select("*").execute()
            df_analytics = pd.DataFrame(res.data)
            st.write("### Visitor Traffic")
            st.line_chart(df_analytics.groupby('country').count())
        except: st.error("Analytics table not found.")
    if st.button("Exit Admin"): st.rerun()
    st.stop()

# --- 6. MAIN ENGINE UI ---
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Forecasting Engine</p>', unsafe_allow_html=True)

col_main, col_ai = st.columns([2.5, 1])

with col_main:
    source = st.radio("Choose Intelligence Source:", ["CSV Upload", "Manual Entry"], horizontal=True)
    df_raw = None
    
    if source == "CSV Upload":
        file = st.file_uploader("Drop Enterprise CSV Here", type="csv")
        if file:
            df_raw = pd.read_csv(file)
            st.dataframe(df_raw.head(3), use_container_width=True)
            c1, c2 = st.columns(2)
            d_col = c1.selectbox("Timeline Column (Dates):", df_raw.columns)
            v_col = c2.selectbox("Financial Column (Values):", df_raw.columns)
    else:
        manual = st.text_area("Paste Data (Comma Separated):", placeholder="100, 250, 400...")
        if manual:
            df_raw = pd.DataFrame({"y": [float(x.strip()) for x in manual.split(",")]})

    if df_raw is not None:
        st.write("### Analysis Configuration")
        c3, c4 = st.columns(2)
        freq_label = c3.selectbox("Time Interval:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        horizon = c4.number_input(f"Predict Future {freq_label}:", min_value=1, value=12)

        if st.button("🚀 Execute Strategic Analysis", type="primary"):
            try:
                # Prepare data
                if source == "CSV Upload":
                    working_df = df_raw[[d_col, v_col]].copy().rename(columns={d_col: 'ds', v_col: 'y'})
                    working_df['ds'] = pd.to_datetime(working_df['ds'], errors='coerce')
                else:
                    working_df = df_raw.copy()
                    working_df['ds'] = pd.date_range(end=datetime.now(), periods=len(working_df), freq='D')
                
                working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()
                working_df['ma'] = working_df['y'].rolling(window=ma_window, min_periods=1).mean()
                
                with st.spinner("AI Engine Processing..."):
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    
                    # Store in Session
                    st.session_state.update({'forecast': f_data, 'model': f_model, 'history': working_df, 'analyzed': True})
                    save_forecast_to_db(project_name, working_df)
                    st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Execution Error: {e}")

# --- 7. AI ANALYST SIDEBAR CARD ---
with col_ai:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    if st.session_state.get('analyzed') and ai_model:
        query = st.text_input("Ask for data insights:")
        if query:
            try:
                total = st.session_state['history']['y'].sum()
                prompt = f"Context: {PRODUCT_NAME} Business Forecaster. Data Total: {total}. User Query: {query}. Respond professionally."
                response = ai_model.generate_content(prompt)
                st.info(response.text)
            except Exception as e: st.error(f"AI Error: {e}")
    else:
        st.write("Run analysis to activate AI.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 8. DASHBOARD & EXPORTS ---
if st.session_state.get('analyzed'):
    hist, fcst = st.session_state['history'], st.session_state['forecast']
    future_only = fcst.tail(horizon)
    proj_sum = future_only['yhat'].sum()
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Lifetime Volume", f"{curr_sym}{hist['y'].sum():,.2f}")
    m2.metric(f"Avg per {freq_label}", f"{curr_sym}{hist['y'].mean():,.2f}")
    m3.metric("Projected Total", f"{curr_sym}{proj_sum:,.2f}")

    view = st.radio("Intelligence View:", ["Forecast Trajectory", "Anomaly Detection", "Raw Data View"], horizontal=True)
    fig = go.Figure()
    
    if view == "Forecast Trajectory":
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name="Historical", line=dict(color="#00FFCC")))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], name="Predicted", line=dict(dash='dash', color="#00B0F6")))
    elif view == "Anomaly Detection":
        perf = fcst.set_index('ds')[['yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
        anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual'))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', name='Anomaly', marker=dict(color='red', size=10)))

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Export Buttons
    c5, c6 = st.columns(2)
    with c5:
        pdf_bytes = create_pdf_report(hist['y'].sum(), hist['y'].mean(), proj_sum, "growing", freq_label, curr_sym, selected_curr)
        st.download_button("📥 Download PDF Report", pdf_bytes, f"{project_name}_report.pdf")
    with c6:
        st.download_button("📊 Download CSV Data", fcst.to_csv(index=False), f"{project_name}_data.csv")

# --- 9. FOOTER: OWNER & ENGINEER INFO ---
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
f_left, f_center, f_right = st.columns([1, 2, 1])
with f_center:
    st.markdown("""
        <div style="text-align: center;">
            <h2 style="color: #00B0F6;">👨‍💻 Meet the Engineer</h2>
            <p style="font-size: 18px;"><b>Monivi Hope</b></p>
            <p>Data & Analytics Engineer at <b>Hope Tech</b></p>
            <p>Creating intelligent, automated solutions for the modern enterprise.</p>
            <div style="margin-top: 20px;">
                <a href="https://linktr.ee/MoniviHope" target="_blank" style="text-decoration: none; padding: 10px 20px; background: #00B0F6; color: white; border-radius: 5px;">🔗 View My Portfolio</a>
                <a href="mailto:moniviogi41@gmail.com" style="text-decoration: none; padding: 10px 20px; border: 1px solid #00B0F6; color: #00B0F6; border-radius: 5px; margin-left: 10px;">📧 Contact Me</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"<center style='color: #8b949e; padding: 20px;'>© {datetime.now().year} {BRAND_NAME} | Simple, Smart & Productive</center>", unsafe_allow_html=True)
