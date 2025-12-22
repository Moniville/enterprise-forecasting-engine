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

# --- 0. BRANDING & UI CONFIG ---
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(page_title=f"{PRODUCT_NAME} | {BRAND_NAME}", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .support-bar {
        background: linear-gradient(90deg, #00B0F6, #00FFCC);
        padding: 12px; border-radius: 8px; text-align: center;
        margin-bottom: 25px; color: #0e1117; font-weight: bold; font-size: 16px;
    }
    .glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
    .main-title { font-size: 42px; font-weight: bold; color: #00B0F6; margin-top: -10px; }
    .interpretation-box { background: rgba(255, 255, 255, 0.05); padding: 25px; border-radius: 12px; border-left: 5px solid #00B0F6; margin-top: 20px; }
    .footer-section { padding: 40px; background: rgba(255,255,255,0.02); border-radius: 15px; margin-top: 50px; border: 1px solid rgba(255,255,255,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CONNECTIONS ---
def init_connections():
    sb, ai = None, None
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            # Auto-discovery of the best available model
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if available_models:
                selected_model = next((m for m in available_models if "1.5-flash" in m), available_models[0])
                ai = genai.GenerativeModel(selected_model)
                st.sidebar.success(f"⚡ AI Engine: {selected_model.split('/')[-1]}")
    except Exception as e:
        st.sidebar.warning("AI Node Standby.")
    return sb, ai

supabase, ai_model = init_connections()

# --- 2. ANALYTICS & HEALTH CHECK ---
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

def perform_health_check(df, date_col, val_col):
    """ Validates data quality before analysis """
    issues = []
    if df[date_col].isnull().any(): issues.append("Missing dates detected.")
    if df[val_col].isnull().any(): issues.append("Missing values in target column.")
    if len(df) < 2: issues.append("Insufficient data for forecasting.")
    return issues

# --- 3. UI LAYOUT ---
st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> Help scale {PRODUCT_NAME}. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline;">Tip/Donate</a></div>', unsafe_allow_html=True)

with st.sidebar:
    logo_path = "assets/Hope tech 2.png"
    if os.path.exists(logo_path):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2: st.image(logo_path, use_container_width=True)
    else: st.subheader(f"🛡️ {BRAND_NAME}")
    
    st.divider()
    project_name = st.text_input("Project Name:", value="Zenith Ecommerce")
    currency_lookup = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£", "GHS (GH₵)": "GH₵"}
    curr_sym = currency_lookup[st.selectbox("Currency:", options=list(currency_lookup.keys()))]
    input_method = st.radio("Source:", ["CSV Upload (Recommended)", "Manual Entry"])
    
    st.divider()
    ma_window = st.slider("Smoothing Window:", 2, 90, 7)
    if st.button("🗑️ Clear All Data"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- 4. DATA INGESTION & PROCESSING ---
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Intelligence Engine</p>', unsafe_allow_html=True)
col_l, col_r = st.columns([2.5, 1])

with col_l:
    df_input = None
    if "CSV Upload" in input_method:
        file = st.file_uploader("Upload Dataset", type="csv")
        if file:
            df_input = pd.read_csv(file)
            st.dataframe(df_input.head(3), use_container_width=True)
            u_date = st.selectbox("Date Column:", df_input.columns)
            u_val = st.selectbox("Value Column:", df_input.columns)
            
            # HEALTH CHECK EXECUTION
            health_issues = perform_health_check(df_input, u_date, u_val)
            if health_issues:
                for issue in health_issues: st.warning(f"⚠️ {issue}")
            else: st.success("✅ Data Integrity Verified.")
    else:
        manual = st.text_area("Paste Data (comma-separated):")
        if manual:
            try:
                vals = [float(x.strip()) for x in manual.split(",") if x.strip()]
                df_input = pd.DataFrame({"y": vals})
            except: st.error("Numbers only.")

    if df_input is not None:
        c_a, c_b = st.columns(2)
        freq_label = c_a.selectbox("Frequency:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        horizon = c_b.number_input(f"Predict Future {freq_label}s:", min_value=1, value=6)
        
        if st.button("🚀 Run Analysis", type="primary"):
            try:
                if "CSV Upload" in input_method:
                    working_df = df_input[[u_date, u_val]].copy().rename(columns={u_date: 'ds', u_val: 'y'})
                    working_df['ds'] = pd.to_datetime(working_df['ds'], errors='coerce')
                else:
                    freq_code = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}[freq_label]
                    working_df = df_input.copy()
                    working_df['ds'] = pd.date_range(end=datetime.now(), periods=len(working_df), freq=freq_code)
                
                working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()
                
                with st.spinner("Analyzing..."):
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    st.session_state.update({'forecast': f_data, 'model': f_model, 'history': working_df, 'analyzed': True})
            except Exception as e: st.error(f"Logic Error: {e}")

# --- 5. DEEP CONTEXT AI ENGINE ---
with col_r:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    if st.session_state.get('analyzed') and ai_model:
        query = st.text_input("Ask about your data:")
        if query:
            hist_data = st.session_state['history']
            forecast_data = st.session_state['forecast']
            
            # Context Preparation for Accurate Yearly Sales Interpretation
            total_hist = hist_data['y'].sum()
            future_proj = forecast_data['yhat'].tail(horizon).sum()
            data_range = f"{hist_data['ds'].min().date()} to {hist_data['ds'].max().date()}"
            
            prompt = f"""
            CONTEXT REPORT for {project_name}:
            - Data Range: {data_range}
            - Historical Total: {curr_sym}{total_hist:,.2f}
            - Forecast Period: {horizon} {freq_label}s
            - Forecast Total: {curr_sym}{future_proj:,.2f}
            
            USER QUESTION: {query}
            
            INSTRUCTION: Use the totals provided above to answer. If the data is monthly but the user asks for yearly, aggregate the historical total and the future projection to provide an annual trajectory estimate.
            """
            try:
                try: response = ai_model.generate_content(prompt)
                except:
                    with st.spinner("Retrying in 60s..."):
                        time.sleep(60)
                        response = ai_model.generate_content(prompt)
                st.info(response.text)
            except: st.error("AI is busy.")
    else: st.info("Run analysis to activate AI.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. VISUALIZATION ---
if st.session_state.get('analyzed'):
    hist, fcst, model = st.session_state['history'], st.session_state['forecast'], st.session_state['model']
    future_only = fcst.tail(horizon)
    view = st.radio("Perspective:", ["Forecast", "Anomalies", "Accuracy", "Monthly", "Weekly", "Annual"], horizontal=True)
    fig = go.Figure()

    if view == "Forecast":
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers', line=dict(color='#00B0F6', width=4), name="Prediction"))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', fillcolor='rgba(0,176,246,0.1)', line=dict(width=0), name="Margin"))
    elif view == "Anomalies":
        perf = fcst.set_index('ds')[['yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
        anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual'))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
    elif view == "Accuracy":
        hist_preds = fcst[fcst['ds'].isin(hist['ds'])]
        hist['ma'] = hist['y'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual', opacity=0.3))
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['ma'], name='Trend', line=dict(color='#00FFCC')))
        fig.add_trace(go.Scatter(x=hist_preds['ds'], y=hist_preds['yhat'], name='AI Backtest', line=dict(dash='dot', color='#00B0F6')))
    elif view == "Monthly":
        m_data = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
        fig.add_trace(go.Bar(x=m_data['ds'], y=m_data['y'], marker_color="#636EFA"))
    elif view == "Weekly":
        sample_week = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=7)})
        w_comp = model.predict(sample_week)[['ds', 'weekly']]
        fig.add_trace(go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=w_comp['weekly'], marker_color='#00FFCC'))
    elif view == "Annual":
        y_data = hist.set_index('ds').resample('YS')['y'].sum().reset_index()
        fig.add_trace(go.Scatter(x=y_data['ds'], y=y_data['y'], mode='lines+markers', line=dict(color="#EF553B")))

    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

# --- 7. FOOTER ---
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
fl, fr = st.columns(2)
with fl:
    st.write("### Engineer")
    st.write("**Monivi Hope** | Lead at **Hope Tech**")
    st.markdown("[🔗 Portfolio](https://linktr.ee/MoniviHope)")
with fr:
    st.write("### Support")
    with st.form("fb"):
        em, ms = st.text_input("Email"), st.text_area("Message")
        if st.form_submit_button("Submit"):
            if supabase and em and ms:
                supabase.table("feedback").insert({"email": em, "message": ms}).execute()
                st.success("Sent!")
st.markdown('</div>', unsafe_allow_html=True)
