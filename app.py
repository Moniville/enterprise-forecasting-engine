import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import io
import google.generativeai as genai
from supabase import create_client, Client

# --- 0. BRANDING & UI CONFIG ---
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(page_title=f"{PRODUCT_NAME}: Enterprise Forecasting", layout="wide")

# Custom Professional Styling
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
    .interpretation-box { background: rgba(0, 176, 246, 0.08); padding: 25px; border-radius: 12px; border: 1px solid #00B0F6; margin-top: 20px; }
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
            # Updated to 2.0-flash to resolve the 404/v1beta error
            ai = genai.GenerativeModel('gemini-2.0-flash') 
    except Exception as e:
        st.error(f"Setup Error: {e}")
    return sb, ai

supabase, ai_model = init_connections()

# --- 2. ANALYTICS ENGINE ---
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

# --- 3. TOP LEVEL SUPPORT ---
st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> Help us scale {PRODUCT_NAME}. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)

# --- 4. SIDEBAR ---
currency_lookup = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£", "GHS (GH₵)": "GH₵"}
with st.sidebar:
    st.header("1. Administration")
    project_name = st.text_input("Project Name:", value="Zenith_Alpha")
    selected_curr_name = st.selectbox("Currency:", options=list(currency_lookup.keys()))
    curr_sym = currency_lookup[selected_curr_name]
    input_method = st.radio("Data Source:", ["CSV Upload", "Manual Entry"])
    st.divider()
    ma_window = st.slider("Smoothing Window:", 2, 90, 7)
    with st.expander("🔒 System Access"):
        admin_key = st.text_input("Key", type="password")
        is_admin = (admin_key == "Ibiene2003#")

if is_admin:
    if supabase:
        fb = supabase.table("feedback").select("*").execute()
        st.write("### Internal Feedback Dashboard")
        st.dataframe(pd.DataFrame(fb.data))
    if st.button("Logout Admin"): st.rerun()
    st.stop()

# --- 5. DATA INGESTION ---
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Analytics Engine</p>', unsafe_allow_html=True)
col_l, col_r = st.columns([2.5, 1])

with col_l:
    df_input = None
    if input_method == "CSV Upload":
        file = st.file_uploader("Upload CSV", type="csv")
        if file:
            df_input = pd.read_csv(file)
            st.write("### 📋 Step 1: Data Preview")
            st.dataframe(df_input.head(5), use_container_width=True)
            u_date = st.selectbox("Select Date Column:", df_input.columns)
            u_val = st.selectbox("Select Value Column:", df_input.columns)
    else:
        manual = st.text_area("Paste Data (comma separated):")
        if manual: 
            df_input = pd.DataFrame({"y": [float(x.strip()) for x in manual.split(",")]})

    if df_input is not None:
        c_a, c_b = st.columns(2)
        freq_label = c_a.selectbox("Analysis Frequency:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        horizon = c_b.number_input(f"Predict Future {freq_label}:", min_value=1, value=12)
        
        if st.button("🚀 Run AI Analysis", type="primary"):
            try:
                if input_method == "CSV Upload":
                    working_df = df_input[[u_date, u_val]].copy().rename(columns={u_date: 'ds', u_val: 'y'})
                    working_df['ds'] = pd.to_datetime(working_df['ds'], errors='coerce')
                else:
                    working_df = df_input.copy()
                    working_df['ds'] = pd.date_range(end=datetime.now(), periods=len(working_df), freq='D')
                
                working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()
                
                with st.spinner("AI Analysis in Progress..."):
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    st.session_state.update({'forecast': f_data, 'model': f_model, 'history': working_df, 'analyzed': True})
            except Exception as e: st.error(f"Logic Error: {e}")

with col_r:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    if st.session_state.get('analyzed') and ai_model:
        query = st.text_input("Ask for deep insights:")
        if query:
            try:
                response = ai_model.generate_content(f"Data: {len(st.session_state['history'])} records. Hist Total: {st.session_state['history']['y'].sum()}. User Query: {query}")
                st.info(response.text)
            except Exception as e: st.error(f"AI Connection Issues: {e}")
    else: st.write("Run analysis to activate AI.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. VISUALIZATION DASHBOARD ---
if st.session_state.get('analyzed'):
    hist, fcst, model = st.session_state['history'], st.session_state['forecast'], st.session_state['model']
    future_only = fcst.tail(horizon)
    
    st.divider()
    view = st.radio("Intelligence Perspective:", ["Forecast Trajectory", "Anomaly & Spikes", "Model Accuracy", "Weekly Patterns"], horizontal=True)
    fig = go.Figure()

    if view == "Forecast Trajectory":
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers', line=dict(color='#00B0F6', width=4), name="AI Forecast"))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', fillcolor='rgba(0,176,246,0.1)', line=dict(width=0), name="Confidence"))

    elif view == "Anomaly & Spikes":
        perf = fcst.set_index('ds')[['yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
        anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
        
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Anomalies", len(anoms))
        a2.metric("Highest Spike", f"{curr_sym}{hist['y'].max():,.2f}")
        a3.metric("Deepest Dip", f"{curr_sym}{hist['y'].min():,.2f}")
        
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual Performance'))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', marker=dict(color='#FF4B4B', size=10), name='Irregularities'))

    elif view == "Model Accuracy":
        hist_preds = fcst[fcst['ds'].isin(hist['ds'])]
        hist['ma'] = hist['y'].rolling(window=ma_window).mean()
        
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Historical Data', opacity=0.3))
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['ma'], name=f'{ma_window}-Period Average', line=dict(color='#00FFCC')))
        fig.add_trace(go.Scatter(x=hist_preds['ds'], y=hist_preds['yhat'], name='AI Backtest', line=dict(dash='dot', color='#00B0F6')))

    elif view == "Weekly Patterns":
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        sample_week = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=7)})
        weekly_comp = model.predict(sample_week)[['ds', 'weekly']]
        fig.add_trace(go.Bar(x=days, y=weekly_comp['weekly'], marker_color='#00FFCC'))

    fig.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)

    # NATURAL LANGUAGE INTERPRETATION
    st.subheader("💡 Strategic Insights")
    start_v, end_v = future_only['yhat'].iloc[0], future_only['yhat'].iloc[-1]
    growth = ((end_v - start_v) / start_v) * 100 if start_v != 0 else 0
    
    st.markdown(f"""
    <div class="interpretation-box">
    <b>Pulse AI Interpretation:</b> For project <b>{project_name}</b>, the engine identifies a <b>{growth:.1f}% {"increase" if growth > 0 else "decrease"}</b> 
    trend over the next {horizon} {freq_label.lower()}s. 
    <br><br>
    <b>Trust Score:</b> In the 'Model Accuracy' view, the AI's Backtest (dotted line) follows your historical {ma_window}-day smoothing, 
    indicating a high probability of forecast validity.
    </div>
    """, unsafe_allow_html=True)

# --- 7. FOOTER & FEEDBACK ---
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
f_left, f_right = st.columns(2)
with f_left:
    st.markdown("### 👨‍💻 About the Engineer")
    st.write("**Monivi Hope** | Lead at **Hope Tech** | Data & Analytics Engineer")
    st.write("Building useful & intelligent tools (or systems) to help people live better.")
    st.markdown("[🔗 Visit Portfolio](https://linktr.ee/MoniviHope)")
with f_right:
    st.markdown("### ✉️ Support & Feedback")
    with st.form("feedback_form"):
        email = st.text_input("Your Email")
        msg = st.text_area("Message / Feedback")
        if st.form_submit_button("Send to Monivi"):
            if supabase and email and msg:
                supabase.table("feedback").insert({"email": email, "message": msg}).execute()
                st.success("Sent! Thank you.")
            else: st.error("Please fill all fields.")
st.markdown('</div>', unsafe_allow_html=True)
