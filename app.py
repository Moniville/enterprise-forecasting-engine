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
# 0. BRANDING & UI CONFIGURATION
# =================================================================
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(
    page_title=f"{PRODUCT_NAME} | {BRAND_NAME}", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GOOGLE ANALYTICS INTEGRATION ---
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
        gtag('config', '{GA_ID}', {{ 'page_path': window.parent.location.pathname, 'debug_mode': true }});
    </script>
"""
components.html(ga_injection, height=0, width=0)

# --- CUSTOM CSS: UNIFIED DARK THEME ---
st.markdown("""
    <style>
        header[data-testid="stHeader"] { background-color: #0e1117 !important; }
        .stAppViewMain, .stApp, [data-testid="stAppViewContainer"] { background-color: #0e1117 !important; color: #ffffff !important; }
        
        button[kind="primary"], button[kind="secondary"], .stButton > button, div[data-testid="stForm"] button {
            background-color: #1a1c23 !important;
            color: #ffffff !important;
            border: 2px solid #00B0F6 !important;
            font-weight: bold !important;
            opacity: 1 !important;
            display: inline-flex !important;
        }
        button:hover { background-color: #00B0F6 !important; color: #0e1117 !important; box-shadow: 0 0 15px #00B0F6 !important; }

        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #ffffff !important; opacity: 1 !important; }
        [data-testid="stSidebar"] { background-color: #1a1c23 !important; border-right: 1px solid rgba(0, 176, 246, 0.2) !important; }

        .support-bar {
            background: linear-gradient(90deg, #00B0F6, #00FFCC);
            padding: 12px; border-radius: 8px; text-align: center;
            margin-bottom: 25px; color: #0e1117 !important; font-weight: bold;
        }
        .glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(0, 176, 246, 0.3); }
        .main-title { font-size: 42px; font-weight: bold; color: #00B0F6 !important; }
        .interpretation-box { background: rgba(0, 176, 246, 0.1); padding: 25px; border-radius: 12px; border-left: 5px solid #00B0F6; margin-top: 20px; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 1. SYSTEM INITIALIZATION
# =================================================================
def init_connections():
    sb, ai = None, None
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            st.sidebar.success("✅ Database Connected")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Database: {str(e)[:50]}")
    
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            ai = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"temperature": 0.7, "top_p": 0.95, "max_output_tokens": 1024}
            )
            st.sidebar.success("✅ AI Engine Active")
        else:
            st.sidebar.error("❌ GOOGLE_API_KEY not found")
    except Exception as e:
        st.sidebar.error(f"❌ AI Setup Failed: {str(e)[:50]}")
    
    return sb, ai

supabase, ai_model = init_connections()

# Initialize session state
if "last_ai_call" not in st.session_state: st.session_state.last_ai_call = 0
if "messages" not in st.session_state: st.session_state.messages = []
if "analyzed" not in st.session_state: st.session_state.analyzed = False

# =================================================================
# 2. ANALYTICS & HEALTH TOOLS
# =================================================================
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

def perform_health_check(df, date_col, val_col):
    issues = []
    if df[date_col].isnull().any(): issues.append("Missing dates detected.")
    if df[val_col].isnull().any(): issues.append("Missing values in target column.")
    if len(df) < 2: issues.append("Insufficient data for forecasting.")
    return issues

def calculate_insights(hist_data, forecast_data, horizon, curr_sym):
    insights = {
        'hist_total': hist_data['y'].sum(),
        'hist_avg': hist_data['y'].mean(),
        'hist_max': hist_data['y'].max(),
        'hist_min': hist_data['y'].min(),
        'forecast_total': forecast_data['yhat'].tail(horizon).sum(),
        'forecast_avg': forecast_data['yhat'].tail(horizon).mean(),
        'forecast_max': forecast_data['yhat'].tail(horizon).max(),
        'forecast_min': forecast_data['yhat'].tail(horizon).min(),
    }
    insights['growth_rate'] = ((insights['forecast_total'] - insights['hist_total']) / insights['hist_total']) * 100 if insights['hist_total'] > 0 else 0
    
    hist_with_date = hist_data.set_index('ds')
    insights['daily_avg'] = hist_with_date['y'].resample('D').sum().mean()
    insights['weekly_total'] = hist_with_date['y'].resample('W').sum()
    insights['monthly_total'] = hist_with_date['y'].resample('MS').sum()
    return insights

# =================================================================
# 3. UI LAYOUT & SIDEBAR CONTROL
# =================================================================
if os.path.exists("assets/Hope tech 2.png"):
    st.sidebar.image("assets/Hope tech 2.png", width=120)

st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    project_name = st.text_input("Project Namespace:", value="Pulse Project")
    currency_lookup = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£", "GHS (GH₵)": "GH₵"}
    selected_curr_name = st.selectbox("Currency:", options=list(currency_lookup.keys()))
    curr_sym = currency_lookup[selected_curr_name]
    input_method = st.radio("Data Source:", ["CSV Upload (Recommended)", "Manual Entry"])
    ma_window = st.slider("Smoothing Window:", 2, 90, 7)
    
    if st.button("🗑️ Reset All Cache"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    with st.expander("🔒 Developer Access"):
        admin_key = st.text_input("Security Key", type="password")
        if admin_key == "Ibiene2003#":
            if supabase:
                try:
                    fb = supabase.table("feedback").select("*").execute()
                    st.write("### Internal Feedback Log")
                    st.dataframe(pd.DataFrame(fb.data))
                except: st.error("Log fetch failed.")

# =================================================================
# 4. DATA PROCESSING
# =================================================================
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Analytics Engine</p>', unsafe_allow_html=True)
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    df_input = None
    if "CSV Upload" in input_method:
        file = st.file_uploader("Drop dataset here", type="csv")
        if file:
            df_input = pd.read_csv(file)
            st.dataframe(df_input.head(3), use_container_width=True)
            u_date = st.selectbox("Map Date Column:", df_input.columns)
            u_val = st.selectbox("Map Target Value:", df_input.columns)
    else:
        manual = st.text_area("Paste comma-separated values:")
        if manual:
            try:
                vals = [float(x.strip()) for x in manual.split(",") if x.strip()]
                df_input = pd.DataFrame({"y": vals})
            except: st.error("Numerical values only please.")

    if df_input is not None:
        c1, c2 = st.columns(2)
        freq_label = c1.selectbox("Frequency:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        horizon = c2.number_input(f"Horizon ({freq_label}s):", min_value=1, value=12)
        
        if st.button("🚀 Process Intelligence", type="primary"):
            try:
                if "CSV Upload" in input_method:
                    working_df = df_input[[u_date, u_val]].copy().rename(columns={u_date: 'ds', u_val: 'y'})
                    working_df['ds'] = pd.to_datetime(working_df['ds'], errors='coerce')
                else:
                    freq_code = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}[freq_label]
                    working_df = df_input.copy()
                    working_df['ds'] = pd.date_range(end=datetime.now(), periods=len(working_df), freq=freq_code)
                
                working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()
                with st.spinner("AI Processing..."):
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    insights = calculate_insights(working_df, f_data, horizon, curr_sym)
                    
                    st.session_state.update({
                        'forecast': f_data, 'model': f_model, 'history': working_df, 
                        'analyzed': True, 'horizon': horizon, 'freq_label': freq_label, 'insights': insights
                    })
                    st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# =================================================================
# 5. CHAT-STYLE AI ASSISTANT
# =================================================================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    chat_container = st.container(height=400)
    
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    if st.session_state.get('analyzed'):
        if query := st.chat_input("Ask about your projections..."):
            if time.time() - st.session_state.last_ai_call < 2:
                st.warning("Wait a moment...")
            else:
                st.session_state.messages.append({"role": "user", "content": query})
                # AI Logic starts here
                ins = st.session_state['insights']
                context = f"Analyst for {project_name}. Total Hist: {curr_sym}{ins['hist_total']:,.2f}. Projected Total: {curr_sym}{ins['forecast_total']:,.2f}. Growth: {ins['growth_rate']:.2f}%. Query: {query}"
                
                try:
                    response = ai_model.generate_content(context)
                    ans = response.text
                except:
                    ans = f"Connection busy. Summary: Projected growth of {ins['growth_rate']:.2f}% expected."
                
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.session_state.last_ai_call = time.time()
                st.rerun()
    else:
        st.info("Upload data to start AI chat.")
    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================
# 6. DASHBOARD
# =================================================================
if st.session_state.get('analyzed'):
    hist, fcst = st.session_state['history'], st.session_state['forecast']
    horizon, ins = st.session_state['horizon'], st.session_state['insights']
    future_only = fcst.tail(horizon)
    
    view = st.radio("View:", ["Forecast", "Anomalies", "Accuracy", "Monthly", "Annual"], horizontal=True)
    fig = go.Figure()

    if view == "Forecast":
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers', name="Prediction", line=dict(color='#00B0F6', width=4)))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', line=dict(width=0), name="Confidence"))
    elif view == "Anomalies":
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Historical Data'))
    elif view == "Monthly":
        monthly = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
        fig.add_trace(go.Bar(x=monthly['ds'], y=monthly['y'], marker_color="#00B0F6"))

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0e1117', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f'<div class="interpretation-box"><b>🔍 Executive Summary:</b> Expected volume of <b>{curr_sym}{future_only["yhat"].sum():,.2f}</b> with <b>{ins["growth_rate"]:.1f}%</b> momentum.</div>', unsafe_allow_html=True)

# =================================================================
# 7. FOOTER
# =================================================================
st.divider()
f1, f2 = st.columns(2)
with f1:
    st.write(f"**Monivi Hope** | Lead at **{BRAND_NAME}**")
    st.markdown("[🔗 Digital Portfolio](https://linktr.ee/MoniviHope)")
with f2:
    with st.form("feedback"):
        email = st.text_input("Email")
        msg = st.text_area("Message")
        if st.form_submit_button("Submit"):
            if supabase and email and msg:
                supabase.table("feedback").insert({"email": email, "message": msg}).execute()
                st.success("Sent.")
