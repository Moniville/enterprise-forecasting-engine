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

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        header[data-testid="stHeader"] { background-color: #0e1117 !important; }
        .stAppViewMain, .stApp, [data-testid="stAppViewContainer"] { background-color: #0e1117 !important; color: #ffffff !important; }
        
        button[kind="primary"], button[kind="secondary"], .stButton > button, div[data-testid="stForm"] button {
            background-color: #1a1c23 !important;
            color: #ffffff !important;
            border: 2px solid #00B0F6 !important;
            font-weight: bold !important;
        }
        button:hover { background-color: #00B0F6 !important; color: #0e1117 !important; box-shadow: 0 0 15px #00B0F6 !important; }

        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #ffffff !important; }
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
                generation_config={"temperature": 0.2, "max_output_tokens": 1024}
            )
            st.sidebar.success("✅ AI Engine Active")
    except Exception as e:
        st.sidebar.error(f"❌ AI Setup Failed: {str(e)[:50]}")
    
    return sb, ai

supabase, ai_model = init_connections()

if "messages" not in st.session_state: st.session_state.messages = []
if "last_ai_call" not in st.session_state: st.session_state.last_ai_call = 0

# =================================================================
# 2. ANALYTICS TOOLS
# =================================================================
@st.cache_resource
def run_forecast_model(df, periods, freq):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

def calculate_insights(hist_data, forecast_data, horizon, curr_sym):
    hist_with_date = hist_data.set_index('ds')
    
    insights = {
        'hist_total': hist_data['y'].sum(),
        'hist_avg': hist_data['y'].mean(),
        'daily_avg': hist_with_date['y'].resample('D').sum().mean(),
        'monthly_total': hist_with_date['y'].resample('MS').sum(),
        'weekly_total': hist_with_date['y'].resample('W').sum(),
        'forecast_total': forecast_data['yhat'].tail(horizon).sum(),
        'forecast_avg': forecast_data['yhat'].tail(horizon).mean(),
        'growth_rate': (((forecast_data['yhat'].tail(horizon).sum()) - hist_data['y'].sum()) / hist_data['y'].sum() * 100) if hist_data['y'].sum() > 0 else 0
    }
    return insights

# =================================================================
# 3. UI LAYOUT
# =================================================================
if os.path.exists("assets/Hope tech 2.png"):
    st.image("assets/Hope tech 2.png", width=120)

st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> Help us scale {PRODUCT_NAME}. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Project Configuration")
    project_name = st.text_input("Project Name:", value="Sample Project")
    currency_lookup = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£"}
    curr_sym = currency_lookup[st.selectbox("Currency:", options=list(currency_lookup.keys()))]
    
    if st.button("🗑️ Reset All Cache"):
        st.session_state.clear()
        st.rerun()

    with st.expander("🔒 Admin"):
        admin_key = st.text_input("Key", type="password")
        if admin_key == "Ibiene2003#":
            st.write("Admin Mode Active")

# =================================================================
# 4. DATA PROCESSING
# =================================================================
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Engine</p>', unsafe_allow_html=True)
col_left, col_right = st.columns([2, 1.5])

with col_left:
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df_input = pd.read_csv(file)
        st.dataframe(df_input.head(3), use_container_width=True)
        u_date = st.selectbox("Date Column:", df_input.columns)
        u_val = st.selectbox("Value Column:", df_input.columns)
        
        c1, c2 = st.columns(2)
        freq_label = c1.selectbox("Interval:", ["Monthly", "Weekly", "Daily"])
        horizon = c2.number_input("Horizon:", min_value=1, value=12)
        
        if st.button("🚀 Run Intelligence", type="primary"):
            working_df = df_input[[u_date, u_val]].copy().rename(columns={u_date: 'ds', u_val: 'y'})
            working_df['ds'] = pd.to_datetime(working_df['ds'])
            working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()
            
            freq_map = {"Monthly": "MS", "Weekly": "W", "Daily": "D"}
            f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
            insights = calculate_insights(working_df, f_data, horizon, curr_sym)
            
            st.session_state.update({
                'history': working_df, 'forecast': f_data, 'model': f_model,
                'insights': insights, 'analyzed': True, 'horizon': horizon, 'freq_label': freq_label
            })
            st.success("Data Synthesized.")

# =================================================================
# 5. DATA-AWARE AI CHAT
# =================================================================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

    if st.session_state.get('analyzed'):
        if query := st.chat_input("Ask about sales..."):
            st.session_state.messages.append({"role": "user", "content": query})
            
            ins = st.session_state['insights']
            hist_df = st.session_state['history']
            
            # This is the "Data Cheat Sheet" for the AI
            context = f"""You are a professional Data Analyst for {BRAND_NAME}.
            
            PROJECT: {project_name}
            - Daily Avg Sales: {curr_sym}{ins['daily_avg']:,.2f}
            - Total Historical: {curr_sym}{ins['hist_total']:,.2f}
            
            RECENT DAILY SAMPLES:
            {hist_df.tail(10).to_string(index=False)}
            
            MONTHLY BREAKDOWN:
            {ins['monthly_total'].to_string()}
            
            FORECAST ({st.session_state['horizon']} {st.session_state['freq_label']}s):
            - Projected Total: {curr_sym}{ins['forecast_total']:,.2f}
            - Growth Trend: {ins['growth_rate']:,.2f}%
            
            Question: {query}
            Instruction: Use the specific numbers above. If asked for a sum, total, or average, look at the values provided."""

            try:
                response = ai_model.generate_content(context)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
            except:
                st.error("AI Busy. Try again in a moment.")
    else:
        st.info("Upload data to start AI chat.")
    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================
# 6. DASHBOARD
# =================================================================
if st.session_state.get('analyzed'):
    st.divider()
    fcst = st.session_state['forecast']
    hist = st.session_state['history']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name="Historical", line=dict(color="#ffffff", width=2)))
    fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name="Forecast", line=dict(color="#00B0F6", width=4, dash='dot')))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="interpretation-box">
        <b>🔍 Executive Analysis:</b> Based on {len(hist)} data points, {PRODUCT_NAME} identifies a 
        <b>{st.session_state['insights']['growth_rate']:.1f}%</b> growth trajectory. 
        Total expected volume: <b>{curr_sym}{st.session_state['insights']['forecast_total']:,.2f}</b>.
    </div>
    """, unsafe_allow_html=True)

# =================================================================
# 7. FEEDBACK FORM
# =================================================================
st.divider()
with st.form("feedback"):
    st.write("### ✉️ Contact Engineer")
    e = st.text_input("Email")
    m = st.text_area("Message")
    if st.form_submit_button("Send"):
        if supabase and e and m:
            supabase.table("feedback").insert({"email": e, "message": m}).execute()
            st.success("Sent!")
