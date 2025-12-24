import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import io
import time
import os
import math  # For safe_format_number
import google.generativeai as genai
from supabase import create_client, Client
import streamlit.components.v1 as components

# Utility function to safely convert values to float
def safe_format_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0

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
# 1. SYSTEM INITIALIZATION (IMPROVED AI CONNECTION)
# =================================================================
def init_connections():
    """Initialize Supabase and Google AI connections with proper error handling"""
    sb, ai = None, None
    
    # Initialize Supabase
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            st.sidebar.success("✅ Database Connected")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Database: {str(e)[:50]}")
    
    # Initialize Google AI with proper configuration
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            # Configure the API
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            
            # Create model instance with safety settings
            ai = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            st.sidebar.success("✅ AI Engine Active: Gemini 1.5 Flash")
        else:
            st.sidebar.error("❌ GOOGLE_API_KEY not found in secrets")
            st.sidebar.info("👉 Add your API key in Streamlit Cloud Secrets")
    except Exception as e:
        st.sidebar.error(f"❌ AI Setup Failed: {str(e)[:50]}")
        st.sidebar.info("Check your Google API Key and quota")
        ai = None
    
    return sb, ai

supabase, ai_model = init_connections()

# Initialize session state
if "last_ai_call" not in st.session_state:
    st.session_state.last_ai_call = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    if df[date_col].isnull().any():
        issues.append("Missing dates detected.")
    if df[val_col].isnull().any():
        issues.append("Missing values in target column.")
    if len(df) < 2:
        issues.append("Insufficient data for forecasting.")
    return issues

def calculate_insights(hist_data, forecast_data, horizon, curr_sym):
    """Calculate comprehensive insights from the data"""
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
    # Calculate growth rate
    if insights['hist_total'] > 0:
        insights['growth_rate'] = ((insights['forecast_total'] - insights['hist_total']) / insights['hist_total']) * 100
    else:
        insights['growth_rate'] = 0
    # Calculate breakdowns
    hist_with_date = hist_data.set_index('ds')
    insights['daily_avg'] = hist_with_date['y'].resample('D').mean()
    insights['weekly_breakdown'] = {f"Week of {date.strftime('%Y-%m-%d')}": value for date, value in hist_with_date['y'].resample('W').sum().items()}
    insights['monthly_breakdown'] = {date.strftime('%B %Y'): value for date, value in hist_with_date['y'].resample('MS').sum().items()}
    insights['yearly_breakdown'] = {date.strftime('%Y'): value for date, value in hist_with_date['y'].resample('YS').sum().items()}
    return insights

# =================================================================
# 3. UI LAYOUT & SIDEBAR CONTROL
# =================================================================
if os.path.exists("assets/Hope tech 2.png"):
    st.image("assets/Hope tech 2.png", width=120)

st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> Help us scale {PRODUCT_NAME}. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    st.header("Project Configuration")
    project_name = st.text_input("Project Namespace:", value="Your Project Name")
    st.caption("💡 *Please remember to name your specific project above.*")
    currency_lookup = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£", "GHS (GH₵)": "G₵"}
    selected_curr_name = st.selectbox("Operational Currency:", options=list(currency_lookup.keys()))
    curr_sym = currency_lookup[selected_curr_name]
    input_method = st.radio("Inbound Data Source:", ["CSV Upload (Recommended)", "Manual Entry"])
    st.divider()
    ma_window = st.slider("Smoothing Window (Days):", 2, 90, 7)
    if st.button("🗑️ Reset All Cache & Chat"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    with st.expander("🔒 Developer Access"):
        admin_key = st.text_input("Security Key", type="password")
        is_admin = (admin_key == "Ibiene2003#")

if is_admin:
    if supabase:
        try:
            fb = supabase.table("feedback").select("*").execute()
            st.write("### Internal Feedback Log")
            st.dataframe(pd.DataFrame(fb.data))
        except:
            st.error("Could not fetch logs.")
    if st.button("End Session"):
        st.rerun()
    st.stop()

# =================================================================
# 4. DATA PROCESSING & INGESTION
# =================================================================
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Analytics Engine</p>', unsafe_allow_html=True)
col_left, col_right = st.columns([2.2, 1.3])

with col_left:
    df_input = None
    u_date, u_val = None, None
    if "CSV Upload" in input_method:
        file = st.file_uploader("Drop dataset here", type="csv")
        if file:
            df_input = pd.read_csv(file)
            st.write("### 📊 Inbound Data Preview")
            st.dataframe(df_input.head(5), use_container_width=True)
            u_date = st.selectbox("Map Date Column:", df_input.columns)
            u_val = st.selectbox("Map Target Value:", df_input.columns)
            health_issues = perform_health_check(df_input, u_date, u_val)
            if health_issues:
                for issue in health_issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ Data Integrity Verified.")
    else:
        manual = st.text_area("Paste comma-separated values (e.g., 100, 200, 150):")
        if manual:
            try:
                vals = [float(x.strip()) for x in manual.split(",") if x.strip()]
                df_input = pd.DataFrame({"y": vals})
            except:
                st.error("Validation Error: Please provide numerical values only.")

    if df_input is not None:
        c1, c2 = st.columns(2)
        freq_label = c1.selectbox("Interval Frequency:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        horizon = c2.number_input(f"Forecasting Horizon ({freq_label}s):", min_value=1, value=12)
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
                with st.spinner("AI Engine executing..."):
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    forecast = None  # Initialize forecast variable
                    try:
                        forecast, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                        # Save forecast to session state
                        st.session_state['forecast'] = forecast
                        st.session_state['model'] = f_model
                        st.session_state['history'] = working_df
                        st.session_state['analyzed'] = True
                        st.session_state['horizon'] = horizon
                        st.session_state['freq_label'] = freq_label
                        # Calculate insights
                        insights = calculate_insights(working_df, forecast, horizon, curr_sym)
                        st.session_state['insights'] = insights
                        st.session_state['project_name'] = project_name
                        st.session_state['curr_sym'] = curr_sym
                    except Exception as e:
                        st.error(f"Forecasting error: {e}")
            except Exception as e:
                st.error(f"Error during processing: {e}")

# =================================================================
# 5. CHAT-STYLE AI ASSISTANT (FULLY FUNCTIONAL WITH DATA INSIGHTS)
# =================================================================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.get('analyzed') and ai_model:
        if query := st.chat_input("Ask about your projections..."):
            # Cooldown to prevent spam
            current_time = time.time()
            if current_time - st.session_state.last_ai_call < 2:
                st.warning("⏳ Please wait a moment before sending another message.")
            else:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"): st.markdown(query)

                # Retrieve data for context
                hist = st.session_state['history']
                forecast = st.session_state['forecast']
                horizon = st.session_state.get('horizon', 12)
                freq_label = st.session_state.get('freq_label', 'Monthly')
                insights = st.session_state.get('insights', {})
                project_name = st.session_state.get('project_name', 'Your Project')
                curr_sym = st.session_state.get('curr_sym', '$')

                # Build detailed prompt for AI
                def build_prompt(insights, project_name, brand_name, project_full_name, curr_sym, user_query):
                    monthly_details = "\n".join([f"  - {month}: {curr_sym}{safe_format_number(value):,.2f}" 
                                                    for month, value in insights.get('monthly_breakdown', {}).items()])
                    weekly_details = "\n".join([f"  - {week}: {curr_sym}{safe_format_number(value):,.2f}" 
                                                 for week, value in list(insights.get('weekly_breakdown', {}).items())[:10]])
                    prompt = f"""
You are an expert data analyst for {brand_name}. Your task is to interpret user questions based on the provided data summaries and insights.

PROJECT: {project_full_name}

HISTORICAL DATA:
- Total Sales: {curr_sym}{safe_format_number(insights.get('hist_total', 0)):,.2f}
- Avg Sales: {curr_sym}{safe_format_number(insights.get('hist_avg', 0)):,.2f}
- Max Sales: {curr_sym}{safe_format_number(insights.get('hist_max', 0)):,.2f}
- Min Sales: {curr_sym}{safe_format_number(insights.get('hist_min', 0)):,.2f}
- Daily Avg: {curr_sym}{safe_format_number(insights.get('daily_avg', 0)):,.2f}

DETAILED MONTHLY BREAKDOWN:
{monthly_details}

DETAILED WEEKLY BREAKDOWN:
{weekly_details}

FORECAST ({st.session_state.get('horizon', 12)} {st.session_state.get('freq_label', 'Monthly').lower()}s):
- Total: {curr_sym}{safe_format_number(insights.get('forecast_total', 0)):,.2f}
- Avg: {curr_sym}{safe_format_number(insights.get('forecast_avg', 0)):,.2f}
- Max: {curr_sym}{safe_format_number(insights.get('forecast_max', 0)):,.2f}
- Min: {curr_sym}{safe_format_number(insights.get('forecast_min', 0)):,.2f}
- Growth: {insights.get('growth_rate', 0):+.2f}%

User Question: {user_query}

INSTRUCTIONS:
- Answer the user's question using the data summaries above.
- Be specific, cite numbers.
- Use only text, not code/json.
"""
                    return prompt

                prompt_text = build_prompt(insights, project_name, BRAND_NAME, project_name, curr_sym, query)

                # Retry logic with exponential backoff for API call
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        if retry_count > 0:
                            wait_time = 2 ** retry_count
                            with st.spinner(f"Retrying AI connection... (Attempt {retry_count + 1}/{max_retries})"):
                                time.sleep(wait_time)
                        # Call the API
                        response = ai_model.generate_content(prompt_text)
                        # Parse response
                        if hasattr(response, 'text'):
                            ai_text = response.text
                        elif hasattr(response, 'parts'):
                            ai_text = response.parts[0].text
                        else:
                            ai_text = str(response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_text})
                        st.session_state.last_ai_call = time.time()
                        success = True
                        st.rerun()
                    except:
                        retry_count += 1
                        if retry_count >= max_retries:
                            # fallback response
                            fallback_msg = f"Sorry, I couldn't get a response. Here's a summary: {insights.get('hist_total',0):,.2f} total sales expected."
                            st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                            st.warning("⚠️ AI is busy. Showing summary instead.")
                            st.session_state.last_ai_call = time.time()
                            st.rerun()

# =================================================================
# 6. VISUALIZATION DASHBOARD
# =================================================================
if st.session_state.get('analyzed'):
    # Retrieve dataframes and insights
    hist = st.session_state['history']
    fcst = st.session_state['forecast']
    insights = st.session_state['insights']
    project_name = st.session_state.get('project_name', 'Your Project')
    horizon = st.session_state.get('horizon', 12)
    freq_label = st.session_state.get('freq_label', 'Monthly')
    curr_sym = st.session_state.get('curr_sym', '$')

    # Generate forecast plot
    future_only = fcst.tail(horizon)
    perf = fcst.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
    anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]

    view = st.radio("Dashboard Perspective:", ["Forecast", "Anomalies", "Accuracy", "Monthly", "Weekly", "Annual"], horizontal=True)
    fig = go.Figure()

    if view == "Forecast":
        fig.add_trace(go.Scatter(
            x=future_only['ds'], y=future_only['yhat'], mode='lines+markers+text',
            text=[f"{curr_sym}{v:,.0f}" for v in future_only['yhat']],
            textposition="top center", line=dict(color='#00B0F6', width=5), name="Prediction"))
        fig.add_trace(go.Scatter(
            x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', fillcolor='rgba(0,176,246,0.1)',
            line=dict(width=0), name="Confidence Interval"))

    elif view == "Anomalies":
        a1, a2, a3 = st.columns(3)
        a1.metric("Irregularities Found", len(anoms))
        a2.metric("Highest Spike", f"{curr_sym}{hist['y'].max():,.2f}")
        a3.metric("Lowest Dip", f"{curr_sym}{hist['y'].min():,.2f}")
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Historical Data', line=dict(width=4)))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers',
                                 marker=dict(color='red', size=15, symbol='x'), name='Anomalous Point'))

    elif view == "Accuracy":
        hist_preds = fcst[fcst['ds'].isin(hist['ds'])]
        hist['ma'] = hist['y'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual', opacity=0.4))
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['ma'], name='Trend', line=dict(color='#00FFCC', width=5)))
        fig.add_trace(go.Scatter(x=hist_preds['ds'], y=hist_preds['yhat'], name='AI Backtest', line=dict(dash='dot', color='#00B0F6')))
    elif view == "Monthly":
        monthly = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
        fig.add_trace(go.Bar(x=monthly['ds'], y=monthly['y'], text=[f"{curr_sym}{v:,.0f}" for v in monthly['y']], textposition='auto', marker_color="#636EFA"))
    elif view == "Weekly":
        sample_week = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=7)})
        weekly_comp = st.session_state['model'].predict(sample_week)[['ds', 'weekly']]
        fig.add_trace(go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=weekly_comp['weekly'], marker_color='#00FFCC'))
    elif view == "Annual":
        yearly = hist.set_index('ds').resample('Y')['y'].sum().reset_index()
        fig.add_trace(go.Scatter(x=yearly['ds'], y=yearly['y'], mode='lines+markers+text',
                                 text=[f"{curr_sym}{v:,.0f}" for v in yearly['y']],
                                 textposition="top left", line=dict(color="#EF553B", width=6)))

    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0e1117', height=450)
    st.plotly_chart(fig, use_container_width=True)

    # --- EXECUTIVE SUMMARY ---
    start_val, end_val = future_only['yhat'].iloc[0], future_only['yhat'].iloc[-1]
    growth_rate = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
    total_vol = future_only['yhat'].sum()
    st.markdown(f"""
    <div class="interpretation-box">
        <b>🔍 Executive Summary for {project_name}</b><br>
        Over the next {horizon} {freq_label.lower()}s, the AI predicts a total volume of <b>{curr_sym}{total_vol:,.2f}</b>. 
        We are seeing <b>{"upward momentum" if growth_rate > 0 else "a cooling period"}</b> with a projected movement of <b>{growth_rate:.1f}%</b>.
    </div>
    """, unsafe_allow_html=True)

# =================================================================
# 7. FOOTER & FEEDBACK SYSTEM
# =================================================================
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
f_left, f_right = st.columns(2)
with f_left:
    st.markdown("### 👨‍💻 Engineer's Profile")
    st.write(f"**Monivi Hope** | Lead at **{BRAND_NAME}**")
    st.write("Data & Analytics Engineer, building solutions to help build a better humanity.")
    st.markdown("[🔗 Digital Portfolio](https://linktr.ee/MoniviHope)")
with f_right:
    st.markdown("### ✉️ Support Gateway")
    with st.form("feedback_system", clear_on_submit=True):
        email_in = st.text_input("Contact Email")
        msg_in = st.text_area("Observations / Request")
        if st.form_submit_button("Submit Ticket"):
            if supabase and email_in and msg_in:
                try:
                    supabase.table("feedback").insert({"email": email_in, "message": msg_in}).execute()
                    st.success("Ticket submitted.")
                except:
                    st.error("Database submission failed.")
st.markdown(f'<div class="support-bar">💖 <b>Empower Hope Tech:</b> <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)
