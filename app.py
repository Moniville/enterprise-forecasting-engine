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

# --- CUSTOM CSS: DARK THEME ---
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
# 1. SYSTEM INITIALIZATION (Connect to AI & Database)
# =================================================================
def init_connections():
    """Initialize Supabase and Google AI with error handling."""
    sb, ai = None, None
    # Supabase
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            st.sidebar.success("✅ Database Connected")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Database: {str(e)[:50]}")

    # Google AI
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
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
            st.sidebar.info("👉 Add your API key in Streamlit Secrets")
    except Exception as e:
        st.sidebar.error(f"❌ AI Setup Failed: {str(e)[:50]}")
        ai = None
    return sb, ai

supabase, ai_model = init_connections()

# Initialize session state variables
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
    """Calculate comprehensive insights from data."""
    insights = {
        'hist_total': hist_data['y'].sum(),
        'hist_avg': hist_data['y'].mean(),
        'hist_max': hist_data['y'].max(),
        'hist_min': hist_data['y'].min(),
        'forecast_total': forecast_data['yhat'].tail(horizon).sum(),
        'forecast_avg': forecast_data['yhat'].tail(horizon).mean(),
        'forecast_max': forecast_data['yhat'].tail(horizon).max(),
        'forecast_min': forecast_data['yhat'].tail(horizon).min()
    }
    # Growth rate
    if insights['hist_total'] != 0:
        insights['growth_rate'] = ((insights['forecast_total'] - insights['hist_total']) / insights['hist_total']) * 100
    else:
        insights['growth_rate'] = 0

    # Weekly, monthly, yearly breakdowns
    hist_with_date = hist_data.set_index('ds')
    insights['daily_avg'] = hist_with_date['y'].resample('D').mean()
    insights['weekly_breakdown'] = {f"Week of {date.strftime('%Y-%m-%d')}": value for date, value in hist_with_date['y'].resample('W').sum().items()}
    insights['monthly_breakdown'] = {date.strftime('%B %Y'): value for date, value in hist_with_date['y'].resample('MS').sum().items()}
    insights['yearly_breakdown'] = {date.strftime('%Y'): value for date, value in hist_with_date['y'].resample('YS').sum().items()}

    return insights

# =================================================================
# 3. UI LAYOUT & SIDEBAR
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
# 4. DATA INGESTION & PROCESSING
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
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    freq_code = freq_map[freq_label]
                    working_df = df_input.copy()
                    working_df['ds'] = pd.date_range(end=datetime.now(), periods=len(working_df), freq=freq_code)
                working_df = working_df.dropna().sort_values('ds').groupby('ds')['y'].sum().reset_index()

                # Run forecast and store in session state
                with st.spinner("AI Engine executing..."):
                    forecast = pd.DataFrame()  # initialize forecast variable
                    forecast, model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    # Save forecast to session state
                    st.session_state['forecast'] = forecast
                    st.session_state['model'] = model
                    st.session_state['history'] = working_df
                    st.session_state['analyzed'] = True
                    st.session_state['horizon'] = horizon
                    st.session_state['freq_label'] = freq_label
                    # Insights
                    insights = calculate_insights(working_df, forecast, horizon, curr_sym)
                    st.session_state['insights'] = insights
                    st.session_state['project_name'] = project_name
                    st.session_state['curr_sym'] = curr_sym
            except Exception as e:
                st.error(f"Forecast error: {e}")

# =================================================================
# 5. AI Chat with Data Insights
# =================================================================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.get('analyzed') and "forecast" in st.session_state and "insights" in st.session_state and ai_model:
        if query := st.chat_input("Ask about your projections..."):
            # Prevent spam
            current_time = time.time()
            if current_time - st.session_state.last_ai_call < 2:
                st.warning("⏳ Please wait a moment before sending another message.")
            else:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                # Retrieve data for context
                hist = st.session_state['history']
                forecast = st.session_state['forecast']
                insights = st.session_state['insights']
                project_name = st.session_state.get('project_name', 'Your Project')
                horizon = st.session_state.get('horizon', 12)
                freq_label = st.session_state.get('freq_label', 'Monthly')
                curr_sym = st.session_state.get('curr_sym', '$')

                # Build prompt
                def build_prompt(insights, project_name, brand_name, project_full_name, curr_sym, user_query):
                    monthly_details = "\n".join([f"  - {month}: {curr_sym}{safe_format_number(value):,.2f}" for month, value in insights.get('monthly_breakdown', {}).items()])
                    weekly_details = "\n".join([f"  - {week}: {curr_sym}{safe_format_number(value):,.2f}" for week, value in list(insights.get('weekly_breakdown', {}).items())[:10]])
                    prompt = f"""
You are an expert data analyst for {brand_name}. Your task is to interpret user questions based on the provided data summaries and insights.

PROJECT: {project_full_name}

HISTORICAL DATA SUMMARY:
- Total Sales: {curr_sym}{safe_format_number(insights.get('hist_total', 0)):,.2f}
- Average Sales: {curr_sym}{safe_format_number(insights.get('hist_avg', 0)):,.2f}
- Highest Sales: {curr_sym}{safe_format_number(insights.get('hist_max', 0)):,.2f}
- Lowest Sales: {curr_sym}{safe_format_number(insights.get('hist_min', 0)):,.2f}
- Daily Average: {curr_sym}{safe_format_number(insights.get('daily_avg', 0)):,.2f}

DETAILED MONTHLY BREAKDOWN:
{monthly_details}

DETAILED WEEKLY BREAKDOWN (First 10 weeks):
{weekly_details}

FORECAST DATA ({st.session_state.get('horizon', 12)} {st.session_state.get('freq_label', 'Monthly').lower()}s ahead):
- Projected Total: {curr_sym}{safe_format_number(insights.get('forecast_total', 0)):,.2f}
- Projected Average: {curr_sym}{safe_format_number(insights.get('forecast_avg', 0)):,.2f}
- Projected Highest: {curr_sym}{safe_format_number(insights.get('forecast_max', 0)):,.2f}
- Projected Lowest: {curr_sym}{safe_format_number(insights.get('forecast_min', 0)):,.2f}
- Growth Rate: {insights.get('growth_rate', 0):+.2f}%

User Question: {user_query}

INSTRUCTIONS:
- Answer the user's question based on the provided data.
- Be specific with months, weeks, and numbers.
- Format currency as {curr_sym}X,XXX.XX
- Refer to summaries for historical, forecast, or breakdown data.
- If ambiguous, ask for clarification or give best-guess.
- Cite specific numbers.
"""
                    return prompt

                prompt_text = build_prompt(insights, project_name, BRAND_NAME, project_name, curr_sym, query)

                # Call API with retry logic
                max_retries = 3
                retry_count = 0
                success = False
                while retry_count < max_retries and not success:
                    try:
                        if retry_count > 0:
                            wait_time = 2 ** retry_count
                            with st.spinner(f"Retrying AI connection... (Attempt {retry_count + 1}/{max_retries})"):
                                time.sleep(wait_time)
                        response = ai_model.generate_content(prompt_text)
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
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            # fallback responses
                            query_lower = query.lower()
                            if 'monthly' in query_lower or 'month' in query_lower:
                                if 'sum' in query_lower or 'total' in query_lower:
                                    monthly_list = "\n".join([f"- **{month}**: {curr_sym}{safe_format_number(value):,.2f}" for month, value in insights.get('monthly_breakdown', {}).items()])
                                    monthly_sum = sum([safe_format_number(v) for v in insights.get('monthly_breakdown', {}).values()])
                                    fallback_msg = f"Monthly sales:\n\n{monthly_list}\n\nTotal: {curr_sym}{monthly_sum:,.2f}"
                                elif 'average' in query_lower or 'avg' in query_lower:
                                    monthly_values = [safe_format_number(v) for v in insights.get('monthly_breakdown', {}).values()]
                                    if monthly_values:
                                        avg_value = sum(monthly_values)/len(monthly_values)
                                        fallback_msg = f"Average monthly sales: {curr_sym}{avg_value:,.2f}"
                                    else:
                                        fallback_msg = "No monthly data."
                                else:
                                    fallback_msg = "Unable to get detailed info. Please check your data."
                            elif 'weekly' in query_lower or 'week' in query_lower:
                                weekly_list = "\n".join([f"- {week}: {curr_sym}{value:,.2f}" for week, value in list(insights.get('weekly_breakdown', {}).items())[:10]])
                                fallback_msg = f"Weekly breakdown:\n\n{weekly_list}"
                            elif 'sum' in query_lower or 'total' in query_lower:
                                fallback_msg = f"Total sales: {curr_sym}{insights.get('hist_total', 0):,.2f}"
                            elif 'growth' in query_lower or 'trend' in query_lower:
                                growth = insights.get('growth_rate', 0)
                                fallback_msg = f"Growth rate: {growth:.2f}%"
                            else:
                                fallback_msg = "Sorry, I couldn't retrieve detailed insights. Please try again later."
                            st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                            st.warning("⚠️ AI connection failed, showing fallback info.")
                            st.session_state.last_ai_call = time.time()
                            st.rerun()
                        else:
                            # show progress
                            if "429" in str(e):
                                st.warning(f"Rate limit reached. Retrying in {2 ** retry_count} seconds...")
                            else:
                                st.warning("Connection issue. Retrying...")
                # end of retries
    elif st.session_state.get('analyzed') and not ai_model:
        st.error("❌ AI Engine unavailable. Please check your GOOGLE_API_KEY in Secrets.")
        st.info("💡 Go to: Settings → Secrets → Add GOOGLE_API_KEY")
    else:
        st.info("📊 Upload data and click 'Process Intelligence' to unlock AI chat.")
    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================
# 6. DASHBOARD & DATA OUTPUT
# =================================================================
if st.session_state.get('analyzed'):
    hist = st.session_state['history']
    forecast = st.session_state['forecast']
    insights = st.session_state['insights']
    project_name = st.session_state.get('project_name', 'Your Project')
    horizon = st.session_state.get('horizon', 12)
    freq_label = st.session_state.get('freq_label', 'Monthly')
    curr_sym = st.session_state.get('curr_sym', '$')

    # Plot forecast
    future_only = forecast.tail(horizon)
    perf = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
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
        hist['ma'] = hist['y'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual', opacity=0.4))
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['ma'], name='Trend', line=dict(color='#00FFCC', width=5)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(dash='dot', color='#00B0F6', width=4)))

    elif view == "Monthly":
        monthly = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
        fig.add_trace(go.Bar(x=monthly['ds'], y=monthly['y'], text=[f"{curr_sym}{v:,.2f}" for v in monthly['y']], textposition='auto', marker_color="#636EFA"))

    elif view == "Weekly":
        sample_week = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=7)})
        weekly_pred = st.session_state['model'].predict(sample_week)[['ds', 'weekly']]
        fig.add_trace(go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=weekly_pred['weekly'], marker_color='#00FFCC'))

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

    st.subheader("📝 Executive Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Value", f"{curr_sym}{start_val:,.2f}")
    col2.metric("End Value", f"{curr_sym}{end_val:,.2f}")
    col3.metric("Growth Rate", f"{growth_rate:.2f}%")
