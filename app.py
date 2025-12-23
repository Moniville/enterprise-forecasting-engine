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

# --- 0. BRANDING & UI CONFIG ---
PRODUCT_NAME = "Pulse AI"
BRAND_NAME = "Hope Tech"

st.set_page_config(page_title=f"{PRODUCT_NAME} | {BRAND_NAME}", layout="wide")

# --- 0.2 BULLETPROOF GOOGLE ANALYTICS ---
GA_ID = "G-2XRSHF2S9F"

# This script injects the tracker into the MAIN window, not just the iframe
ga_injection = f"""
    <script>
        const script = window.parent.document.createElement('script');
        script.async = true;
        script.src = 'https://www.googletagmanager.com/gtag/js?id={GA_ID}';
        window.parent.document.head.appendChild(script);

        window.parent.dataLayer = window.parent.dataLayer || [];
        function gtag(){{window.parent.dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_ID}', {{
            'page_path': window.parent.location.pathname,
            'debug_mode': true
        }});
        console.log("Pulse AI: Google Analytics Injected to Parent");
    </script>
"""
import streamlit.components.v1 as components
components.html(ga_injection, height=0, width=0)

# Professional Dark-Mode Styling (FORCED)
st.markdown("""
    <style>
        /* 1. Force the absolute background of the entire app */
        .stAppViewMain, .stApp {
            background-color: #0e1117 !important;
            color: #ffffff !important;
        }
        
        /* 2. Style the sidebar to match */
        [data-testid="stSidebar"] {
            background-color: #1a1c23 !important;
        }

        /* 3. Fix text visibility in input boxes */
        input, textarea, select {
            color: #ffffff !important;
            background-color: #262730 !important;
        }

        /* Your existing support bar & cards */
        .support-bar {
            background: linear-gradient(90deg, #00B0F6, #00FFCC);
            padding: 12px; border-radius: 8px; text-align: center;
            margin-bottom: 25px; color: #0e1117; font-weight: bold; font-size: 16px;
        }
        .glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
        .main-title { font-size: 42px; font-weight: bold; color: #00B0F6; margin-top: 0px; margin-bottom: 5px; }
        .interpretation-box { background: rgba(255, 255, 255, 0.05); padding: 25px; border-radius: 12px; border-left: 5px solid #00B0F6; margin-top: 20px; }
        .footer-section { padding: 40px; background: rgba(255,255,255,0.02); border-radius: 15px; margin-top: 50px; border: 1px solid rgba(255,255,255,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. SYSTEM INITIALIZATION ---

def init_connections():
    """ Handles secure connections to Supabase and Google's Generative AI """
    sb, ai = None, None
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if available_models:
                selected_model = next((m for m in available_models if "1.5-flash" in m), available_models[0])
                ai = genai.GenerativeModel(selected_model)
                st.sidebar.success(f"⚡ AI Engine Linked: {selected_model.split('/')[-1]}")
    except Exception as e:
        st.sidebar.warning("System restricted: AI connectivity is currently limited.")
    return sb, ai

supabase, ai_model = init_connections()

# --- 2. FORECASTING & HEALTH TOOLS ---

@st.cache_resource
def run_forecast_model(df, periods, freq):
    """ Uses Facebook Prophet for time-series forecasting. """
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

# --- 3. UI LAYOUT & BRANDING ---

# TOP BRANDING (Logo Only - Positioned above Support Bar)
if os.path.exists("assets/Hope tech 2.png"):
    st.image("assets/Hope tech 2.png", width=120)

# Top Support Banner
st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> Help us scale {PRODUCT_NAME}. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    logo_path = "assets/Hope tech 2.png"
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(logo_path, use_container_width=True)
    else:
        st.markdown(f"## 🛡️ {BRAND_NAME}")
    
    st.divider()
    st.header("Project Configuration")
    
    # Updated Project Namespace with Placeholder and Reminder
    project_name = st.text_input("Project Namespace:", value="Your Project Name")
    st.caption("💡 *Please remember to name your specific project above.*")
    
    currency_lookup = {"USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£", "GHS (GH₵)": "GH₵"}
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
    if st.button("End Session"): st.rerun()
    st.stop()

# --- 4. DATA PROCESSING ---

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
            
            # Health Check Execution
            health_issues = perform_health_check(df_input, u_date, u_val)
            if health_issues:
                for issue in health_issues: st.warning(f"⚠️ {issue}")
            else: st.success("✅ Data Integrity Verified.")
    else:
        manual = st.text_area("Paste comma-separated values:")
        if manual: 
            try:
                vals = [float(x.strip()) for x in manual.split(",") if x.strip()]
                df_input = pd.DataFrame({"y": vals})
            except: st.error("Validation Error: Please provide numerical values only.")

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
                    f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    st.session_state.update({'forecast': f_data, 'model': f_model, 'history': working_df, 'analyzed': True})
            except Exception as e: st.error(f"Computation Error: {e}")

# --- 5. CHAT-STYLE AI ASSISTANT ---
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
            st.session_state.messages.append({"role": "user", "content": query})
            with chat_container:
                with st.chat_message("user"): st.markdown(query)

            hist_data = st.session_state['history']
            forecast_data = st.session_state['forecast']
            
            prompt = f"""
            ANALYST CONTEXT for {project_name}:
            - Historical Total: {curr_sym}{hist_data['y'].sum():,.2f}
            - Forecast Total (next {horizon} {freq_label}s): {curr_sym}{forecast_data['yhat'].tail(horizon).sum():,.2f}
            - Data Range: {hist_data['ds'].min().date()} to {hist_data['ds'].max().date()}
            
            USER QUERY: {query}
            
            INSTRUCTION: Answer as a professional business analyst for {BRAND_NAME}. Do NOT output JSON, code, or charts. 
            Use only text.
            """

            try:
                response = ai_model.generate_content(prompt)
                ai_text = response.text
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                with chat_container:
                    with st.chat_message("assistant"): st.markdown(ai_text)
            except:
                st.error("AI node is momentarily busy.")
    else:
        st.info("Process data to unlock AI chat.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. VISUALIZATION DASHBOARD ---

if st.session_state.get('analyzed'):
    hist, fcst, model = st.session_state['history'], st.session_state['forecast'], st.session_state['model']
    future_only = fcst.tail(horizon)
    
    view = st.radio("Dashboard Perspective:", ["Forecast", "Anomalies", "Accuracy", "Monthly", "Weekly", "Annual"], horizontal=True)
    fig = go.Figure()

    if view == "Forecast":
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers', line=dict(color='#00B0F6', width=4), name="Prediction"))
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', fillcolor='rgba(0,176,246,0.1)', line=dict(width=0), name="Confidence Interval"))
    elif view == "Anomalies":
        perf = fcst.set_index('ds')[['yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
        anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
        a1, a2, a3 = st.columns(3)
        a1.metric("Irregularities", len(anoms))
        a2.metric("Project Peak", f"{curr_sym}{hist['y'].max():,.2f}")
        a3.metric("Project Floor", f"{curr_sym}{hist['y'].min():,.2f}")
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Historical Data'))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
    elif view == "Accuracy":
        hist_preds = fcst[fcst['ds'].isin(hist['ds'])]
        hist['ma'] = hist['y'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual', opacity=0.3))
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['ma'], name='Trend', line=dict(color='#00FFCC')))
        fig.add_trace(go.Scatter(x=hist_preds['ds'], y=hist_preds['yhat'], name='AI Backtest', line=dict(dash='dot', color='#00B0F6')))
    elif view == "Monthly":
        monthly = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
        fig.add_trace(go.Bar(x=monthly['ds'], y=monthly['y'], marker_color="#636EFA"))
    elif view == "Weekly":
        sample_week = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=7)})
        weekly_comp = model.predict(sample_week)[['ds', 'weekly']]
        fig.add_trace(go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=weekly_comp['weekly'], marker_color='#00FFCC'))
    elif view == "Annual":
        yearly = hist.set_index('ds').resample('YS')['y'].sum().reset_index()
        fig.add_trace(go.Scatter(x=yearly['ds'], y=yearly['y'], mode='lines+markers', line=dict(color="#EF553B")))

    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Narrative Summary
    start_val, end_val = future_only['yhat'].iloc[0], future_only['yhat'].iloc[-1]
    growth_rate = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
    st.markdown(f"""
    <div class="interpretation-box">
    <b>Report:</b> {project_name} is projected to generate <b>{curr_sym}{future_only['yhat'].sum():,.2f}</b> in total volume. 
    <br><br>
    <b>Strategic Outlook:</b> Currently trending at a <b>{growth_rate:.1f}% {"Growth" if growth_rate > 0 else "Decline"}</b>. 
    </div>
    """, unsafe_allow_html=True)

# --- 7. FOOTER & ROBUST FEEDBACK ---

st.markdown('<div class="footer-section">', unsafe_allow_html=True)
f_left, f_right = st.columns(2)
with f_left:
    st.markdown("### 👨‍💻 Engineer's Profile")
    st.write(f"**Monivi Hope** | Lead at **{BRAND_NAME}**")
    st.write("Data & Analytics Engineer dedicated to building intelligent solutions for a better world.")
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
                    st.success("Ticket submitted successfully.")
                except:
                    st.error("Database submission failed. Check RLS settings.")
            else: st.error("Incomplete fields.")

# Final Bottom Support Banner
st.markdown(f'<div class="support-bar">💖 <b>Empower Hope Tech:</b> Your support drives our innovation. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)






