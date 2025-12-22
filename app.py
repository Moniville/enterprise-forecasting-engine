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
    .glass-card {
        background: rgba(255, 255, 255, 0.05); border-radius: 12px;
        padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .main-title { font-size: 42px; font-weight: bold; color: #00B0F6; margin-top: -10px; }
    .interpretation-box { 
        background: rgba(0, 176, 246, 0.08); padding: 25px; 
        border-radius: 12px; border: 1px solid #00B0F6; margin-top: 20px;
    }
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
            ai = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Setup Error: {e}")
    return sb, ai

supabase, ai_model = init_connections()

# --- 2. ANALYTICS & PDF UTILITIES ---
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
    pdf.cell(200, 10, txt=f"{PRODUCT_NAME} Executive Summary", ln=True, align='C')
    pdf.ln(10)
    display_curr = curr_sym if curr_sym not in ["GH₵", "₦", "د.إ", "﷼"] else curr_name.split(" ")[0]
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Key Metrics:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"- Lifetime History: {display_curr}{hist_total:,.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Projected Revenue: {display_curr}{proj_total:,.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Trend Status: {status.upper()} ({growth_pct:.1f}%)", ln=True)
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 3. TOP LEVEL SUPPORT ---
st.markdown(f'<div class="support-bar">🚀 <b>Support Zenith Innovation:</b> Help us scale {PRODUCT_NAME}. <a href="https://selar.com/showlove/hopetech" target="_blank" style="color: #0e1117; text-decoration: underline; margin-left: 10px;">Click to Tip/Donate</a></div>', unsafe_allow_html=True)

# --- 4. SIDEBAR ---
currency_lookup = {
    "USD ($)": "$", "NGN (₦)": "₦", "EUR (€)": "€", "GBP (£)": "£",
    "GHS (GH₵)": "GH₵", "AED (د.إ)": "DH ", "SAR (﷼)": "SR "
}

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
    st.title("🔐 Internal Intelligence")
    if supabase:
        fb = supabase.table("feedback").select("*").execute()
        st.write("### Recent Feedback")
        st.dataframe(pd.DataFrame(fb.data))
    if st.button("Logout Admin"): st.rerun()
    st.stop()

# --- 5. DATA INGESTION ---
st.markdown(f'<p class="main-title">{PRODUCT_NAME} Forecast Engine</p>', unsafe_allow_html=True)
col_l, col_r = st.columns([2.5, 1])

with col_l:
    df_input = None
    if input_method == "CSV Upload":
        file = st.file_uploader("Upload Business Data", type="csv")
        if file:
            df_input = pd.read_csv(file)
            st.subheader("📋 Step 1: Data Preview")
            st.dataframe(df_input.head(5), use_container_width=True)
            c1, c2 = st.columns(2)
            u_date = c1.selectbox("Date Column:", df_input.columns)
            u_val = c2.selectbox("Value Column:", df_input.columns)
    else:
        manual = st.text_area("Paste Data (Comma Separated):")
        if manual:
            df_input = pd.DataFrame({"y": [float(x.strip()) for x in manual.split(",")]})

    if df_input is not None:
        c_a, c_b = st.columns(2)
        freq_label = c_a.selectbox("Frequency:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
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
                working_df['ma'] = working_df['y'].rolling(window=ma_window, min_periods=1).mean()
                
                with st.spinner("Analyzing Trends..."):
                    freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
                    f_data, f_model = run_forecast_model(working_df, horizon, freq_map[freq_label])
                    st.session_state.update({'forecast': f_data, 'model': f_model, 'history': working_df, 'analyzed': True})
            except Exception as e: st.error(f"Logic Error: {e}")

with col_r:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Pulse AI Analyst")
    if st.session_state.get('analyzed') and ai_model:
        query = st.text_input("Question about trends:")
        if query:
            try:
                ctx = f"Historical Total: {st.session_state['history']['y'].sum()}. Question: {query}"
                response = ai_model.generate_content(ctx)
                st.info(response.text)
            except Exception as e: st.error(f"AI Connection Issues: {e}")
    else: st.write("Awaiting data analysis...")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. VISUALIZATION & INTERPRETATION ---
if st.session_state.get('analyzed'):
    hist, fcst, model = st.session_state['history'], st.session_state['forecast'], st.session_state['model']
    future_only = fcst.tail(horizon)
    proj_sum = future_only['yhat'].sum()
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Historical Volume", f"{curr_sym}{hist['y'].sum():,.2f}")
    m2.metric(f"Avg per {freq_label}", f"{curr_sym}{hist['y'].mean():,.2f}")
    m3.metric("Projected Total", f"{curr_sym}{proj_sum:,.2f}")

    view = st.radio("View Mode:", ["AI Strategic Forecast", "Anomaly Detector", "Model Performance", "Monthly History", "Weekly Patterns", "Annual Growth"], horizontal=True)
    fig = go.Figure()

    if view == "AI Strategic Forecast":
        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers+text', line=dict(color='#00B0F6', width=4), text=[f"{curr_sym}{x:,.0f}" for x in future_only['yhat']], textposition="top center", name="Forecast"))
    elif view == "Anomaly Detector":
        perf = fcst.set_index('ds')[['yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
        anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual', line=dict(color='#FFFFFF')))
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
    elif view == "Model Performance":
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Actual'))
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['ma'], name='Moving Avg', line=dict(color='#00FFCC')))
    elif view == "Monthly History":
        monthly = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
        fig.add_trace(go.Bar(x=monthly['ds'], y=monthly['y'], marker_color="#636EFA"))
    elif view == "Weekly Patterns":
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        sample_week = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=7)})
        weekly_comp = model.predict(sample_week)[['ds', 'weekly']]
        fig.add_trace(go.Bar(x=days, y=weekly_comp['weekly'], marker_color='#00FFCC'))
    elif view == "Annual Growth":
        yearly = hist.set_index('ds').resample('YS')['y'].sum().reset_index()
        fig.add_trace(go.Scatter(x=yearly['ds'], y=yearly['y'], mode='lines+markers', line=dict(color="#EF553B")))

    fig.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)

    # NATURAL LANGUAGE INTERPRETATION
    st.subheader("💡 Strategic Insights")
    start_v, end_v = future_only['yhat'].iloc[0], future_only['yhat'].iloc[-1]
    growth_pct = ((end_v - start_v) / start_v) * 100 if start_v != 0 else 0
    status = "growth" if growth_pct > 0 else "contraction"
    
    st.markdown(f"""
    <div class="interpretation-box">
    The engine forecasts a period of <b>{status}</b> with an overall change of <b>{growth_pct:.1f}%</b>. 
    Total expected volume is <b>{curr_sym}{proj_sum:,.2f}</b>. 
    {"Upward momentum suggests scaling operations." if growth_pct > 0 else "Downward trend suggests cost optimization."}
    </div>
    """, unsafe_allow_html=True)

    # Exports
    ex1, ex2 = st.columns(2)
    with ex1: st.download_button("📥 Download Data", fcst.to_csv(index=False), "forecast.csv")
    with ex2:
        pdf_b = create_pdf_report(hist['y'].sum(), hist['y'].mean(), proj_sum, status, growth_pct, freq_label, curr_sym, selected_curr_name)
        st.download_button("📥 PDF Report", pdf_b, "summary.pdf")

# --- 7. FOOTER & FEEDBACK ---
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
f_left, f_right = st.columns(2)
with f_left:
    st.markdown("### 👨‍💻 About the Engineer")
    st.write("**Monivi Hope** | Lead at **Hope Tech**")
    st.write("Crafting automated intelligence for the next generation of business.")
    st.markdown("[🔗 View My Full Portfolio](https://linktr.ee/MoniviHope)")
with f_right:
    st.markdown("### ✉️ Contact & Feedback")
    with st.form("feedback_box"):
        email = st.text_input("Email Address")
        msg = st.text_area("Message / Feature Request")
        if st.form_submit_button("Send to Monivi"):
            if supabase and email and msg:
                supabase.table("feedback").insert({"email": email, "message": msg}).execute()
                st.success("Message received. Thank you!")
            else: st.error("Incomplete fields or DB offline.")
st.markdown('</div>', unsafe_allow_html=True)
