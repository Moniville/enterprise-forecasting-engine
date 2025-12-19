import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import io

from supabase import create_client, Client

# ACCESSING SUPABASE CONFIGURATIONS FROM STREAMLIT SECRETS
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
except:
    # This allows us to run it locally  if we have a .env file
    url = "Your_Local_Test_URL"
    key = "Your_Local_Test_Key"
superbase: Client = create_client(url, key)

# --- 1. ANALYTICS ENGINE ---
@st.cache_resource
def run_forecast_model(df, periods, freq):
    """
    Trains the Prophet model using historical data to project future performance.
    """
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# --- 2. PDF REPORT GENERATOR ---
def create_pdf_report(hist_total, avg_val, proj_total, status, growth_pct, freq_label):
    """
    Generates a professional PDF executive summary.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Executive Forecast Summary", ln=True, align='C')
    pdf.ln(10)
    
    # KPIs Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Key Performance Indicators:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"- Lifetime Historical Amount: ${hist_total:,.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Average Amount per {freq_label}: ${avg_val:,.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Total Projected Amount (Horizon): ${proj_total:,.2f}", ln=True)
    pdf.ln(5)

    # Strategy Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Strategic AI Insights:", ln=True)
    pdf.set_font("Arial", '', 11)
    insight_text = (f"The AI model predicts a market {status} of approximately {abs(growth_pct):.1f}% "
                    f"over the next horizon. This projection includes an automated scan for historical anomalies.")
    pdf.multi_cell(0, 8, txt=insight_text)
    
    return bytes(pdf.output())

# --- 3. UI INITIALIZATION ---
st.set_page_config(page_title="Enterprise Forecasting", layout="wide")
st.title("📈 Strategic Enterprise Forecasting")

with st.sidebar:
    st.header("1. Administration")
    input_method = st.radio("Data Source:", ["CSV Upload", "Manual Entry"])
    if st.button("🔄 Reset System"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# --- 4. DATA INGESTION ---
df_input = None

if input_method == "CSV Upload":
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df_input = pd.read_csv(file)
        st.subheader("📋 Step 1: Data Schema")
        st.dataframe(df_input.head(3), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: user_date_col = st.selectbox("Timeline Column:", df_input.columns, key="date_sel")
        with c2: user_val_col = st.selectbox("Value Column:", df_input.columns, key="val_sel")
else:
    st.subheader("📋 Step 1: Manual Entry")
    manual_data = st.text_area("Input Values (comma separated):", placeholder="140000, 200000, 420, 5000")
    if manual_data:
        try:
            vals = [float(x.strip()) for x in manual_data.split(",")]
            df_input = pd.DataFrame({"y": vals})
            st.success(f"Captured {len(vals)} data points.")
        except ValueError: st.error("Numbers and commas only.")

# --- 5. CONFIGURATION & EXECUTION ---
if df_input is not None:
    st.subheader("⚙️ Step 2: Analysis Configuration")
    col_a, col_b = st.columns(2)
    with col_a:
        freq_label = st.selectbox("Interval:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        freq_map = {"Yearly": "YS", "Monthly": "MS", "Weekly": "W", "Daily": "D"}
    with col_b:
        horizon = st.number_input(f"Horizon ({freq_label}):", min_value=1, value=24)

    if st.button("🚀 Execute Analysis", type="primary"):
        try:
            if input_method == "CSV Upload":
                working_df = df_input[[user_date_col, user_val_col]].copy().rename(columns={user_date_col: 'ds', user_val_col: 'y'})
                working_df['ds'] = pd.to_datetime(working_df['ds'], errors='coerce')
            else:
                working_df = df_input.copy()
                # STABILITY FIX: Lock dates to midnight to prevent shifting results
                fixed_today = pd.Timestamp.today().normalize()
                working_df['ds'] = pd.date_range(end=fixed_today, periods=len(working_df), freq=freq_map[freq_label])
            
            working_df = working_df.dropna(subset=['ds', 'y'])
            with st.spinner("Calculating Business Projections..."):
                st.session_state['forecast'] = run_forecast_model(working_df, horizon, freq_map[freq_label])
                st.session_state['history'] = working_df
                st.session_state['analyzed'] = True
        except Exception as e: st.error(f"Analysis Error: {e}")

    # --- 6. VISUALIZATION DASHBOARD ---
    if st.session_state.get('analyzed'):
        st.divider()
        hist, fcst = st.session_state['history'], st.session_state['forecast']
        
        future_only = fcst.tail(horizon)
        projected_sum = future_only['yhat'].sum()
        
        start_val = future_only['yhat'].iloc[0]
        end_val = future_only['yhat'].iloc[-1]
        growth_pct = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
        status = "increase" if growth_pct > 0 else "decrease"

        m1, m2, m3 = st.columns(3)
        m1.metric("Lifetime Amount", f"${hist['y'].sum():,.2f}")
        m2.metric(f"Avg per {freq_label}", f"${hist['y'].mean():,.2f}")
        m3.metric("Projected Amount (Horizon)", f"${projected_sum:,.2f}")

        st.write("### 📊 Business Intelligence Perspectives")
        view = st.radio("Switch View:", ["AI Strategic Forecast", "Anomaly Detector", "Monthly History", "Annual Growth"], horizontal=True)
        fig = go.Figure()

        if view == "AI Strategic Forecast":
            fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], mode='lines', line=dict(width=0), fill='toself', fillcolor='rgba(0,176,246,0.1)', name="Range of Probability"))
            fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers+text', line=dict(color='#00B0F6', width=4), text=[f"${x:,.0f}" for x in future_only['yhat']], textposition="top center", name="AI Prediction"))
            fig.update_layout(title=f"{horizon}-Period Future Roadmap")
        
        elif view == "Anomaly Detector":
            perf = fcst.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(hist.set_index('ds'))
            anoms = perf[(perf['y'] > perf['yhat_upper']) | (perf['y'] < perf['yhat_lower'])]
            
            # Shaded "Safe Zone" for business users
            fig.add_trace(go.Scatter(x=perf.index, y=perf['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=perf.index, y=perf['yhat_lower'], mode='lines', line=dict(width=0), fill='toself', fillcolor='rgba(200,200,200,0.1)', name="Expected Range"))
            fig.add_trace(go.Scatter(x=perf.index, y=perf['y'], mode='lines', name='Actual Performance', line=dict(color='#FFFFFF', width=1)))
            fig.add_trace(go.Scatter(x=anoms.index, y=anoms['y'], mode='markers', name='Significant Deviation', marker=dict(color='#FF4B4B', size=10, line=dict(color='white', width=1))))

            if not anoms.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Anomalies Found", len(anoms))
                c2.metric("Highest Spike", f"${anoms['y'].max():,.0f}")
                c3.metric("Deepest Dip", f"${anoms['y'].min():,.0f}")
                st.dataframe(anoms[['y']].rename(columns={'y': 'Irregular Amount'}).style.format("${:,.2f}"), use_container_width=True)
            else:
                st.success("No significant business anomalies detected.")

        elif view == "Monthly History":
            monthly = hist.set_index('ds').resample('MS')['y'].sum().reset_index()
            fig.add_trace(go.Bar(x=monthly['ds'], y=monthly['y'], marker_color="#636EFA", text=[f"${x:,.0f}" for x in monthly['y']], textposition="outside", name="Monthly Total"))
            fig.update_layout(title="Historical Monthly Volume (Aggregated)")

        elif view == "Annual Growth":
            yearly = hist.set_index('ds').resample('YS')['y'].sum().reset_index()
            fig.add_trace(go.Scatter(x=yearly['ds'], y=yearly['y'], mode='lines+markers+text', line=dict(color="#EF553B", width=4), text=[f"${x:,.0f}" for x in yearly['y']], textposition="top center", name="Annual Performance"))
            fig.update_layout(title="Year-over-Year Trend")

        fig.update_layout(template="plotly_dark", height=600, hovermode="x unified", xaxis=dict(autorange=True, nticks=20))
        st.plotly_chart(fig, use_container_width=True)

        # --- 7. EXPORT & PLAIN ENGLISH INSIGHTS ---
        st.subheader("📥 Export Reports")
        ex1, ex2 = st.columns(2)
        with ex1:
            csv = fcst.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Data", data=csv, file_name='forecast_data.csv', mime='text/csv')
        with ex2:
            pdf_bytes = create_pdf_report(hist['y'].sum(), hist['y'].mean(), projected_sum, status, growth_pct, freq_label)
            st.download_button("Download PDF Summary", data=pdf_bytes, file_name='executive_report.pdf', mime='application/pdf')

        st.divider()
        st.subheader("💡 Strategic Insights for Management")
        with st.expander("How to interpret this data", expanded=True):
            st.write(f"""
            * **Visual Coverage:** This chart displays the full **{horizon} {freq_label.lower()}** horizon.
            * **Anomaly Detection:** Red markers highlight points that significantly deviated from the expected statistical range.
            * **Prediction Logic:** The AI expects a **{status}** toward **${end_val:,.2f}**.
            * **Total Projected Amount:** The cumulative sum of the future points is **${projected_sum:,.2f}**.
            """)