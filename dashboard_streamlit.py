import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(
    page_title="POWERGRID | AI Risk Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #0068c9;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0053a0;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("## ⚡")
with col2:
    st.title('POWERGRID — Cost & Timeline Risk AI')
    st.markdown("### Intelligent Project Forecasting System")

st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header('🏗️ Project Parameters')

with st.sidebar.form("prediction_form"):
    project_type = st.selectbox('Project Type', ['substation', 'overhead_line', 'underground_cable'])
    terrain = st.selectbox('Terrain', ['plains', 'hilly', 'forest', 'urban'])
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        planned_days = st.number_input('Planned Days', min_value=10, max_value=2000, value=180)
    with col_sb2:
        planned_cost = st.number_input('Budget (INR)', value=50000000.0, step=1000000.0)
        
    st.sidebar.markdown("---")
    st.sidebar.header('⚠️ Risk Factors')
    
    regulatory_risk = st.select_slider('Regulatory Risk', options=['Low', 'Medium', 'High'], value='Low')
    season = st.selectbox('Season', ['Summer', 'Winter', 'Monsoon'])
    market_condition = st.selectbox('Market Condition', ['Stable', 'Volatile'])
    
    st.sidebar.markdown("---")
    st.sidebar.header('🤝 Vendor Info')
    vendor = st.selectbox('Vendor ID', [f'vendor_{i}' for i in range(1, 21)])
    vendor_rating = st.slider('Vendor Rating (Historical)', 1.0, 5.0, 4.0, 0.1)
    
    submit_btn = st.form_submit_button("🔮 Predict Risk")

# --- Main Content ---

if submit_btn:
    payload = {
        'project_type': project_type,
        'terrain': terrain,
        'planned_days': int(planned_days),
        'planned_cost': float(planned_cost),
        'regulatory_risk': regulatory_risk,
        'season': season,
        'vendor': vendor,
        'vendor_rating': float(vendor_rating),
        'market_condition': market_condition
    }
    
    try:
        res = requests.post('http://localhost:8000/predict_cost_overrun', json=payload)
        if res.status_code == 200:
            result = res.json()
            prob = result['probability']
            is_overrun = result['predicted_overrun']
            
            # --- Results Display ---
            st.markdown("### 📊 Prediction Analysis")
            
            # Gauge Chart for Risk Probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Overrun Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if prob > 0.5 else "#22c55e"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},
                        {'range': [30, 70], 'color': "rgba(234, 179, 8, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_res2:
                if is_overrun:
                    st.error(f"🚨 **High Risk Detected**")
                    st.markdown(f"This project has a **{prob*100:.1f}%** probability of exceeding its budget/timeline.")
                    st.markdown("""
                    **Key Risk Drivers:**
                    - High Regulatory Risk
                    - Volatile Market Conditions
                    - Complex Terrain
                    """)
                else:
                    st.success(f"✅ **Low Risk**")
                    st.markdown(f"This project is likely to stay within limits (**{prob*100:.1f}%** risk).")
                    
        else:
            st.error("Error connecting to prediction service.")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.info("Make sure the FastAPI server is running: `python serve_model_fastapi.py`")

st.markdown("---")
st.markdown("### 📈 Historical Data Insights")

try:
    df = pd.read_csv('synthetic_projects.csv')
    
    tab1, tab2 = st.tabs(["Dataset Preview", "Risk Distribution"])
    
    with tab1:
        st.dataframe(df.sample(10), use_container_width=True)
        
    with tab2:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_pie = px.pie(df, names='regulatory_risk', title='Projects by Regulatory Risk', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart2:
            fig_bar = px.bar(df.groupby('project_type')['actual_cost'].mean().reset_index(), x='project_type', y='actual_cost', title='Avg Actual Cost by Type')
            st.plotly_chart(fig_bar, use_container_width=True)
            
except Exception as e:
    st.warning("Dataset not found. Please run data generation script.")
