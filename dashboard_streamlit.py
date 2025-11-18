import streamlit as st
import pandas as pd
import requests

st.title('POWERGRID — Cost & Timeline Risk Dashboard (MVP)')

st.sidebar.header('Predict a Project')
project_type = st.sidebar.selectbox('Project Type', ['substation','overhead_line','underground_cable'])
terrain = st.sidebar.selectbox('Terrain', ['plains','hilly','forest','urban'])
planned_days = st.sidebar.number_input('Planned Days', min_value=10, max_value=1000, value=120)
planned_cost = st.sidebar.number_input('Planned Cost (INR)', value=10000000.0)
material_cost_index = st.sidebar.slider('Material Cost Index', 0.5, 1.8, 1.0)
labour_cost_index = st.sidebar.slider('Labour Cost Index', 0.5, 1.8, 1.0)
vendor = st.sidebar.text_input('Vendor', 'vendor_1')
vendor_perf = st.sidebar.slider('Vendor Performance (0-1.2)', 0.4, 1.1, 0.9)
weather_risk = st.sidebar.selectbox('Weather Risk (0/1)', [0,1])
regulatory_delay_days = st.sidebar.number_input('Regulatory delay (days)', min_value=0, max_value=120, value=5)
material_avail = st.sidebar.selectbox('Material availability', ['good','ok','poor'])
demand_supply_shock = st.sidebar.selectbox('Demand-supply shock', [0,1])

if st.sidebar.button('Predict'):
payload = {
'project_type': project_type,
'terrain': terrain,
'planned_days': int(planned_days),
'planned_cost': float(planned_cost),
'material_cost_index': float(material_cost_index),
'labour_cost_index': float(labour_cost_index),
'vendor': vendor,
'vendor_perf': float(vendor_perf),
'weather_risk': int(weather_risk),
'regulatory_delay_days': int(regulatory_delay_days),
'material_avail': material_avail,
'demand_supply_shock': int(demand_supply_shock)
}
res = requests.post('http://localhost:8000/predict_cost_overrun', json=payload)
st.json(res.json())

st.markdown('---')
st.write('Sample dataset preview:')
try:
df = pd.read_csv('synthetic_projects.csv')
st.dataframe(df.sample(5))
except Exception as e:
st.write('No dataset found. Run `generate_synthetic_data.py` first.')
