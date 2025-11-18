# 1. Install
pip install -r requirements.txt

# 2. Generate data
python generate_synthetic_data.py

# 3. Train model
python train_model.py

# 4. Start API
python serve_model_fastapi.py
# or: uvicorn serve_model_fastapi:app --reload --port 8000

# 5. Start dashboard (in new terminal)
streamlit run dashboard_streamlit.py
