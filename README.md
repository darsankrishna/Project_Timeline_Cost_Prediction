# POWERGRID Cost & Timeline Predictor

## 1) Install
```bash
pip install -r requirements.txt
```

## 2) Generate data
```bash
python generate_synthetic_data.py
```

## 3) Train models (cost + timeline)
```bash
python train_model.py
```
This produces:
- `artifacts/model_cost.pkl`
- `artifacts/model_time.pkl`

## 4) Test all valid API prediction cases
```bash
python tests/test_valid_cases.py
```

## 5) Host the API locally
```bash
uvicorn serve_model_fastapi:app --host 0.0.0.0 --port 8000
```

## 6) Run dashboard (new terminal)
```bash
streamlit run dashboard_streamlit.py
```

## API Endpoints
- `GET /health`
- `POST /predict` (cost + timeline outputs + `key_risk_factors` + `vendor_info`)
- `POST /predict_cost_overrun` (backward-compatible cost-only output + `key_risk_factors` + `vendor_info`)
