from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from model_utils import load_model

app = FastAPI(title='POWERGRID Cost & Timeline Predictor')
model = load_model('artifacts/model.pkl')

class ProjectIn(BaseModel):
      project_type: str
      terrain: str
      planned_days: int
      planned_cost: float
      regulatory_risk: str
      season: str
      vendor: str
      vendor_rating: float
      market_condition: str

@app.post('/predict_cost_overrun')
def predict_cost_overrun(payload: ProjectIn):
    df = pd.DataFrame([payload.dict()])
    prob = model.predict_proba(df)[:,1][0]
    pred = int(prob>0.5)
    return {'probability': float(prob), 'predicted_overrun': pred}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
