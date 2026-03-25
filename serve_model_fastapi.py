from functools import lru_cache
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from model_utils import load_artifact

app = FastAPI(title='POWERGRID Cost & Timeline Predictor')


@lru_cache(maxsize=2)
def _load_models():
    model_cost = load_artifact('artifacts/model_cost.pkl')
    model_time = load_artifact('artifacts/model_time.pkl')
    return model_cost, model_time


class ProjectIn(BaseModel):
    model_config = ConfigDict(extra='forbid')

    project_type: Literal['substation', 'overhead_line', 'underground_cable']
    terrain: Literal['plains', 'hilly', 'forest', 'urban']
    planned_days: int = Field(ge=10, le=2000)
    planned_cost: float = Field(gt=0)
    regulatory_risk: Literal['Low', 'Medium', 'High']
    season: Literal['Summer', 'Winter', 'Monsoon']
    vendor: str = Field(pattern=r'^vendor_([1-9]|1[0-9]|20)$')
    vendor_rating: float = Field(ge=1.0, le=5.0)
    market_condition: Literal['Stable', 'Volatile']


def _predict(payload: ProjectIn) -> dict:
    model_cost, model_time = _load_models()
    df = pd.DataFrame([payload.model_dump()])

    cost_prob = float(model_cost.predict_proba(df)[:, 1][0])
    time_prob = float(model_time.predict_proba(df)[:, 1][0])

    return {
        'cost_overrun_probability': cost_prob,
        'cost_overrun_predicted': int(cost_prob > 0.5),
        'time_overrun_probability': time_prob,
        'time_overrun_predicted': int(time_prob > 0.5),
    }


@app.get('/health')
def health():
    _load_models()
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: ProjectIn):
    return _predict(payload)


@app.post('/predict_cost_overrun')
def predict_cost_overrun(payload: ProjectIn):
    result = _predict(payload)
    return {
        'probability': result['cost_overrun_probability'],
        'predicted_overrun': result['cost_overrun_predicted'],
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
