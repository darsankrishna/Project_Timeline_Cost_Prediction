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


def _vendor_index(vendor: str) -> int:
    return int(vendor.split('_')[1])


def _vendor_cohort_risk(vendor: str) -> str:
    idx = _vendor_index(vendor)
    if idx <= 6:
        return 'low'
    if idx <= 14:
        return 'medium'
    return 'high'


def _key_risk_factors(payload: ProjectIn, cost_prob: float, time_prob: float) -> list[str]:
    factors: list[tuple[int, str]] = []

    if payload.regulatory_risk == 'High':
        factors.append((100, 'High regulatory approval complexity'))
    elif payload.regulatory_risk == 'Medium':
        factors.append((75, 'Moderate regulatory clearance risk'))

    if payload.market_condition == 'Volatile':
        factors.append((85, 'Volatile market may increase material and execution risk'))
    if payload.season == 'Monsoon':
        factors.append((80, 'Monsoon season can delay on-site execution'))
    elif payload.season == 'Winter':
        factors.append((40, 'Winter season may slow field productivity'))

    if payload.terrain in {'hilly', 'forest'}:
        factors.append((62, f'Complex terrain ({payload.terrain}) increases execution difficulty'))
    elif payload.terrain == 'urban':
        factors.append((45, 'Urban terrain may involve right-of-way and utility conflicts'))

    if payload.vendor_rating < 3.0:
        factors.append((92, 'Low vendor rating indicates high delivery risk'))
    elif payload.vendor_rating < 3.8:
        factors.append((60, 'Mid vendor rating indicates moderate delivery variance'))

    cohort = _vendor_cohort_risk(payload.vendor)
    if cohort == 'high':
        factors.append((65, f'{payload.vendor} belongs to higher-risk vendor cohort'))
    elif cohort == 'medium':
        factors.append((48, f'{payload.vendor} belongs to medium-risk vendor cohort'))

    if payload.planned_days > 300:
        factors.append((50, 'Long planned duration increases schedule slippage exposure'))

    if payload.project_type == 'substation' and payload.planned_cost > 60_000_000:
        factors.append((52, 'Large substation budget increases cost sensitivity'))
    elif payload.project_type == 'underground_cable' and payload.planned_cost > 25_000_000:
        factors.append((52, 'Large underground cable budget increases cost sensitivity'))
    elif payload.project_type == 'overhead_line' and payload.planned_cost > 15_000_000:
        factors.append((52, 'Large overhead line budget increases cost sensitivity'))

    if cost_prob > 0.7:
        factors.append((64, 'Model indicates high cost-overrun probability'))
    if time_prob > 0.7:
        factors.append((64, 'Model indicates high timeline-overrun probability'))

    if not factors:
        return ['Balanced profile: no major risk drivers triggered']

    factors.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in factors[:4]]


def _vendor_info(payload: ProjectIn) -> dict:
    cohort = _vendor_cohort_risk(payload.vendor)
    if payload.vendor_rating >= 4.3:
        rating_band = 'Strong'
        note = 'Strong historical rating; maintain current controls.'
    elif payload.vendor_rating >= 3.5:
        rating_band = 'Watchlist'
        note = 'Acceptable rating; monitor milestones closely.'
    else:
        rating_band = 'Critical'
        note = 'Low rating; enforce strict governance and contingency.'

    cohort_note = {
        'low': 'Vendor cohort is generally stable.',
        'medium': 'Vendor cohort has mixed reliability.',
        'high': 'Vendor cohort is historically volatile.',
    }[cohort]

    return {
        'vendor': payload.vendor,
        'vendor_rating': payload.vendor_rating,
        'vendor_rating_band': rating_band,
        'vendor_cohort_risk': cohort,
        'notes': [note, cohort_note],
    }


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
        'key_risk_factors': _key_risk_factors(payload, cost_prob, time_prob),
        'vendor_info': _vendor_info(payload),
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
        'key_risk_factors': result['key_risk_factors'],
        'vendor_info': result['vendor_info'],
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
