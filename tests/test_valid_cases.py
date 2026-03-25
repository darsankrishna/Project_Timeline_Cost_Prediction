"""Validate all supported categorical combinations against the prediction models."""

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from serve_model_fastapi import ProjectIn, _load_models, _predict

PROJECT_TYPES = ['substation', 'overhead_line', 'underground_cable']
TERRAINS = ['plains', 'hilly', 'forest', 'urban']
REGULATORY = ['Low', 'Medium', 'High']
SEASONS = ['Summer', 'Winter', 'Monsoon']
MARKET = ['Stable', 'Volatile']
VENDORS = [f'vendor_{i}' for i in range(1, 21)]


def run_all_valid_cases() -> int:
    rows = []
    for project_type, terrain, reg_risk, season, market, vendor in itertools.product(
        PROJECT_TYPES, TERRAINS, REGULATORY, SEASONS, MARKET, VENDORS
    ):
        rows.append(
            {
                'project_type': project_type,
                'terrain': terrain,
                'planned_days': 180,
                'planned_cost': 50_000_000.0,
                'regulatory_risk': reg_risk,
                'season': season,
                'vendor': vendor,
                'vendor_rating': 4.0,
                'market_condition': market,
            }
        )

    df = pd.DataFrame(rows)
    model_cost, model_time = _load_models()

    cost_probs = model_cost.predict_proba(df)[:, 1]
    time_probs = model_time.predict_proba(df)[:, 1]

    assert len(cost_probs) == 4320
    assert len(time_probs) == 4320
    assert ((cost_probs >= 0.0) & (cost_probs <= 1.0)).all()
    assert ((time_probs >= 0.0) & (time_probs <= 1.0)).all()

    return len(df)


def test_predict_contains_non_blank_risk_factors_and_vendor_info():
    payload = ProjectIn(
        project_type='substation',
        terrain='urban',
        planned_days=220,
        planned_cost=55_000_000.0,
        regulatory_risk='Medium',
        season='Winter',
        vendor='vendor_12',
        vendor_rating=3.6,
        market_condition='Volatile',
    )
    result = _predict(payload)

    assert 'key_risk_factors' in result
    assert len(result['key_risk_factors']) > 0
    assert all(str(x).strip() for x in result['key_risk_factors'])
    assert 'vendor_info' in result
    assert result['vendor_info']['vendor'] == 'vendor_12'
    assert len(result['vendor_info']['notes']) > 0


if __name__ == '__main__':
    total = run_all_valid_cases()
    print(f'Validated {total} valid prediction cases successfully.')
