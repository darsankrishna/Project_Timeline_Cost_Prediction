import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N = 2000

project_types = ['substation','overhead_line','underground_cable']
terrains = ['plains','hilly','forest','urban']
vendors = [f'vendor_{i}' for i in range(1,21)]

rows = []
for i in range(N):

    pid = f'P{i+1:05d}'
    ptype = np.random.choice(project_types, p=[0.4,0.45,0.15])
    terrain = np.random.choice(terrains, p=[0.5,0.2,0.15,0.15])
    planned_days = int(np.random.normal(180 if ptype=='substation' else 120, 30))
    base_cost = 50e6 if ptype=='substation' else (20e6 if ptype=='underground_cable' else 10e6)
    material_cost_index = 1 + np.random.normal(0,0.08)
    labour_cost_index = 1 + np.random.normal(0,0.06)
    vendor = np.random.choice(vendors)
    vendor_perf = np.clip(np.random.normal(0.9,0.1), 0.4, 1.1)
    weather_risk = np.random.choice([0,1], p=[0.75,0.25])
    regulatory_delay = max(0, int(np.random.poisson(5) - (0 if terrain=='urban' else 1)))
    material_avail = np.random.choice(['good','ok','poor'], p=[0.7,0.2,0.1])
    demand_supply_shock = np.random.choice([0,1], p=[0.9,0.1])

# simulate overruns
    cost_multiplier = 1 + 0.15*(material_avail=='poor') + 0.12*demand_supply_shock + 0.2*(vendor_perf<0.75) + 0.05*regulatory_delay + np.random.normal(0,0.03)
    time_multiplier = 1 + 0.12*(material_avail=='poor') + 0.2*(vendor_perf<0.75) + 0.07*regulatory_delay + 0.08*weather_risk + np.random.normal(0,0.03)

    actual_cost = base_cost * material_cost_index * labour_cost_index * cost_multiplier
    actual_days = int(planned_days * time_multiplier)

    rows.append({
        'project_id': pid,
        'project_type': ptype,
        'terrain': terrain,
        'planned_days': planned_days,
        'planned_cost': base_cost,
        'material_cost_index': material_cost_index,
        'labour_cost_index': labour_cost_index,
        'vendor': vendor,
        'vendor_perf': vendor_perf,
        'weather_risk': weather_risk,
        'regulatory_delay_days': regulatory_delay,
        'material_avail': material_avail,
        'demand_supply_shock': demand_supply_shock,
        'actual_cost': actual_cost,
        'actual_days': actual_days
    })


df = pd.DataFrame(rows)
# create binary targets for overrun

df['cost_overrun_pct'] = (df['actual_cost'] - df['planned_cost']) / df['planned_cost']
df['time_overrun_pct'] = (df['actual_days'] - df['planned_days']) / df['planned_days']
df['cost_overrun'] = (df['cost_overrun_pct'] > 0.10).astype(int) # >10% over
# time overrun threshold 10%
df['time_overrun'] = (df['time_overrun_pct'] > 0.10).astype(int)

# write

df.to_csv('synthetic_projects.csv', index=False)
print('Wrote synthetic_projects.csv', df.shape)
