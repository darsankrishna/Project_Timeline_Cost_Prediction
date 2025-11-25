import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

project_types = ['substation', 'overhead_line', 'underground_cable']
terrains = ['plains', 'hilly', 'forest', 'urban']
regulatory_risks = ['Low', 'Medium', 'High']
seasons = ['Summer', 'Winter', 'Monsoon']
market_conditions = ['Stable', 'Volatile']
vendors = [f'vendor_{i}' for i in range(1, 21)]

rows = []
for i in range(N):
    pid = f'P{i+1:05d}'
    ptype = np.random.choice(project_types, p=[0.4, 0.45, 0.15])
    terrain = np.random.choice(terrains, p=[0.5, 0.2, 0.15, 0.15])
    
    # Base parameters
    if ptype == 'substation':
        base_days = 180
        base_cost = 50e6
    elif ptype == 'underground_cable':
        base_days = 150
        base_cost = 20e6
    else:
        base_days = 120
        base_cost = 10e6
        
    planned_days = int(np.random.normal(base_days, 20))
    planned_cost = base_cost
    
    # Risk Factors (Features known BEFORE project starts)
    regulatory_risk = np.random.choice(regulatory_risks, p=[0.6, 0.3, 0.1])
    season = np.random.choice(seasons, p=[0.4, 0.4, 0.2])
    vendor = np.random.choice(vendors)
    vendor_rating = np.round(np.random.uniform(2.5, 5.0), 1) # Historical rating
    market_condition = np.random.choice(market_conditions, p=[0.7, 0.3])
    
    # Generate Actual Outcomes based on Risks (Hidden logic)
    
    # 1. Regulatory Delay
    if regulatory_risk == 'High':
        actual_delay = int(np.random.exponential(20)) # Avg 20 days delay
    elif regulatory_risk == 'Medium':
        actual_delay = int(np.random.exponential(10))
    else:
        actual_delay = int(np.random.exponential(2))
        
    # 2. Weather Impact (Monsoon is bad)
    weather_factor = 1.0
    if season == 'Monsoon':
        if np.random.random() < 0.6: # 60% chance of rain delay
            weather_factor = 1.15
    elif season == 'Winter':
        weather_factor = 1.02 # Slight delay due to fog/cold?
        
    # 3. Vendor Performance (Rating affects outcome)
    # Higher rating -> better performance (factor < 1 means faster/cheaper)
    vendor_perf_factor = 1.0 + (3.5 - vendor_rating) * 0.05 # 5.0 -> 0.925 (good), 2.5 -> 1.05 (bad)
    vendor_perf_factor += np.random.normal(0, 0.05) # Random variance
    
    # 4. Market Impact
    cost_inflation = 1.0
    if market_condition == 'Volatile':
        cost_inflation = 1.0 + np.abs(np.random.normal(0.1, 0.05)) # 10% inflation avg
    else:
        cost_inflation = 1.0 + np.abs(np.random.normal(0.02, 0.01)) # 2% normal inflation
        
    # Calculate Actuals
    # Time: Planned * Weather * Vendor + Delay
    actual_days = int(planned_days * weather_factor * vendor_perf_factor + actual_delay)
    
    # Cost: Planned * Inflation * Vendor + (Delay cost ~ 0.5% per day)
    delay_cost_penalty = actual_delay * (0.005 * planned_cost)
    actual_cost = (planned_cost * cost_inflation * vendor_perf_factor) + delay_cost_penalty
    
    rows.append({
        'project_id': pid,
        'project_type': ptype,
        'terrain': terrain,
        'planned_days': planned_days,
        'planned_cost': planned_cost,
        'regulatory_risk': regulatory_risk,
        'season': season,
        'vendor': vendor,
        'vendor_rating': vendor_rating,
        'market_condition': market_condition,
        'actual_days': actual_days,
        'actual_cost': actual_cost
    })

df = pd.DataFrame(rows)

# Create Targets
df['cost_overrun_pct'] = (df['actual_cost'] - df['planned_cost']) / df['planned_cost']
df['time_overrun_pct'] = (df['actual_days'] - df['planned_days']) / df['planned_days']

df['cost_overrun'] = (df['cost_overrun_pct'] > 0.10).astype(int)
df['time_overrun'] = (df['time_overrun_pct'] > 0.10).astype(int)

df.to_csv('synthetic_projects.csv', index=False)
print('Wrote synthetic_projects.csv', df.shape)
