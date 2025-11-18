import shap
import pandas as pd
from model_utils import load_model

model = load_model('artifacts/model.pkl')

# load data

df = pd.read_csv('synthetic_projects.csv')
X = df.drop(['project_id','actual_cost','actual_days','cost_overrun_pct','time_overrun_pct','cost_overrun','time_overrun'], axis=1)

# prepare a small sample
X_sample = X.sample(200, random_state=42)

# shap explain: use TreeExplainer on xgboost
explainer = shap.Explainer(model.named_steps['clf'])
# Need transformed features
X_trans = model.named_steps['pre'].transform(X_sample)
shap_values = explainer(X_trans)

# global importance
shap.summary_plot(shap_values, X_trans, show=False)
print('Generated SHAP summary (display locally)')
