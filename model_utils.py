import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

CATEGORICAL = ['project_type','terrain','vendor','material_avail']
NUMERICAL = ['planned_days','planned_cost','material_cost_index','labour_cost_index','vendor_perf','weather_risk','regulatory_delay_days','demand_supply_shock']


def build_pipeline():
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preproc = ColumnTransformer([
    ('cat', ohe, CATEGORICAL)
    ], remainder='passthrough')

    clf = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
    pipe = Pipeline([
        ('pre', preproc),
        ('clf', clf)
    ])
    return pipe


def save_artifacts(pipe, fname_model='artifacts/model.pkl'):
    joblib.dump(pipe, fname_model)


def load_model(fname_model='artifacts/model.pkl'):
    return joblib.load(fname_model)
