import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

CATEGORICAL = [
    'project_type',
    'terrain',
    'vendor',
    'regulatory_risk',
    'season',
    'market_condition',
]
NUMERICAL = ['planned_days', 'planned_cost', 'vendor_rating']


def build_pipeline(random_state: int = 42) -> Pipeline:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preproc = ColumnTransformer([
        ('cat', ohe, CATEGORICAL),
    ], remainder='passthrough')

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=random_state,
    )
    return Pipeline([
        ('pre', preproc),
        ('clf', clf),
    ])


def save_artifact(obj, fname: str):
    joblib.dump(obj, fname)


def load_artifact(fname: str):
    return joblib.load(fname)
