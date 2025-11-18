import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from model_utils import build_pipeline, save_artifacts


df = pd.read_csv('synthetic_projects.csv')

# target: cost_overrun
X = df.drop(['project_id','actual_cost','actual_days','cost_overrun_pct','time_overrun_pct','cost_overrun','time_overrun'], axis=1)
y = df['cost_overrun']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, stratify=y)

pipe = build_pipeline()
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:,1]

print(classification_report(y_test, pred))
print('AUC:', roc_auc_score(y_test, prob))

save_artifacts(pipe)
print('Saved model to artifacts/model.pkl')
