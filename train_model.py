import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from model_utils import build_pipeline, save_artifact


def train_and_evaluate_target(df: pd.DataFrame, target_col: str, model_path: str):
    X = df.drop(
        [
            'project_id',
            'actual_cost',
            'actual_days',
            'cost_overrun_pct',
            'time_overrun_pct',
            'cost_overrun',
            'time_overrun',
        ],
        axis=1,
    )
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(random_state=42)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    print(f'===== {target_col} =====')
    print(classification_report(y_test, pred))
    print('AUC:', roc_auc_score(y_test, prob))
    print()

    save_artifact(pipe, model_path)
    print(f'Saved model to {model_path}')


def main():
    df = pd.read_csv('synthetic_projects.csv')
    train_and_evaluate_target(df, 'cost_overrun', 'artifacts/model_cost.pkl')
    train_and_evaluate_target(df, 'time_overrun', 'artifacts/model_time.pkl')


if __name__ == '__main__':
    main()
