import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from churn.data import TARGET, load
from churn.features import build_preprocessor
from churn.metrics import evaluate

SEED = 42
TEST_SIZE = 0.2
REPORTS = Path("reports")


def split(df: pd.DataFrame):
    X, y = df.drop(columns=TARGET), df[TARGET]
    # Stratify so train and test both keep the 26.5% churn rate.
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def models() -> dict[str, Pipeline]:
    return {
        "baseline_prior": make_pipeline(build_preprocessor(), DummyClassifier(strategy="prior")),
        "logistic_regression": make_pipeline(
            build_preprocessor(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "logistic_regression_balanced": make_pipeline(
            build_preprocessor(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        ),
    }


def run() -> pd.DataFrame:
    X_train, X_test, y_train, y_test = split(load())
    rows = []
    for name, model in models().items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, **evaluate(y_test, scores)})

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "logistic_baselines.json").write_text(json.dumps(rows, indent=2))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run().round(3).to_string(index=False))
