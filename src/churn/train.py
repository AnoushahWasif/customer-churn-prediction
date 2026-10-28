import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier

from churn.data import TARGET, load
from churn.features import build_preprocessor
from churn.metrics import evaluate
from churn.plots import plot_cv_depth, plot_tree_top

SEED = 42
TEST_SIZE = 0.2
REPORTS = Path("reports")
TREE_GRID = {
    "decisiontreeclassifier__max_depth": [2, 3, 4, 5, 6, 8, 10, 12, None],
    "decisiontreeclassifier__min_samples_leaf": [1, 20, 50],
}


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


def tune_tree(X_train, y_train, grid: dict = TREE_GRID, folds: int = 5) -> GridSearchCV:
    """Pick tree depth and leaf size by cross-validated ROC-AUC on the training set only."""
    tree = make_pipeline(
        build_preprocessor(scale=False),
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    search = GridSearchCV(tree, grid, scoring="roc_auc", cv=cv, n_jobs=-1, return_train_score=True)
    return search.fit(X_train, y_train)


def run() -> pd.DataFrame:
    X_train, X_test, y_train, y_test = split(load())
    fitted = {}
    for name, model in models().items():
        fitted[name] = model.fit(X_train, y_train)

    search = tune_tree(X_train, y_train)
    fitted["decision_tree"] = search.best_estimator_

    rows = []
    for name, model in fitted.items():
        scores = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, **evaluate(y_test, scores)})

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_results.json").write_text(json.dumps(rows, indent=2))
    best = {k.split("__")[1]: v for k, v in search.best_params_.items()}
    (REPORTS / "tree_search.json").write_text(
        json.dumps({"best_params": best, "best_cv_roc_auc": search.best_score_}, indent=2)
    )
    figures = REPORTS / "figures"
    plot_cv_depth(search.cv_results_, figures / "tree_cv_depth.png")
    plot_tree_top(search.best_estimator_, figures / "tree_top.png")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run().round(3).to_string(index=False))
