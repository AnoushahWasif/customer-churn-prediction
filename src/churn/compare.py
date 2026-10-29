import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.plots import plot_drivers, plot_roc_pr
from churn.train import REPORTS, fit_all

CURVE_MODELS = ["logistic_regression", "logistic_regression_balanced", "decision_tree"]


def feature_names(model: Pipeline) -> list[str]:
    # model[0] is the preprocessing pipeline; its last step (the ColumnTransformer) knows the names.
    return list(model[0][-1].get_feature_names_out())


def coefficients(model: Pipeline) -> pd.Series:
    """Logistic regression weights per feature, largest effect first. Positive = more churn."""
    coefs = pd.Series(model[-1].coef_[0], index=feature_names(model))
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def importances(model: Pipeline) -> pd.Series:
    """Share of the tree's impurity reduction per feature. Always >= 0, sums to 1."""
    imp = pd.Series(model[-1].feature_importances_, index=feature_names(model))
    return imp.sort_values(ascending=False)


def confusion(y_true, scores, threshold: float = 0.5) -> dict[str, int]:
    y_true = np.asarray(y_true)
    pred = np.asarray(scores) >= threshold
    return {
        "stayed_kept": int((~pred & (y_true == 0)).sum()),
        "wasted_offers": int((pred & (y_true == 0)).sum()),
        "missed_churners": int((~pred & (y_true == 1)).sum()),
        "churners_caught": int((pred & (y_true == 1)).sum()),
    }


def run() -> dict:
    fitted, _, X_test, y_test = fit_all()
    scores = {name: fitted[name].predict_proba(X_test)[:, 1] for name in CURVE_MODELS}
    coefs = coefficients(fitted["logistic_regression"])
    imps = importances(fitted["decision_tree"])

    report = {
        "confusion_at_0.5": {name: confusion(y_test, s) for name, s in scores.items()},
        "top_coefficients": coefs.head(10).round(3).to_dict(),
        "top_importances": imps.head(10).round(3).to_dict(),
        "unused_by_tree": sorted(imps[imps == 0].index),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "comparison.json").write_text(json.dumps(report, indent=2))
    figures = REPORTS / "figures"
    plot_roc_pr(y_test, scores, figures / "roc_pr.png")
    plot_drivers(coefs.head(10), imps.head(10), figures / "drivers.png")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
