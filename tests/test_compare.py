import numpy as np
import pandas as pd
import pytest

from churn.compare import coefficients, confusion, feature_names, importances
from churn.plots import plot_drivers, plot_roc_pr
from churn.train import fit_all
from tests.test_metrics_train import labelled_rows


@pytest.fixture(scope="module")
def fitted():
    models, _, X_test, y_test = fit_all(labelled_rows(1000))
    return models, X_test, y_test


def test_feature_names_line_up_with_coefficients(fitted):
    models, *_ = fitted
    lr = models["logistic_regression"]
    assert len(feature_names(lr)) == lr[-1].coef_.shape[1]
    assert "tenure" in feature_names(lr)


def test_coefficients_sorted_by_size_and_signed(fitted):
    models, *_ = fitted
    coefs = coefficients(models["logistic_regression"])
    assert list(coefs.abs()) == sorted(coefs.abs(), reverse=True)
    # The planted signal: month-to-month customers churn more.
    assert coefs["Contract_Month-to-month"] > 0


def test_importances_are_a_share(fitted):
    models, *_ = fitted
    imps = importances(models["decision_tree"])
    assert (imps >= 0).all()
    assert imps.sum() == pytest.approx(1.0)
    assert imps.index[0] == "Contract_Month-to-month"


def test_confusion_counts_add_up():
    y = [1, 1, 0, 0, 0]
    scores = [0.9, 0.2, 0.7, 0.1, 0.5]
    c = confusion(y, scores, threshold=0.5)
    assert c == {"stayed_kept": 1, "wasted_offers": 2, "missed_churners": 1, "churners_caught": 1}
    assert sum(c.values()) == len(y)


def test_confusion_with_threshold_above_every_score():
    c = confusion([0, 1], [0.2, 0.3], threshold=1.01)
    assert c["churners_caught"] == 0 and c["wasted_offers"] == 0


def test_comparison_plots_are_written(fitted, tmp_path):
    models, X_test, y_test = fitted
    scores = {
        name: m.predict_proba(X_test)[:, 1]
        for name, m in models.items()
        if name != "baseline_prior"
    }
    plot_roc_pr(y_test, scores, tmp_path / "roc.png")
    plot_drivers(
        coefficients(models["logistic_regression"]).head(5),
        importances(models["decision_tree"]).head(5),
        tmp_path / "drivers.png",
    )
    for name in ("roc.png", "drivers.png"):
        assert (tmp_path / name).read_bytes()[:4] == b"\x89PNG"


def test_roc_pr_rejects_too_many_models(tmp_path):
    y = np.array([0, 1, 0, 1])
    scores = {f"m{i}": np.array([0.1, 0.9, 0.2, 0.8]) for i in range(4)}
    with pytest.raises(ValueError, match="At most"):
        plot_roc_pr(y, scores, tmp_path / "x.png")


def test_plot_drivers_handles_all_zero_importances(tmp_path):
    # A depth-0 tree (one leaf) has no splits, so every importance is 0.
    coefs = pd.Series({"a": 0.5, "b": -0.2})
    imps = pd.Series({"a": 0.0, "b": 0.0})
    plot_drivers(coefs, imps, tmp_path / "d.png")
    assert (tmp_path / "d.png").exists()
