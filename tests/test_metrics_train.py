import numpy as np
import pytest

from churn.metrics import evaluate
from churn.train import models, split
from tests.test_features import make_rows


def test_evaluate_counts():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    scores = [0.9, 0.6, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]
    m = evaluate(y, scores, threshold=0.5)
    assert m["missed_churners"] == 1
    assert m["wasted_offers"] == 1
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_evaluate_threshold_moves_the_tradeoff():
    y = [1, 1, 0, 0]
    scores = [0.8, 0.4, 0.3, 0.1]
    assert evaluate(y, scores, 0.5)["recall"] == 0.5
    assert evaluate(y, scores, 0.35)["recall"] == 1.0


def test_evaluate_rejects_single_class():
    with pytest.raises(ValueError, match="both classes"):
        evaluate([0, 0, 0], [0.1, 0.2, 0.3])


def test_evaluate_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate([0, 1], [0.1, 0.2, 0.3])


def labelled_rows(n=400):
    df = make_rows(n, seed=3)
    # Churn mostly on month-to-month, so there's a learnable signal.
    rng = np.random.default_rng(3)
    p = np.where(df["Contract"] == "Month-to-month", 0.6, 0.05)
    return df.assign(Churn=(rng.random(n) < p).astype(int))


def test_split_is_stratified_and_disjoint():
    df = labelled_rows()
    X_train, X_test, y_train, y_test = split(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.03
    assert "Churn" not in X_train.columns


def test_logistic_regression_beats_baseline_on_learnable_data():
    df = labelled_rows(1000)
    X_train, X_test, y_train, y_test = split(df)
    auc = {}
    for name, model in models().items():
        model.fit(X_train, y_train)
        auc[name] = evaluate(y_test, model.predict_proba(X_test)[:, 1])["roc_auc"]
    assert auc["baseline_prior"] == pytest.approx(0.5)
    assert auc["logistic_regression"] > 0.75


def test_trained_model_scores_unseen_category():
    df = labelled_rows()
    X_train, X_test, y_train, _ = split(df)
    model = models()["logistic_regression"].fit(X_train, y_train)
    odd = X_test.head(1).assign(PaymentMethod="Crypto wallet")
    score = model.predict_proba(odd)[0, 1]
    assert 0 <= score <= 1
