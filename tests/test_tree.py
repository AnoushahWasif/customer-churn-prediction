import numpy as np
import pytest

from churn.features import SCALED, build_preprocessor
from churn.metrics import evaluate
from churn.plots import plot_cv_depth, plot_tree_top
from churn.train import split, tune_tree
from tests.test_features import make_rows
from tests.test_metrics_train import labelled_rows

SMALL_GRID = {
    "decisiontreeclassifier__max_depth": [2, 4, None],
    "decisiontreeclassifier__min_samples_leaf": [1, 20],
}


@pytest.fixture(scope="module")
def searched():
    X_train, X_test, y_train, y_test = split(labelled_rows(1000))
    return tune_tree(X_train, y_train, grid=SMALL_GRID, folds=3), X_train, X_test, y_test


def test_unscaled_preprocessor_keeps_original_units():
    rows = make_rows(50)
    out = build_preprocessor(scale=False).fit_transform(rows)
    np.testing.assert_allclose(out[:, SCALED.index("tenure")], rows["tenure"])


def test_cv_only_uses_training_rows(searched):
    search, X_train, _, _ = searched
    # Every training row lands in exactly one validation fold; nothing else is ever scored.
    folds = list(search.cv.split(X_train, np.zeros(len(X_train))))
    assert sum(len(val) for _, val in folds) == len(X_train)
    assert search.n_splits_ == 3


def test_search_covers_the_whole_grid(searched):
    search, *_ = searched
    assert len(search.cv_results_["params"]) == 3 * 2


def test_tuned_tree_beats_chance_on_held_out_rows(searched):
    search, _, X_test, y_test = searched
    scores = search.best_estimator_.predict_proba(X_test)[:, 1]
    assert evaluate(y_test, scores)["roc_auc"] > 0.75


def test_tuning_is_reproducible():
    X_train, _, y_train, _ = split(labelled_rows(600))
    a = tune_tree(X_train, y_train, grid=SMALL_GRID, folds=3)
    b = tune_tree(X_train, y_train, grid=SMALL_GRID, folds=3)
    assert a.best_params_ == b.best_params_
    np.testing.assert_allclose(a.cv_results_["mean_test_score"], b.cv_results_["mean_test_score"])


def test_tree_scores_unseen_category(searched):
    search, _, X_test, _ = searched
    odd = X_test.head(1).assign(InternetService="Satellite")
    assert 0 <= search.best_estimator_.predict_proba(odd)[0, 1] <= 1


def test_plots_write_pngs(searched, tmp_path):
    search, *_ = searched
    plot_cv_depth(search.cv_results_, tmp_path / "cv.png")
    plot_tree_top(search.best_estimator_, tmp_path / "tree.png")
    for name in ("cv.png", "tree.png"):
        assert (tmp_path / name).read_bytes()[:4] == b"\x89PNG"


def test_cv_plot_handles_grid_without_unlimited_depth(tmp_path):
    X_train, _, y_train, _ = split(labelled_rows(400))
    grid = {
        "decisiontreeclassifier__max_depth": [2, 3],
        "decisiontreeclassifier__min_samples_leaf": [1],
    }
    search = tune_tree(X_train, y_train, grid=grid, folds=3)
    plot_cv_depth(search.cv_results_, tmp_path / "cv.png")
    assert (tmp_path / "cv.png").exists()
