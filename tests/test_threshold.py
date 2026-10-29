import math

import numpy as np
import pytest

from churn.plots import plot_cost_curves
from churn.threshold import (
    Costs,
    cost_curve,
    out_of_fold_scores,
    pick_threshold,
    profile,
    total_cost,
    who_gets_missed,
)
from churn.train import models, split
from tests.test_metrics_train import labelled_rows

Y = np.array([1, 1, 0, 0, 0, 0])
SCORES = np.array([0.9, 0.3, 0.6, 0.2, 0.1, 0.05])


def brute_force(y, scores, costs):
    """Try every threshold the slow way, to check the vectorised curve against."""
    candidates = sorted(set(scores)) + [math.inf]
    return {t: total_cost(y, scores, t, costs) for t in candidates}


@pytest.mark.parametrize(
    "offer, missed",
    [(0, 5), (-1, 5), (1, 0), (1, float("nan")), (1, float("inf")), ("1", 5), (True, 5), (None, 5)],
)
def test_costs_must_be_positive_numbers(offer, missed):
    with pytest.raises(ValueError, match="positive finite"):
        Costs(offer=offer, missed=missed)


def test_bayes_threshold():
    assert Costs(offer=1, missed=5).bayes_threshold == pytest.approx(0.2)


def test_total_cost_counts_offers_and_misses():
    # At 0.5: offers to 0.9 and 0.6 (2 x 1), misses the 0.3 churner (1 x 5).
    assert total_cost(Y, SCORES, 0.5, Costs(1, 5)) == 7
    assert total_cost(Y, SCORES, 2.0, Costs(1, 5)) == 10  # offer nobody: 2 churners missed
    assert total_cost(Y, SCORES, 0.0, Costs(1, 5)) == 6  # offer everyone


def test_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = np.round(rng.random(200), 2)  # rounding forces ties
    costs = Costs(1, 4)
    thresholds, per_customer = cost_curve(y, scores, costs)
    slow = brute_force(y, scores, costs)
    for t, c in zip(thresholds[:-1], per_customer[:-1], strict=True):
        assert c * len(y) == pytest.approx(slow[t])
    assert per_customer[-1] * len(y) == pytest.approx(slow[math.inf])


def test_pick_is_the_cheapest():
    costs = Costs(1, 5)
    t = pick_threshold(Y, SCORES, costs)
    assert total_cost(Y, SCORES, t, costs) == min(brute_force(Y, SCORES, costs).values())


def test_ties_are_never_split():
    # Two customers share 0.5; any threshold flags both or neither.
    y = np.array([1, 0, 0])
    scores = np.array([0.5, 0.5, 0.1])
    thresholds, _ = cost_curve(y, scores)
    for t in thresholds:
        assert (scores[0] >= t) == (scores[1] >= t)
    assert len(thresholds) == 3  # 0.1, 0.5 and "offer nobody"


def test_cheap_offers_lower_the_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    scores = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    picked = [pick_threshold(y, scores, Costs(1, r)) for r in (2, 5, 20)]
    assert picked == sorted(picked, reverse=True)


def test_when_missing_is_cheap_offer_nobody():
    # An offer costs more than losing the customer, so never offer.
    t = pick_threshold(Y, SCORES, Costs(offer=10, missed=1))
    assert t > SCORES.max()
    assert total_cost(Y, SCORES, t, Costs(10, 1)) == 2


def test_when_every_customer_churns_offer_everyone():
    y = np.ones(4, dtype=int)
    t = pick_threshold(y, np.array([0.1, 0.2, 0.3, 0.4]), Costs(1, 5))
    assert t == pytest.approx(0.1)


def test_constant_scores():
    t = pick_threshold(Y, np.full(len(Y), 0.4), Costs(1, 5))
    assert t in (0.4, pytest.approx(np.nextafter(0.4, np.inf)))


@pytest.mark.parametrize(
    "y, scores, match",
    [
        ([1, 0], [0.5], "Shape mismatch"),
        ([], [], "No rows"),
        ([1, 0], [0.5, float("nan")], "NaN"),
        ([1, 2], [0.5, 0.4], "Labels"),
        ([[1, 0]], [[0.5, 0.4]], "Shape mismatch"),
    ],
)
def test_bad_input_is_rejected(y, scores, match):
    with pytest.raises(ValueError, match=match):
        cost_curve(np.array(y), np.array(scores, dtype=float))


def test_out_of_fold_scores_never_see_their_own_row():
    df = labelled_rows(600)
    X_train, _, y_train, _ = split(df)
    model = models()["logistic_regression"]
    oof = out_of_fold_scores(model, X_train, y_train)
    assert oof.shape == (len(X_train),)
    assert ((oof >= 0) & (oof <= 1)).all()
    # Fitting on everything and scoring the same rows gives different (optimistic) scores.
    in_sample = model.fit(X_train, y_train).predict_proba(X_train)[:, 1]
    assert not np.allclose(oof, in_sample)


def test_who_gets_missed_counts_add_up():
    df = labelled_rows(300)
    X, y = df.drop(columns="Churn"), df["Churn"].to_numpy()
    rng = np.random.default_rng(2)
    flagged = {"a": rng.random(len(y)) > 0.5, "b": rng.random(len(y)) > 0.5}
    out = who_gets_missed(X, y, flagged)
    assert out["missed_by_all"] + out["missed_by_some"] + out["caught_by_all"] == y.sum()
    assert out["profile_missed_by_all"]["customers"] == out["missed_by_all"]


def test_profile_of_nobody():
    df = labelled_rows(10)
    assert profile(df.iloc[0:0]) == {"customers": 0}


def test_cost_plot_is_written(tmp_path):
    curves = {"m": cost_curve(Y, SCORES)}
    plot_cost_curves(curves, {"m": 0.3}, tmp_path / "c.png")
    assert (tmp_path / "c.png").read_bytes()[:4] == b"\x89PNG"


def test_cost_plot_handles_a_score_of_exactly_one(tmp_path):
    # A pure tree leaf can score 1.0; "offer nobody" then sits just above 1.
    curves = {"tree": cost_curve(Y, np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0]))}
    plot_cost_curves(curves, {"tree": 1.0}, tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
