"""Pick the decision threshold from what each mistake costs, not from 0.5.

Costs are an assumption, stated here and in the README: a retention offer (a discount) costs
1 unit, and losing a customer costs MISSED_COST units. Catching a churner still costs the
offer, so it's counted too. Thresholds are picked on out-of-fold predictions from the
training set; the test set is only used to report the result.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn.compare import CURVE_MODELS, confusion
from churn.data import load
from churn.plots import plot_cost_curves
from churn.train import REPORTS, SEED, fit_all, split

OFFER_COST = 1.0
MISSED_COST = 5.0
COST_RATIOS = [2, 3, 5, 10, 20]


@dataclass(frozen=True)
class Costs:
    offer: float = OFFER_COST
    missed: float = MISSED_COST

    def __post_init__(self):
        for name, value in (("offer", self.offer), ("missed", self.missed)):
            ok = isinstance(value, int | float) and not isinstance(value, bool)
            if not (ok and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} cost must be a positive finite number, got {value!r}")

    @property
    def bayes_threshold(self) -> float:
        """Best cut if the scores were true probabilities: offer when p * missed > offer."""
        return self.offer / self.missed


DEFAULT_COSTS = Costs()


def _check(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape or y_true.ndim != 1:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {scores.shape}")
    if len(y_true) == 0:
        raise ValueError("No rows to pick a threshold from")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinity")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("Labels must be 0 (stayed) and 1 (churned)")
    return y_true, scores


def total_cost(y_true, scores, threshold: float, costs: Costs = DEFAULT_COSTS) -> float:
    y_true, scores = _check(y_true, scores)
    flagged = scores >= threshold
    missed = int((~flagged & (y_true == 1)).sum())
    return float(flagged.sum() * costs.offer + missed * costs.missed)


def cost_curve(y_true, scores, costs: Costs = DEFAULT_COSTS) -> tuple[np.ndarray, np.ndarray]:
    """Average cost per customer at every threshold where the decision actually changes.

    Candidates are the distinct scores plus one above them all ("offer nobody"), so ties are
    never split: customers with the same score always get the same decision.
    """
    y_true, scores = _check(y_true, scores)
    thresholds = np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))
    order = np.argsort(scores)
    sorted_scores, sorted_y = scores[order], y_true[order]
    # For each threshold t: flagged = scores >= t, missed = churners with scores < t.
    below = np.searchsorted(sorted_scores, thresholds, side="left")
    churners_below = np.concatenate([[0], np.cumsum(sorted_y)])[below]
    flagged = len(scores) - below
    cost = flagged * costs.offer + churners_below * costs.missed
    return thresholds, cost / len(scores)


def pick_threshold(y_true, scores, costs: Costs = DEFAULT_COSTS) -> float:
    """Cheapest threshold; on a tie, the highest one (fewer offers for the same cost)."""
    thresholds, cost = cost_curve(y_true, scores, costs)
    best = np.flatnonzero(np.isclose(cost, cost.min()))
    return float(thresholds[best[-1]])


def out_of_fold_scores(model, X_train, y_train, folds: int = 5) -> np.ndarray:
    """Churn scores for every training customer from a model that never saw them."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def profile(X: pd.DataFrame) -> dict:
    """A few plain facts about a group of customers, for comparing who gets missed."""
    if len(X) == 0:
        return {"customers": 0}
    return {
        "customers": len(X),
        "month_to_month": round(float((X["Contract"] == "Month-to-month").mean()), 3),
        "fiber": round(float((X["InternetService"] == "Fiber optic").mean()), 3),
        "electronic_check": round(float((X["PaymentMethod"] == "Electronic check").mean()), 3),
        "median_tenure_months": float(X["tenure"].median()),
        "median_monthly_charge": round(float(X["MonthlyCharges"].median()), 2),
    }


def who_gets_missed(X_test: pd.DataFrame, y_test, flagged: dict[str, np.ndarray]) -> dict:
    """Do the models miss the same churners, and what do the missed ones look like?"""
    y_test = np.asarray(y_test)
    churned = y_test == 1
    missed = np.column_stack([~f & churned for f in flagged.values()])
    n_missing = missed.sum(axis=1)
    return {
        "churners": int(churned.sum()),
        "missed_by_all": int((n_missing == len(flagged)).sum()),
        "missed_by_some": int(((n_missing > 0) & (n_missing < len(flagged))).sum()),
        "caught_by_all": int((churned & (n_missing == 0)).sum()),
        "profile_missed_by_all": profile(X_test[n_missing == len(flagged)]),
        "profile_caught_by_all": profile(X_test[churned & (n_missing == 0)]),
    }


def run(costs: Costs = DEFAULT_COSTS) -> dict:
    df = load()
    fitted, _, X_test, y_test = fit_all(df)
    X_train, _, y_train, _ = split(df)

    chosen, test_scores, flagged, picked, curves = {}, {}, {}, {}, {}
    for name in CURVE_MODELS:
        oof = out_of_fold_scores(fitted[name], X_train, y_train)
        test_scores[name] = fitted[name].predict_proba(X_test)[:, 1]
        sweep = {}
        for ratio in COST_RATIOS:
            c = Costs(offer=1.0, missed=float(ratio))
            t = pick_threshold(y_train, oof, c)
            per_customer = total_cost(y_test, test_scores[name], t, c) / len(y_test)
            sweep[str(ratio)] = {
                "threshold": round(t, 3),
                "cost_per_customer": round(per_customer, 3),
            }
        t = pick_threshold(y_train, oof, costs)
        flagged[name], picked[name] = test_scores[name] >= t, t
        curves[name] = cost_curve(y_test, test_scores[name], costs)
        chosen[name] = {
            "threshold": round(t, 3),
            "at_threshold": confusion(y_test, test_scores[name], t),
            "at_0.5": confusion(y_test, test_scores[name], 0.5),
            "test_cost": total_cost(y_test, test_scores[name], t, costs),
            "test_cost_at_0.5": total_cost(y_test, test_scores[name], 0.5, costs),
            "by_cost_ratio": sweep,
        }

    report = {
        "costs": {"offer": costs.offer, "missed": costs.missed},
        "bayes_threshold": costs.bayes_threshold,
        "offer_everyone_cost": total_cost(y_test, np.ones(len(y_test)), 0.5, costs),
        "offer_nobody_cost": total_cost(y_test, np.zeros(len(y_test)), 0.5, costs),
        "models": chosen,
        "errors": who_gets_missed(X_test, y_test, flagged),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cost_threshold.json").write_text(json.dumps(report, indent=2))
    plot_cost_curves(curves, picked, REPORTS / "figures" / "cost_curve.png")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
