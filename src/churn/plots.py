from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_cv_depth(cv_results: dict, path: Path) -> None:
    """Validation ROC-AUC against tree depth, one line per minimum leaf size."""
    depths = np.array(cv_results["param_decisiontreeclassifier__max_depth"], dtype=object)
    leaves = np.array(cv_results["param_decisiontreeclassifier__min_samples_leaf"])
    labels = sorted({d for d in depths if d is not None})
    if any(d is None for d in depths):
        labels.append(None)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for color, leaf in zip(SERIES, sorted(set(leaves)), strict=False):
        means = [cv_results["mean_test_score"][(leaves == leaf) & (depths == d)][0] for d in labels]
        ax.plot(x, means, color=color, lw=2, marker="o", ms=4, label=f"min leaf {leaf}")
        ax.annotate(
            f"min leaf {leaf}",
            (x[-1], means[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xticks(x, [str(d) if d is not None else "no limit" for d in labels])
    ax.set_xlim(-0.3, len(labels) + 0.6)
    ax.set_xlabel("Maximum tree depth", color=MUTED)
    ax.set_ylabel("ROC-AUC, 5-fold CV on training set", color=MUTED)
    ax.set_title("Deeper trees overfit unless leaves are kept large", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tree_top(model: Pipeline, path: Path, depth: int = 2) -> None:
    """The first `depth` levels of the fitted tree, with readable feature names."""
    names = model[0][-1].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    plot_tree(
        model[-1],
        max_depth=depth,
        feature_names=list(names),
        class_names=["stays", "churns"],
        impurity=False,
        proportion=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


RAISES, LOWERS = "#e34948", "#2a78d6"


def plot_roc_pr(y_true, scores_by_model: dict, path: Path) -> None:
    """ROC and precision-recall curves side by side, one line per model."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    if len(scores_by_model) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} models per chart")
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    base_rate = float(np.mean(y_true))
    for color, (name, scores) in zip(SERIES, scores_by_model.items(), strict=False):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_ax.plot(fpr, tpr, color=color, lw=2, label=name)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_ax.plot(recall, precision, color=color, lw=2, label=name)

    roc_ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    roc_ax.text(0.55, 0.45, "random guess", color=MUTED, fontsize=8, rotation=33)
    pr_ax.axhline(base_rate, color=MUTED, lw=1, ls="--")
    pr_ax.text(0.02, base_rate + 0.02, f"churn rate {base_rate:.0%}", color=MUTED, fontsize=8)
    roc_ax.set(xlabel="Customers who stay, flagged anyway", ylabel="Churners caught")
    pr_ax.set(xlabel="Churners caught (recall)", ylabel="Flagged customers who churn (precision)")
    roc_ax.set_title("ROC", color=INK, loc="left")
    pr_ax.set_title("Precision-recall", color=INK, loc="left")
    for ax in (roc_ax, pr_ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        _style(ax)
    pr_ax.legend(frameon=False, fontsize=8, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drivers(coefs, importances, path: Path) -> None:
    """What each model leans on: signed LR weights next to tree importances."""
    fig, (lr_ax, tree_ax) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    c = coefs.iloc[::-1]
    lr_ax.barh(c.index, c.values, color=[RAISES if v > 0 else LOWERS for v in c.values], height=0.6)
    lr_ax.axvline(0, color=MUTED, lw=1)
    lr_ax.set_title("Logistic regression: weight per feature", color=INK, loc="left", fontsize=10)
    lr_ax.set_xlabel("← lowers churn      raises churn →", color=MUTED)

    i = importances.iloc[::-1]
    tree_ax.barh(i.index, i.values, color=SERIES[2], height=0.6)
    tree_ax.set_title("Decision tree: share of splits' gain", color=INK, loc="left", fontsize=10)
    tree_ax.set_xlabel("Importance (sums to 1 over all features)", color=MUTED)
    for ax in (lr_ax, tree_ax):
        ax.tick_params(labelsize=8)
        _style(ax)
        ax.grid(axis="y", visible=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cost_curves(curves: dict, picked: dict[str, float], path: Path) -> None:
    """Cost per customer against threshold; dots where each model's threshold was picked."""
    if len(curves) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} models per chart")
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    for color, (name, (thresholds, cost)) in zip(SERIES, curves.items(), strict=False):
        # Cost only changes at a customer's score, so draw steps (the tree has few distinct
        # scores, and a sloped line between them would show costs no threshold gives).
        # Past the highest score nobody gets an offer, which is the last cost.
        keep = thresholds <= 1.0
        x, y = np.append(thresholds[keep], 1.0), np.append(cost[keep], cost[-1])
        ax.step(x, y, where="pre", color=color, lw=2, label=name)
        if name in picked:
            i = min(np.searchsorted(thresholds, picked[name]), len(cost) - 1)
            ax.plot(picked[name], cost[i], "o", color=color, ms=7, mec="white", mew=1.5)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.505, ax.get_ylim()[1] * 0.97, "default 0.5", color=MUTED, fontsize=8, va="top")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Threshold (offer a discount when score is at or above it)", color=MUTED)
    ax.set_ylabel("Cost per test customer", color=MUTED)
    ax.set_title(
        "Cost of offers and missed churners on the test set (dots: picked on training folds)",
        color=INK,
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
