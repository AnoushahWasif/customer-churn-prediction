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
