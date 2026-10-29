"""Command line: `churn train`, `churn score customers.csv`, `churn reports`."""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from churn.data import CATEGORICAL, FILENAME, NUMERIC, data_dir, load
from churn.model_store import ModelIntegrityError, load_model, save_model
from churn.threshold import Costs, out_of_fold_scores, pick_threshold, total_cost

SERVED_MODEL = "logistic_regression"
MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_ROWS = 200_000
# A cell starting with one of these is run as a formula by Excel and Sheets (CWE-1236).
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


class InputError(ValueError):
    pass


def read_customers(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Read and check a customer CSV. Returns (ids, features); raises InputError on bad input."""
    if not path.is_file():
        raise InputError(f"{path} not found")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise InputError(f"{path} is larger than {MAX_INPUT_BYTES // (1024 * 1024)} MB")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InputError(f"{path} is not a readable CSV ({type(exc).__name__})") from None
    if len(df) == 0:
        raise InputError(f"{path} has no customer rows")
    if len(df) > MAX_ROWS:
        raise InputError(f"{path} has more than {MAX_ROWS:,} rows; split it up")

    df.columns = df.columns.str.strip()
    missing = sorted(set(CATEGORICAL + NUMERIC) - set(df.columns))
    if missing:
        raise InputError(f"missing columns: {', '.join(missing)}")
    df = df.apply(lambda col: col.str.strip())

    # Same rule as training: a brand-new customer (tenure 0) hasn't been billed yet.
    new = (df["TotalCharges"] == "") & (df["tenure"] == "0")
    df.loc[new, "TotalCharges"] = "0"
    for col in NUMERIC:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.isna() | (values < 0) | values.isin([float("inf"), float("-inf")])
        if bad.any():
            lines = ", ".join(str(i + 2) for i in df.index[bad][:5])  # +2: header, 1-based
            raise InputError(f"{col} must be a number >= 0 (bad on line {lines})")
        df[col] = values

    ids = df["customerID"] if "customerID" in df.columns else pd.Series(df.index + 1).astype(str)
    return ids, df[CATEGORICAL + NUMERIC]


def safe_cell(value: str) -> str:
    return "'" + value if value.startswith(FORMULA_START) else value


def score(
    input_path: Path, model_dir: Path, expected_sha256: str | None = None
) -> tuple[pd.DataFrame, dict]:
    loaded = load_model(model_dir, expected_sha256)
    ids, X = read_customers(input_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = loaded.model.predict_proba(X)[:, 1]
    unseen = any("unknown categories" in str(w.message) for w in caught)

    result = pd.DataFrame(
        {
            "customer_id": ids.map(safe_cell).to_numpy(),
            "churn_score": scores.round(4),
            "offer": ["yes" if s >= loaded.threshold else "no" for s in scores],
        }
    )
    return result, {
        "customers": len(result),
        "offers": int((result["offer"] == "yes").sum()),
        "threshold": loaded.threshold,
        "unseen_categories": unseen,
    }


def train(data: Path | None, out_dir: Path, costs: Costs) -> dict:
    from churn.train import models, split

    df = load(data)
    X_train, X_test, y_train, y_test = split(df)
    model = models()[SERVED_MODEL]
    threshold = pick_threshold(y_train, out_of_fold_scores(model, X_train, y_train), costs)
    model.fit(X_train, y_train)
    test_scores = model.predict_proba(X_test)[:, 1]
    summary = {
        "model": SERVED_MODEL,
        "costs": {"offer": costs.offer, "missed": costs.missed},
        "test_customers": len(y_test),
        "test_cost": total_cost(y_test, test_scores, threshold, costs),
        "test_cost_at_0.5": total_cost(y_test, test_scores, 0.5, costs),
    }
    save_model(model, threshold, out_dir, summary)
    return {"threshold": threshold, **summary}


def reports() -> None:
    from churn import compare, threshold, train

    train.run()
    compare.run()
    threshold.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churn", description="Telco churn model")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fit the served model and pick its threshold from costs")
    t.add_argument("--data", type=Path, help=f"CSV to train on (default {data_dir() / FILENAME})")
    t.add_argument("--out", type=Path, default=Path("models"), help="model folder")
    t.add_argument("--offer-cost", type=float, default=Costs().offer)
    t.add_argument("--missed-cost", type=float, default=Costs().missed)

    s = sub.add_parser("score", help="score a CSV of customers")
    s.add_argument("input", type=Path)
    s.add_argument("--model", type=Path, default=Path("models"), help="model folder")
    s.add_argument("--out", type=Path, help="output CSV (default: print to stdout)")
    s.add_argument("--sha256", help="expected model hash, checked before loading")

    sub.add_parser("reports", help="rebuild every report and figure in reports/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            costs = Costs(offer=args.offer_cost, missed=args.missed_cost)
            if args.data is None and not (data_dir() / FILENAME).exists():
                raise InputError("no training data; run `make data` first or pass --data")
            print(json.dumps(train(args.data, args.out, costs), indent=2))
        elif args.command == "score":
            if args.out and args.out.resolve() == args.input.resolve():
                raise InputError("--out would overwrite the input file")
            # Score fully before opening the output, so bad input never leaves a half-written file.
            result, info = score(args.input, args.model, args.sha256)
            result.to_csv(args.out or sys.stdout, index=False, lineterminator="\n")
            note = "; some values were never seen in training" if info["unseen_categories"] else ""
            print(
                f"scored {info['customers']} customers, {info['offers']} get an offer "
                f"(threshold {info['threshold']:.3f}){note}",
                file=sys.stderr,
            )
        else:
            reports()
    except (InputError, ModelIntegrityError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
