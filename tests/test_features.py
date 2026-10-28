import numpy as np
import pandas as pd
import pytest

from churn.data import CATEGORICAL, NUMERIC
from churn.features import add_features, build_preprocessor


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = ["Yes", "No"]
    addon = ["Yes", "No", "No internet service"]
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n),
            "tenure": tenure,
            "PhoneService": rng.choice(yes_no, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(addon, n),
            "OnlineBackup": rng.choice(addon, n),
            "DeviceProtection": rng.choice(addon, n),
            "TechSupport": rng.choice(addon, n),
            "StreamingTV": rng.choice(addon, n),
            "StreamingMovies": rng.choice(addon, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(yes_no, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly).round(2),
        }
    )


@pytest.fixture
def rows():
    return make_rows(200)


def test_output_is_numeric_without_nans(rows):
    out = build_preprocessor().fit_transform(rows)
    assert out.dtype.kind == "f"
    assert not np.isnan(out).any()
    assert out.shape[0] == len(rows)


def test_no_internet_service_folded_into_no(rows):
    out = add_features(rows)
    for col in ["OnlineSecurity", "StreamingTV", "MultipleLines"]:
        assert set(out[col]) <= {"Yes", "No"}


def test_protective_addon_count():
    row = make_rows(1)
    row[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]] = [
        "Yes",
        "Yes",
        "No",
        "No internet service",
    ]
    assert add_features(row)["n_protective_addons"].iloc[0] == 2


def test_does_not_mutate_input(rows):
    before = rows.copy()
    add_features(rows)
    pd.testing.assert_frame_equal(rows, before)


def test_scaler_learns_from_training_rows_only():
    train, test = make_rows(300, seed=1), make_rows(300, seed=2)
    test["tenure"] = test["tenure"] + 1000  # wildly different test data
    pre = build_preprocessor().fit(train)
    scaler = pre[-1].named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(train["tenure"].mean())


def test_unseen_category_does_not_crash(rows):
    pre = build_preprocessor().fit(rows)
    new = make_rows(1)
    new["PaymentMethod"] = "Crypto wallet"
    out = pre.transform(new)
    names = list(pre[-1].get_feature_names_out())
    payment_cols = [i for i, n in enumerate(names) if n.startswith("PaymentMethod_")]
    assert out[0, payment_cols].sum() == 0


def test_missing_column_gives_clear_error(rows):
    with pytest.raises(ValueError, match="Missing columns: \\['Contract'\\]"):
        build_preprocessor().fit(rows.drop(columns="Contract"))


def test_negative_values_rejected(rows):
    rows.loc[0, "MonthlyCharges"] = -5
    with pytest.raises(ValueError, match="MonthlyCharges"):
        build_preprocessor().fit(rows)


def test_extra_columns_are_ignored(rows):
    pre = build_preprocessor().fit(rows)
    with_extra = rows.assign(customerID="x", Churn=1)
    assert pre.transform(with_extra).shape == pre.transform(rows).shape


def test_feature_names_cover_all_inputs(rows):
    names = set(build_preprocessor().fit(rows)[-1].get_feature_names_out())
    for col in [c for c in CATEGORICAL + NUMERIC if c != "SeniorCitizen"]:
        assert any(n == col or n.startswith(col + "_") for n in names), col
    assert "n_protective_addons" in names


def test_works_on_real_loader_dtypes(rows):
    # load() returns pandas categoricals, not strings; the pipeline must accept both.
    as_category = rows.astype({c: "category" for c in CATEGORICAL})
    assert build_preprocessor().fit_transform(as_category).shape[0] == len(rows)
