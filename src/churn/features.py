import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL, NUMERIC

PROTECTIVE_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
INTERNET_ADDONS = PROTECTIVE_ADDONS + ["StreamingTV", "StreamingMovies"]

SCALED = ["tenure", "MonthlyCharges", "TotalCharges", "n_protective_addons"]
BINARY = ["SeniorCitizen"]


def check_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CATEGORICAL + NUMERIC) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    bad = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if (df[c] < 0).any()]
    if bad:
        raise ValueError(f"Negative values in: {bad}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = check_columns(df).copy()
    # "No internet service" just repeats InternetService == "No"; folding it into "No"
    # removes six redundant one-hot columns without losing information.
    for col in INTERNET_ADDONS:
        df[col] = df[col].astype(str).replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].astype(str).replace("No phone service", "No")
    df["n_protective_addons"] = (df[PROTECTIVE_ADDONS] == "Yes").sum(axis=1)
    return df


def build_preprocessor(scale: bool = True) -> Pipeline:
    # Trees don't need scaling, and unscaled splits read as "tenure <= 17 months".
    columns = ColumnTransformer(
        [
            ("numeric", StandardScaler() if scale else "passthrough", SCALED),
            ("binary", "passthrough", BINARY),
            (
                "onehot",
                # An unseen category at predict time becomes all zeros instead of an error.
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return make_pipeline(FunctionTransformer(add_features), columns)
