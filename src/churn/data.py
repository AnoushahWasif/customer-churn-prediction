import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

# Pinned to a commit so the file can't change under us; the checksum catches a tampered download.
SOURCE_COMMIT = "1fd6fd70906479be1712e64cfc0ea89f58466629"
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{SOURCE_COMMIT}/data/Telco-Customer-Churn.csv"
)
SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
FILENAME = "telco_churn.csv"

CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"


def data_dir() -> Path:
    return Path(os.environ.get("CHURN_DATA_DIR", "data/raw"))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest_dir: Path | None = None) -> Path:
    dest_dir = dest_dir or data_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / FILENAME

    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)

    actual = sha256_of(path)
    if actual != SHA256:
        raise ValueError(f"Checksum mismatch for {path}: expected {SHA256}, got {actual}")
    return path


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or data_dir() / FILENAME
    df = pd.read_csv(path, dtype={c: "category" for c in CATEGORICAL})

    # 11 new customers (tenure 0) have a blank TotalCharges. They haven't been billed yet,
    # so 0 is the true value, not a missing one.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", "0"))
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns="customerID")


if __name__ == "__main__":
    print(download())
