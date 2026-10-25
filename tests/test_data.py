import pandas as pd
import pytest

from churn import data

HEADER = (
    "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
    "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
    "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
)
ROWS = (
    "0001-A,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,"
    "Month-to-month,Yes,Electronic check,29.85,29.85,Yes\n"
    "0002-B,Male,1,No,No,0,Yes,No,DSL,Yes,No,Yes,No,No,No,Two year,No,Mailed check,56.95, ,No\n"
)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / data.FILENAME
    path.write_text(HEADER + ROWS)
    return path


def test_load_fills_blank_total_charges_with_zero(csv_path):
    df = data.load(csv_path)
    assert df.loc[1, "TotalCharges"] == 0
    assert df["TotalCharges"].dtype == float


def test_load_encodes_target_as_int(csv_path):
    df = data.load(csv_path)
    assert df["Churn"].tolist() == [1, 0]


def test_load_drops_customer_id(csv_path):
    assert "customerID" not in data.load(csv_path).columns


def test_load_returns_expected_columns(csv_path):
    df = data.load(csv_path)
    assert set(df.columns) == set(data.CATEGORICAL + data.NUMERIC + [data.TARGET])
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in data.CATEGORICAL)


def test_download_rejects_bad_checksum(tmp_path):
    (tmp_path / data.FILENAME).write_text("not the real file")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.download(tmp_path)
