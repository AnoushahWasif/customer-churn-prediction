import json
import shutil

import pandas as pd
import pytest

from churn import cli
from churn.model_store import ModelIntegrityError, load_model, sha256
from tests.test_metrics_train import labelled_rows


def raw_csv(df: pd.DataFrame, path):
    """Write rows the way the real Telco file looks: ids, Yes/No churn, blank TotalCharges."""
    out = df.copy()
    out.insert(0, "customerID", [f"{i:04d}-ABCDE" for i in range(len(out))])
    out["TotalCharges"] = out["TotalCharges"].round(2).astype(str)
    out.loc[out["tenure"] == 0, "TotalCharges"] = " "
    if "Churn" in out:
        out["Churn"] = out["Churn"].map({1: "Yes", 0: "No"})
    out.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    df = labelled_rows(800)
    df.loc[df.index[:3], "tenure"] = 0  # make sure the blank-TotalCharges rule is exercised
    data = raw_csv(df, root / "train.csv")
    assert cli.main(["train", "--data", str(data), "--out", str(root / "model")]) == 0
    return root / "model"


@pytest.fixture
def customers(tmp_path):
    df = labelled_rows(20).drop(columns="Churn")
    df.loc[df.index[0], "tenure"] = 0
    return raw_csv(df, tmp_path / "customers.csv")


def run_score(capsys, *args):
    code = cli.main(["score", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


# --- train


def test_train_writes_manifest_with_threshold_and_costs(trained):
    manifest = json.loads((trained / "model.json").read_text())
    assert manifest["sha256"] == sha256(trained / "model.joblib")
    assert 0 < manifest["threshold"] < 1
    assert manifest["costs"] == {"offer": 1.0, "missed": 5.0}
    assert manifest["model"] == "logistic_regression"


@pytest.mark.parametrize(
    "flags", [["--offer-cost", "0"], ["--missed-cost", "-1"], ["--missed-cost", "nan"]]
)
def test_train_rejects_bad_costs(tmp_path, capsys, flags):
    assert cli.main(["train", "--out", str(tmp_path), *flags]) == 2
    assert "positive finite" in capsys.readouterr().err


def test_train_without_data_says_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHURN_DATA_DIR", str(tmp_path / "nothing"))
    assert cli.main(["train", "--out", str(tmp_path / "m")]) == 2
    assert "make data" in capsys.readouterr().err


# --- score


def test_score_writes_one_row_per_customer(trained, customers, tmp_path, capsys):
    out = tmp_path / "scores.csv"
    code, _, err = run_score(capsys, customers, "--model", trained, "--out", out)
    assert code == 0
    result = pd.read_csv(out)
    assert list(result.columns) == ["customer_id", "churn_score", "offer"]
    assert len(result) == 20
    assert result["churn_score"].between(0, 1).all()
    assert set(result["offer"]) <= {"yes", "no"}
    assert "scored 20 customers" in err


def test_score_to_stdout(trained, customers, capsys):
    code, out, _ = run_score(capsys, customers, "--model", trained)
    assert code == 0
    assert out.splitlines()[0] == "customer_id,churn_score,offer"


def test_offer_follows_the_saved_threshold(trained, customers, tmp_path, capsys):
    out = tmp_path / "s.csv"
    run_score(capsys, customers, "--model", trained, "--out", out)
    result = pd.read_csv(out)
    threshold = json.loads((trained / "model.json").read_text())["threshold"]
    # Scores are rounded to 4 places in the file, so skip anything right at the line.
    clear = (result["churn_score"] - threshold).abs() > 1e-4
    expected = (result["churn_score"] >= threshold).map({True: "yes", False: "no"})
    assert (result["offer"][clear] == expected[clear]).all()


def test_rows_without_ids_get_line_numbers(trained, customers, tmp_path, capsys):
    df = pd.read_csv(customers, dtype=str, keep_default_na=False).drop(columns="customerID")
    df.to_csv(tmp_path / "noid.csv", index=False)
    code, out, _ = run_score(capsys, tmp_path / "noid.csv", "--model", trained)
    assert code == 0
    assert out.splitlines()[1].startswith("1,")


@pytest.mark.parametrize("evil", ['=HYPERLINK("http://x")', "+1+1", "-2+3", "@SUM(A1)"])
def test_formula_ids_are_neutralised(trained, customers, tmp_path, capsys, evil):
    df = pd.read_csv(customers, dtype=str, keep_default_na=False)
    df.loc[0, "customerID"] = evil
    df.to_csv(tmp_path / "evil.csv", index=False)
    _, out, _ = run_score(capsys, tmp_path / "evil.csv", "--model", trained)
    first = pd.read_csv(pd.io.common.StringIO(out), dtype=str)["customer_id"][0]
    assert first == "'" + evil


def test_unseen_category_is_scored_and_mentioned(trained, customers, tmp_path, capsys):
    df = pd.read_csv(customers, dtype=str, keep_default_na=False)
    df.loc[0, "PaymentMethod"] = "Crypto"
    df.to_csv(tmp_path / "new.csv", index=False)
    code, _, err = run_score(capsys, tmp_path / "new.csv", "--model", trained)
    assert code == 0
    assert "never seen in training" in err


def test_whitespace_around_values_is_ignored(trained, customers, tmp_path, capsys):
    df = pd.read_csv(customers, dtype=str, keep_default_na=False)
    padded = df.apply(lambda col: " " + col + " ")
    padded.columns = [f" {c}" for c in df.columns]
    padded.to_csv(tmp_path / "padded.csv", index=False)
    _, a, _ = run_score(capsys, customers, "--model", trained)
    _, b, _ = run_score(capsys, tmp_path / "padded.csv", "--model", trained)
    assert a == b


# --- bad input: clear message, exit code 2, no traceback, no output file


def write(tmp_path, text, name="bad.csv"):
    path = tmp_path / name
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return path


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "not a readable CSV"),
        ("customerID,tenure\n", "no customer rows"),
        ("customerID,tenure\nA,5\n", "missing columns"),
        (b"\xff\xfe\x00bad", "not a readable CSV"),
    ],
)
def test_bad_files(trained, tmp_path, capsys, content, message):
    out = tmp_path / "out.csv"
    code, _, err = run_score(capsys, write(tmp_path, content), "--model", trained, "--out", out)
    assert code == 2
    assert message in err
    assert "Traceback" not in err
    assert not out.exists()


@pytest.mark.parametrize(
    "column, value",
    [
        ("tenure", "abc"),
        ("tenure", "-1"),
        ("MonthlyCharges", ""),
        ("TotalCharges", "inf"),
        ("SeniorCitizen", "nan"),
    ],
)
def test_bad_numbers_name_the_column_and_line(trained, customers, tmp_path, capsys, column, value):
    df = pd.read_csv(customers, dtype=str, keep_default_na=False)
    df.loc[3, column] = value
    df.to_csv(tmp_path / "bad.csv", index=False)
    code, _, err = run_score(capsys, tmp_path / "bad.csv", "--model", trained)
    assert code == 2
    assert column in err and "line 5" in err


def test_blank_total_charges_only_allowed_for_new_customers(trained, customers, tmp_path, capsys):
    df = pd.read_csv(customers, dtype=str, keep_default_na=False)
    df.loc[1, "tenure"] = "12"
    df.loc[1, "TotalCharges"] = ""
    df.to_csv(tmp_path / "bad.csv", index=False)
    code, _, err = run_score(capsys, tmp_path / "bad.csv", "--model", trained)
    assert code == 2 and "TotalCharges" in err


def test_missing_input_file(trained, tmp_path, capsys):
    code, _, err = run_score(capsys, tmp_path / "nope.csv", "--model", trained)
    assert code == 2 and "not found" in err


def test_refuses_to_overwrite_input(trained, customers, capsys):
    before = customers.read_bytes()
    code, _, err = run_score(capsys, customers, "--model", trained, "--out", customers)
    assert code == 2 and "overwrite" in err
    assert customers.read_bytes() == before


def test_size_and_row_limits(trained, customers, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "MAX_ROWS", 5)
    code, _, err = run_score(capsys, customers, "--model", trained)
    assert code == 2 and "more than 5 rows" in err
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 10)
    code, _, err = run_score(capsys, customers, "--model", trained)
    assert code == 2 and "larger than" in err


# --- model integrity


def test_tampered_model_is_refused_before_loading(
    trained, customers, tmp_path, capsys, monkeypatch
):
    bad = tmp_path / "model"
    shutil.copytree(trained, bad)
    with (bad / "model.joblib").open("ab") as f:
        f.write(b"\x00")
    opened = []
    monkeypatch.setattr("churn.model_store.joblib.load", lambda p: opened.append(p))
    code, out, err = run_score(capsys, customers, "--model", bad)
    assert code == 2 and "hash does not match" in err
    assert opened == [] and out == ""


def test_expected_hash_from_outside_wins(trained, customers, capsys):
    code, _, err = run_score(capsys, customers, "--model", trained, "--sha256", "0" * 64)
    assert code == 2 and "hash does not match" in err
    good = sha256(trained / "model.joblib")
    assert run_score(capsys, customers, "--model", trained, "--sha256", good.upper())[0] == 0


@pytest.mark.parametrize(
    "manifest", ["", "not json", "[]", '{"threshold": 0.3}', '{"sha256": "ab", "threshold": 2}']
)
def test_broken_manifest_is_refused(trained, tmp_path, manifest):
    bad = tmp_path / "model"
    shutil.copytree(trained, bad)
    (bad / "model.json").write_text(manifest)
    with pytest.raises(ModelIntegrityError):
        load_model(bad)


def test_missing_model_folder(customers, tmp_path, capsys):
    code, _, err = run_score(capsys, customers, "--model", tmp_path / "none")
    assert code == 2 and "manifest" in err
