import numpy as np
import pandas as pd
import pytest

from data import preprocess_data, split_data
from train import train_model
from predict import predict

CONFIG = {
    "model": {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8},
    "training": {"random_state": 42, "test_size": 0.2}
}

BINARY_COLS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']


def make_df(n=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "price": rng.integers(2_000_000, 10_000_000, n),
        "area": rng.integers(2000, 9000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 4, n),
        "stories": rng.integers(1, 4, n),
        "parking": rng.integers(0, 3, n),
        "furnishingstatus": rng.choice(["unfurnished", "semi-furnished", "furnished"], n)
    })
    for col in BINARY_COLS:
        df[col] = rng.choice(["yes", "no"], n)
    return df


def make_record():
    return [{
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }]


@pytest.fixture(autouse=True)
def isolated_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_preprocess_encodes_valid_input():
    df = preprocess_data(make_df())
    assert not df.isna().any().any()
    assert set(df[BINARY_COLS].to_numpy().ravel()) <= {0, 1}
    assert set(df["furnishingstatus"]) <= {0, 1, 2}


def test_preprocess_flags_invalid_category():
    df = make_df()
    df.loc[0, "mainroad"] = "maybe"
    df.loc[1, "furnishingstatus"] = "luxury"
    result = preprocess_data(df)
    assert pd.isna(result.loc[0, "mainroad"])
    assert pd.isna(result.loc[1, "furnishingstatus"])


def test_split_data_sizes():
    df = preprocess_data(make_df(40))
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) == 32
    assert len(X_test) == 8
    assert "price" not in X_train.columns
    assert len(y_train) == len(X_train)


def test_prediction_output_shape_and_values():
    df = preprocess_data(make_df())
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train, CONFIG)

    records = make_record() * 3
    predictions = predict(model, records)

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()