
import os
import joblib
import pandas as pd
from pathlib import Path

try:
    from src.data import preprocess_data
except ImportError:
    from data import preprocess_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models/house_price_pipeline.pkl"))


def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, records):
    df = preprocess_data(pd.DataFrame(records))
    return model.predict(df[model.feature_names_in_])

def main():
    model = load_model()

    records = [{
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

    predictions = predict(model, records)

    for i, price in enumerate(predictions):
        print(f"Record {i + 1}: ₹{price:,.2f}")


if __name__ == "__main__":
    main()
