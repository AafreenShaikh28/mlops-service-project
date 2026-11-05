
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv("data/raw/Housing.csv")
    return df

def preprocess_data(df, save=False):
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    if save:
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "processed_data.csv", index=False)

    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
