import json
import joblib
import yaml
import hashlib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import __version__ as sklearn_version

from data import load_data, preprocess_data, split_data
from evaluate import evaluate_model

BASE_DIR = Path(__file__).resolve().parent.parent


def train_model(X_train, y_train, config):
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    preprocessor = ColumnTransformer(transformers=[
        ('binary', OrdinalEncoder(categories=[['no', 'yes']] * len(binary_cols)), binary_cols),
        ('furnishing', OrdinalEncoder(categories=[['unfurnished', 'semi-furnished', 'furnished']]), ['furnishingstatus'])
    ], remainder='passthrough')

    regressor = TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(
            n_estimators=config['model']['n_estimators'],
            max_depth=config['model']['max_depth'],
            learning_rate=config['model']['learning_rate'],
            subsample=config['model']['subsample'],
            random_state=config['training']['random_state']
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', regressor)])
    pipeline.fit(X_train, y_train)
    return pipeline





def main():
    with open(BASE_DIR / "configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    dataset_hash = hashlib.sha256((BASE_DIR / "data/raw/Housing.csv").read_bytes()).hexdigest()

    df = load_data()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    model = train_model(X_train, y_train, config)

    metrics = evaluate_model(model, X_test, y_test)

    model_path = BASE_DIR / "models/house_price_pipeline.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    report = {
        "metrics": metrics,
        "config": config,
        "dataset_sha256": dataset_hash,
        "dataset_rows": len(df),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "sklearn_version": sklearn_version
    }

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    with open(reports_dir / "metrics.json", "w") as file:
        json.dump(report, file, indent=4)

    try:
        import mlflow
        import os
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment("House Price Prediction")

        mlflow.set_experiment("House Price Prediction")

        with mlflow.start_run():
            mlflow.log_params({
                "model_type": config['model']['type'],
                "n_estimators": config['model']['n_estimators'],
                "max_depth": config['model']['max_depth'],
                "learning_rate": config['model']['learning_rate'],
                "subsample": config['model']['subsample'],
                "random_state": config['training']['random_state'],
                "test_size": config['training']['test_size']
            })

            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(reports_dir / "metrics.json"))
            mlflow.log_artifact(str(model_path))

            print("MLflow run logged successfully!")

    except ImportError:
        print("MLflow not installed. Skipping MLflow logging.")

    print("Training completed!")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()