# House Price Prediction

Predicts house prices from the Housing dataset with a Gradient Boosting model trained on log-transformed prices. Training runs are logged to MLflow (optional).

```
data/raw/Housing.csv --> src/data.py --> src/train.py --> src/evaluate.py --> models/house_price_pipeline.pkl
                         load/encode/     Gradient Boosting   held-out              |
                         split            + MLflow logging    metrics         src/predict.py
```

## Layout

```
mlops-service-project/
├── configs/config.yaml     model and training settings
├── data/
│   ├── raw/Housing.csv
│   └── processed/          processed_data.csv (written by preprocess_data)
├── models/                 house_price_pipeline.pkl
├── reports/                metrics.json
├── src/
│   ├── data.py             load_data, preprocess_data, split_data
│   ├── train.py            trains the model, saves it, writes the report, logs to MLflow
│   ├── evaluate.py         evaluate_model (MAE, MSE, RMSE, R2)
│   └── predict.py          load the saved model and predict on new records
└── tests/                  conftest.py, test_pipeline.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn pyyaml joblib pytest mlflow
```

`mlflow` is optional. Without it, training skips logging.

## Usage

Run everything from the project root.

```bash
python src/train.py          # train, evaluate, save model and reports/metrics.json
python src/predict.py        # predict a sample record with the saved model
pytest tests -v              # run the tests
mlflow ui                    # browse logged runs at http://127.0.0.1:5000
```

## Configuration

`configs/config.yaml`:

```yaml
model:
  type: GradientBoostingRegressor
  n_estimators: 300
  max_depth: 3
  learning_rate: 0.05
  subsample: 0.8
training:
  test_size: 0.2
  random_state: 42
```

## Results

Latest run on the held-out test split:

| Metric | Value |
|---|---|
| MAE | 941,879 |
| RMSE | 1,303,160 |
| R2 | 0.664 |

## Tests

`tests/test_pipeline.py` uses a small synthetic dataset and a temporary folder, so it needs no real data or saved model.

- Preprocessing encodes valid input correctly
- Invalid categories become NaN
- Train/test split sizes are correct
- Prediction output has the right shape, and values are finite and positive

## Notes

- `preprocess_data` writes `data/processed/processed_data.csv` every time it is called.
- `predict.py` applies `preprocess_data` to input records, so they can use raw values like `"yes"` and `"furnished"`.
- Paths are relative, so scripts must be run from the project root.