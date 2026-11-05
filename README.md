# House Price Prediction

Predicts house prices from the Housing dataset. A scikit-learn Pipeline (categorical encoding + Gradient Boosting on log-transformed prices) is trained, tracked in MLflow, saved as one file, and served through a FastAPI service that runs locally or in Docker.

```
data/raw/Housing.csv --> src/data.py --> src/train.py --> models/house_price_pipeline.pkl
   (tracked by DVC)      load, clean,     Pipeline fit,          |
                         split            evaluate, MLflow       +--> src/predict.py  (local predictions)
                                                                 +--> src/api.py      (FastAPI: /health, /predict)
                                                                          |
                                                                     Docker / Compose (api + mlflow)
```

## Layout

```
mlops-service-project/
├── configs/config.yaml        model and training settings
├── data/
│   ├── raw/Housing.csv        raw data (versioned with DVC, not stored in Git)
│   └── processed/             optional output of preprocess_data
├── models/                    house_price_pipeline.pkl
├── reports/                   metrics.json
├── src/
│   ├── data.py                load_data, preprocess_data, split_data
│   ├── train.py               train_model, main (training, report, MLflow logging)
│   ├── evaluate.py            evaluate_model (MAE, MSE, RMSE, R2)
│   ├── predict.py             load the saved pipeline and predict raw records
│   └── api.py                 FastAPI app
├── tests/
├── Dockerfile                 API image
├── Dockerfile.train           training image
├── docker-compose.yml         api + mlflow (+ train profile)
├── .dockerignore
├── requirements.txt
├── requirements-train.txt
└── README.md
```

## Model

The saved file is a single scikit-learn `Pipeline`:

1. `preprocessor`: `ColumnTransformer` that encodes the yes/no columns and `furnishingstatus` (ordinal encoding; remaining columns pass through).
2. `model`: `TransformedTargetRegressor` around `GradientBoostingRegressor`, trained on `log1p(price)` and converted back to the original scale at prediction time.

Because the encoding is inside the saved pipeline, prediction takes raw values such as `"yes"` and `"furnished"`, and training, local prediction and the API all use exactly the same preprocessing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull
```

`dvc pull` restores `data/raw/Housing.csv`. The DVC remote in this project is a local folder (`../dvc-storage`), so it only works on the machine that created it. On another machine, place `Housing.csv` in `data/raw/` yourself, or point DVC at a shared remote.

## Usage

Run everything from the project root.

```bash
python src/train.py          # train, evaluate, save model and reports/metrics.json
python src/predict.py        # predict a sample record with the saved model
pytest tests -v              # run the tests
uvicorn src.api:app --port 8000
```

### Training and MLflow

`train.py` logs parameters, metrics and artifacts to MLflow. By default runs go to a local `mlruns/` folder. To log to the MLflow server started by Docker Compose, set the tracking URI:

```bash
MLFLOW_TRACKING_URI=http://localhost:<mlflow-port> python src/train.py
```

If `mlflow` is not installed, training skips logging.

## API

Swagger UI: `http://localhost:8000/docs`

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check, returns `{"status": "ok"}` |
| `POST /predict` | One house, returns `{"predicted_price": ...}` |
| `POST /predict-batch` | CSV upload (up to 1000 rows), returns a list of prices |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "yes", "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}'
```

Requests are validated with Pydantic. Missing fields, wrong types, unknown categories or non-positive area return `422` with a description of the problem.

## Docker

```bash
docker compose up -d --build                     # API (port 8000) + MLflow
docker compose ps
curl http://localhost:8000/health
docker compose --profile train run --rm train    # train inside a container, log to MLflow
docker compose logs api
docker compose down                              # stop; MLflow data stays on its volume
```

The MLflow UI port is shown by `docker compose ps`. Do not use `docker compose down -v` unless you want to delete the MLflow data.

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

Held-out test split (109 rows of 545):

| Metric | Value |
|---|---|
| MAE | 944,069 |
| RMSE | 1,311,434 |
| R2 | 0.660 |

### Local vs container consistency

The same valid house record was sent to each execution method:

| Execution method | Predicted price | Difference from local Python |
|---|---|---|
| Local Python (`joblib.load` + `predict`) | 11,858,369.438975034 | n/a |
| Docker API (`POST /predict`) | 11,858,369.438975032 | 1.9e-9 |

Tolerance: relative difference <= 1e-6. Observed difference is far below it and comes from floating-point rounding.

## Tests

```bash
pytest tests -v
```

Tests use small synthetic data and a temporary folder, so they need neither the real dataset nor a saved model.

## Reproducing the workflow

```bash
git clone <repo-url> && cd mlops-service-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull                                   # or place Housing.csv in data/raw/
python src/train.py
docker compose up -d --build
curl http://localhost:8000/health
```

Git stores the code and the `.dvc` pointer, DVC restores the exact dataset version, pinned requirements keep library versions the same, and Docker Compose provides the same runtime for the API and MLflow.

## Notes

- Paths are relative, so run scripts from the project root.
- Secrets, `.env` files, virtual environments, `mlruns/` and trained model files are not committed.