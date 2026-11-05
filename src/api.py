from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.data import preprocess_data

from src.predict import load_model, predict

app = FastAPI(title="House Price Prediction API", version="1.0.0")
model = load_model()

YesNo = Literal["yes", "no"]


class HouseInput(BaseModel):
    area: int = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    stories: int = Field(ge=0)
    mainroad: YesNo
    guestroom: YesNo
    basement: YesNo
    hotwaterheating: YesNo
    airconditioning: YesNo
    parking: int = Field(ge=0)
    prefarea: YesNo
    furnishingstatus: Literal["unfurnished", "semi-furnished", "furnished"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_price(payload: HouseInput):
    prediction = predict(model, [payload.model_dump()])
    return {"predicted_price": float(prediction[0])}

import pandas as pd
from fastapi import File, HTTPException, UploadFile
from pydantic import ValidationError

MAX_BATCH_ROWS = 1000


@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"Maximum {MAX_BATCH_ROWS} rows")
    try:
        records = [HouseInput(**row).model_dump() for row in df.to_dict("records")]
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_input=False))
    predictions = predict(model, records)
    return {"predicted_prices": [float(p) for p in predictions]}