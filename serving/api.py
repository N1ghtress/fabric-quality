from fastapi import FastAPI, Request
from joblib import load
from pandas import read_json, read_csv, DataFrame
import random
import sklearn
from pydantic import BaseModel
from typing import List

class Report(BaseModel):
    data: List
    y_true: int | None = None
    y_pred: int | None = None

TRAIN_THRESHOLD = 10
random.seed(42)
REF_DATA = read_csv('/data/ref_data.zip', skiprows=lambda x: x > 0 and random.random() > 0.01)
try:
    PROD_DATA = read_csv('/data/prod_data.csv')
except FileNotFoundError:
    PROD_DATA = DataFrame()
SCALER = load('/artifacts/scaler.pickle')
ESTIMATOR = load('/artifacts/estimator.pickle')

app = FastAPI()

@app.post('/predict')
def predict_data(r: Report):
    # X = EMBEDDING.transform(X)
    X = SCALER.transform([r.data])
    y_hat = ESTIMATOR.predict(X)
    return int(y_hat[0])

@app.post('/feedback')
def feedback_data(r: Report):
    row = r.data + [r.y_true] + [r.y_pred]
    PROD_DATA = PROD_DATA.append(row)
    print(PROD_DATA.head())
    
