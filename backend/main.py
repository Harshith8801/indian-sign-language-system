from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

LABELS = [
    '0','1','2','3','4','5','6','7','8','9',
    'Hello','Thank-you','Yes','No'
]

clf = joblib.load("model/gesture_classifier.pkl")
scaler = joblib.load("model/gesture_scaler.pkl")

class FeatureInput(BaseModel):
    features: list

@app.post("/predict")
def predict_number(data: FeatureInput):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_idx = int(clf.predict(X_scaled)[0])
    pred_label = LABELS[pred_idx]
    return {"prediction": pred_label}

