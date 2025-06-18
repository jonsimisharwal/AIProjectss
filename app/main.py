from fastapi import FastAPI
from app.schema import UserData
from app.model import make_prediction

app = FastAPI()

@app.post("/predict")
def predict(data: UserData):
    result = make_prediction(data)
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    }
