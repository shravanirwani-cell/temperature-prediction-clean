from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# load model
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Temperature Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    # convert input to dataframe
    df = pd.DataFrame([{
        "co": data["co"],
        "humidity": data["humidity"],
        "light": data["light"],
        "lpg": data["lpg"],
        "motion": data["motion"],
        "smoke": data["smoke"]
    }])

    prediction = model.predict(df)[0]

    return {"predicted_temperature": float(prediction)}