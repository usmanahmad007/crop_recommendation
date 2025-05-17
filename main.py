from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load Model
model = joblib.load("crop_model.pkl")

# FastAPI app
app = FastAPI()

# Input Schema
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def read_root():
    return {"message": "Crop Recommendation API"}

@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = [[
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]]
    prediction = model.predict(input_data)
    return {"recommended_crop": prediction[0]}
