from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="California Housing API", version="1.0")

class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API MLOps California Housing"}

@app.post("/predict")
def predict(features: HousingFeatures):
    input_data = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms, 
        features.AveBedrms, features.Population, features.AveOccup, 
        features.Latitude, features.Longitude
    ]])
    fake_prediction = float(input_data[0][0] * 0.5 + 2.0)
    return {"predicted_median_house_value": fake_prediction}
