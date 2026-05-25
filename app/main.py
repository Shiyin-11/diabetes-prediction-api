from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("models/diabetes_model.pkl")


# Define input data structure
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# Home route
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


# Prediction route
@app.post("/predict")
def predict(data: PatientData):

    features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    result = "High Diabetes Risk" if prediction == 1 else "Low Diabetes Risk"

    return {
        "prediction": result,
        "probability": round(float(probability), 2)
    }