from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load model from MLflow Staging
model_name = "diabetes_classifier"
model = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")
@app.get("/")
def home():
    return {"status": "Model is running in Staging"}
@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return {"diabetes_prediction": int(prediction[0])}
