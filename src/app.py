import os
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

# --- THE DEFINITIVE DOCKER PATH FIX ---
# We point to the specific folder that contains 'MLmodel' and 'model.pkl'
if os.path.exists("/app"):
    # Based on your image, the model files are inside the 'artifacts' folder
    model_path = "/app/mlruns/0/models/m-dac6ecb3e937423db8ccf9a99d002db3/artifacts"
else:
    model_path = "models:/diabetes_classifier/Staging"

print(f"🚀 Environment: {'Docker' if os.path.exists('/app') else 'Local'}")
print(f"📂 Loading model from: {model_path}")

model = mlflow.sklearn.load_model(model_path)
# ---------------------------------------

@app.get("/")
def home():
    return {"status": "Running", "model_source": model_path}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return {"diabetes_prediction": int(prediction[0])}