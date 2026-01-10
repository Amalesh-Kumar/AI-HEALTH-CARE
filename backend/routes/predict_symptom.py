import os
import pickle
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

router = APIRouter()

# -----------------------------
# Request body schema
# -----------------------------
class SymptomRequest(BaseModel):
    symptoms: list[str]

# -----------------------------
# Load model safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "model", "symptom_disease", "model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model, symptom_encoder, disease_encoder = pickle.load(f)

symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

# -----------------------------
# Prediction endpoint
# -----------------------------
@router.post("/predict/symptom")
def predict_symptom(request: SymptomRequest):

    # Extract symptoms correctly
    symptoms = request.symptoms

    input_data = {col: "" for col in symptom_columns}

    for i, symptom in enumerate(symptoms):
        if i < 17:
            input_data[f"Symptom_{i+1}"] = symptom

    df = pd.DataFrame([input_data])

    df_encoded = symptom_encoder.transform(df)

    prediction = model.predict(df_encoded)
    disease = disease_encoder.inverse_transform(prediction)[0]

    return {
        "predicted_disease": disease
    }
