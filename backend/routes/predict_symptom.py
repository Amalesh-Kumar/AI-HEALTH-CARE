import os
import pickle
from fastapi import APIRouter
import pandas as pd

router = APIRouter()

# 🔥 Absolute path (NO relative path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "model", "symptom_disease", "model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model, symptom_encoder, disease_encoder = pickle.load(f)

symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

@router.post("/predict/symptom")
def predict_symptom(symptoms: list[str]):
    input_data = {col: "" for col in symptom_columns}

    for i, symptom in enumerate(symptoms):
        if i < 17:
            input_data[f"Symptom_{i+1}"] = symptom

    df = pd.DataFrame([input_data])

    for col in df.columns:
        df[col] = symptom_encoder.transform(df[col])

    prediction = model.predict(df)
    disease = disease_encoder.inverse_transform(prediction)[0]

    return {"predicted_disease": disease}
