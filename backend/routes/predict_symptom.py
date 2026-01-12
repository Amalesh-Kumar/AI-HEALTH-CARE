import os
import pickle
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

router = APIRouter()

# -----------------------------
# Request schema
# -----------------------------
class SymptomRequest(BaseModel):
    symptoms: list[str]

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model", "symptom_disease")
DATASET_DIR = os.path.join(MODEL_DIR, "dataset")

# -----------------------------
# Load ML model
# -----------------------------
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model, all_symptoms, disease_encoder = pickle.load(f)

# Normalize symptom list from training
all_symptoms = [
    s.strip().lower().replace(" ", "_") for s in all_symptoms
]

# -----------------------------
# Load medical datasets
# -----------------------------
desc_df = pd.read_csv(os.path.join(DATASET_DIR, "symptom_description.csv"))
prec_df = pd.read_csv(os.path.join(DATASET_DIR, "symptom_precaution.csv"))
sev_df = pd.read_csv(os.path.join(DATASET_DIR, "Symptom-severity.csv"))

# Normalize datasets
desc_df["Disease"] = desc_df["Disease"].str.strip()

prec_df["Disease"] = prec_df["Disease"].str.strip()

sev_df["Symptom"] = (
    sev_df["Symptom"]
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# -----------------------------
# Prediction endpoint
# -----------------------------
@router.post("/predict/symptom")
def predict_symptom(request: SymptomRequest):

    # Normalize user symptoms
    user_symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in request.symptoms
    ]

    # -------------------------
    # Multi-hot encoding
    # -------------------------
    input_vector = pd.DataFrame(
        [[1 if s in user_symptoms else 0 for s in all_symptoms]],
        columns=all_symptoms
    )

    # -------------------------
    # Predict disease
    # -------------------------
    pred = model.predict(input_vector)
    disease = disease_encoder.inverse_transform(pred)[0]

    # -------------------------
    # Fetch description
    # -------------------------
    description_row = desc_df.loc[
        desc_df["Disease"] == disease, "Description"
    ]
    description = (
        description_row.values[0]
        if not description_row.empty
        else "Description not available"
    )

    # -------------------------
    # Fetch precautions
    # -------------------------
    precautions = []
    prec_row = prec_df.loc[prec_df["Disease"] == disease]
    if not prec_row.empty:
        precautions = prec_row.iloc[0, 1:].dropna().tolist()

    # -------------------------
    # Severity calculation
    # -------------------------
    severity_score = 0
    for s in user_symptoms:
        match = sev_df.loc[sev_df["Symptom"] == s, "weight"]
        if not match.empty:
            severity_score += int(match.values[0])

    if severity_score < 10:
        risk = "Low"
    elif severity_score < 20:
        risk = "Medium"
    else:
        risk = "High"

    # -------------------------
    # Final response
    # -------------------------
    return {
        "predicted_disease": disease,
        "description": description,
        "precautions": precautions,
        "severity_score": severity_score,
        "risk_level": risk
    }
