import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Load datasets
main_data = pd.read_csv(os.path.join(DATASET_DIR, "dataset.csv"))
severity = pd.read_csv(os.path.join(DATASET_DIR, "Symptom-severity.csv"))

# Get unique symptom list
all_symptoms = severity["Symptom"].str.strip().unique().tolist()

# Create binary feature matrix
X = pd.DataFrame(0, index=main_data.index, columns=all_symptoms)

for i in range(1, 18):
    col = f"Symptom_{i}"
    for idx, symptom in main_data[col].items():
        if pd.notna(symptom):
            symptom = symptom.strip()
            if symptom in X.columns:
                X.loc[idx, symptom] = 1

# Target
y = main_data["Disease"]

# Encode disease labels
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X, y_encoded)

# Save model correctly (🔥 IMPORTANT)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, all_symptoms, disease_encoder), f)

print("✅ Multi-hot symptom model trained & saved successfully")
