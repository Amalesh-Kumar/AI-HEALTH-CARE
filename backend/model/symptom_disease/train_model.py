import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dataset/dataset.csv")

# Split features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# ✅ Correct encoder for multiple symptom columns
symptom_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = symptom_encoder.fit_transform(X)

# Encode disease labels
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_encoded, y_encoded)

# Save model + encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, symptom_encoder, disease_encoder), f)

print("✅ Symptom disease model trained successfully")
