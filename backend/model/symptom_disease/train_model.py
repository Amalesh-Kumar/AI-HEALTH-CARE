import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dataset/dataset.csv")

# Separate features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Encode symptom text values
symptom_encoder = LabelEncoder()
for col in X.columns:
    X[col] = symptom_encoder.fit_transform(X[col])

# Encode disease labels
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model + encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, symptom_encoder, disease_encoder), f)

print("✅ Symptom disease model trained successfully")
