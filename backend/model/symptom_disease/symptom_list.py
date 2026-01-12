import pandas as pd

df = pd.read_csv("dataset/Symptom-severity.csv")
SYMPTOMS = df["Symptom"].str.strip().tolist()
