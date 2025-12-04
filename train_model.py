import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/cuaca.csv")

print("\n=== PREVIEW DATA ===")
print(df.head())
print("\n=== Kolom tersedia ===")
print(df.columns)

# Pastikan nama kolom sesuai dataset
required_cols = ["Suhu", "Kelembaban", "Angin", "Cuaca"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset.")

# =========================
# 2. Encode Target
# =========================
le_target = LabelEncoder()
df["Cuaca_label"] = le_target.fit_transform(df["Cuaca"])

X = df[["Suhu", "Kelembaban", "Angin"]]
y = df["Cuaca_label"]

# =========================
# 3. Train Model
# =========================
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# =========================
# 4. Save Model
# =========================
os.makedirs("models", exist_ok=True)

artifact = {
    "model": model,
    "le_target": le_target,
    "feature_cols": ["Suhu", "Kelembaban", "Angin"]
}

joblib.dump(artifact, "models/model.joblib")

print("\n=== MODEL BERHASIL DIBUAT ===")
print("Disimpan ke: models/model.joblib")
