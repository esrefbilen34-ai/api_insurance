import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import json

np.random.seed(42)
N = 5000

arac_tipleri = ["Sedan", "SUV", "Hatchback", "Pickup", "Minivan"]
markalar = ["Toyota", "Ford", "Renault", "Volkswagen", "Fiat", "Honda", "Hyundai", "BMW", "Mercedes", "Dacia"]
sehirler = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep"]

arac_tipi = np.random.choice(arac_tipleri, N)
marka = np.random.choice(markalar, N)
arac_yasi = np.random.randint(0, 20, N)
motor_hacmi = np.random.choice([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], N)
surucu_yasi = np.random.randint(18, 70, N)
surucu_deneyim = np.clip(surucu_yasi - 18 - np.random.randint(0, 5, N), 0, 50)
kaza_gecmisi = np.random.poisson(0.3, N)
trafik_cezasi = np.random.poisson(0.5, N)
sehir = np.random.choice(sehirler, N)
yillik_km = np.random.randint(5000, 60000, N)
garaj_var = np.random.choice([0, 1], N, p=[0.4, 0.6])

sehir_risk = {
    "İstanbul": 25, "Ankara": 18, "İzmir": 15,
    "Bursa": 12, "Antalya": 14, "Adana": 16,
    "Konya": 10, "Gaziantep": 13
}
marka_risk = {
    "BMW": 18, "Mercedes": 16, "Toyota": 5, "Honda": 6,
    "Volkswagen": 8, "Ford": 9, "Hyundai": 7,
    "Renault": 10, "Fiat": 11, "Dacia": 8
}
tip_risk = {
    "Pickup": 15, "SUV": 10, "Minivan": 8,
    "Sedan": 5, "Hatchback": 6
}

risk_skoru = (
    np.array([sehir_risk[s] for s in sehir]) +
    np.array([marka_risk[m] for m in marka]) +
    np.array([tip_risk[t] for t in arac_tipi]) +
    arac_yasi * 1.5 +
    motor_hacmi * 4 +
    np.where(surucu_yasi < 25, 20, 0) +
    np.where(surucu_yasi > 65, 10, 0) +
    np.maximum(0, 5 - surucu_deneyim) * 2 +
    kaza_gecmisi * 15 +
    trafik_cezasi * 5 +
    yillik_km / 3000 +
    np.where(garaj_var == 0, 5, -3) +
    np.random.normal(0, 5, N)
)

risk_skoru = np.clip(risk_skoru, 0, 100).astype(float)

df = pd.DataFrame({
    "arac_tipi": arac_tipi,
    "marka": marka,
    "arac_yasi": arac_yasi,
    "motor_hacmi": motor_hacmi,
    "surucu_yasi": surucu_yasi,
    "surucu_deneyim": surucu_deneyim,
    "kaza_gecmisi": kaza_gecmisi,
    "trafik_cezasi": trafik_cezasi,
    "sehir": sehir,
    "yillik_km": yillik_km,
    "garaj_var": garaj_var,
    "risk_skoru": risk_skoru
})

le_tip = LabelEncoder().fit(arac_tipleri)
le_marka = LabelEncoder().fit(markalar)
le_sehir = LabelEncoder().fit(sehirler)

df["arac_tipi_enc"] = le_tip.transform(df["arac_tipi"])
df["marka_enc"] = le_marka.transform(df["marka"])
df["sehir_enc"] = le_sehir.transform(df["sehir"])

features = [
    "arac_tipi_enc", "marka_enc", "arac_yasi", "motor_hacmi",
    "surucu_yasi", "surucu_deneyim", "kaza_gecmisi",
    "trafik_cezasi", "sehir_enc", "yillik_km", "garaj_var"
]

X = df[features]
y = df["risk_skoru"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae:.2f} puan")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

encoders = {
    "arac_tipleri": arac_tipleri,
    "markalar": markalar,
    "sehirler": sehirler
}
with open("encoders.json", "w", encoding="utf-8") as f:
    json.dump(encoders, f, ensure_ascii=False)

print("Model ve encoder kaydedildi.")
print(f"Feature importance:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")
