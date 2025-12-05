import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os
 
os.makedirs("models", exist_ok=True)
 
# Load
df = pd.read_csv("data/raw/climate_crop_data.csv")
 
# Features and targets
feature_cols = ["temperature", "rainfall", "humidity", "soil_moisture", "gdd"]
X = df[feature_cols]
y_yield = df["yield"]
y_pest = (df["pest_risk"] > 0.25).astype(int) if "pest_risk" in df.columns else (df["gdd"]>1500).astype(int)
 
# Split
X_train, X_test, y_yield_train, y_yield_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_pest_train, y_pest_test = train_test_split(X, y_pest, test_size=0.2, random_state=42)
 
# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
X_train_scaled_cls = scaler.transform(X_train_cls)
X_test_scaled_cls = scaler.transform(X_test_cls)
 
# Train models
yield_model = RandomForestRegressor(n_estimators=200, random_state=42)
pest_model = RandomForestClassifier(n_estimators=200, random_state=42)
 
yield_model.fit(X_train_scaled, y_yield_train)
pest_model.fit(X_train_scaled_cls, y_pest_train)
 
# Save
joblib.dump(yield_model, "models/yield_model.pkl")
joblib.dump(pest_model, "models/pest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
 
print("Models trained and saved to models/ directory")
 
 