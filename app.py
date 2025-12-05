from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
 
app = Flask(__name__)
 
# Load models
MODEL_DIR = "models"
yield_model = joblib.load(os.path.join(MODEL_DIR, "yield_model.pkl"))
pest_model = joblib.load(os.path.join(MODEL_DIR, "pest_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
 
@app.route("/")
def home():
    return "Agriculture API running. Use /predict_yield and /predict_pest via POST JSON."
 
@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    data = request.get_json()
    # expected keys: temperature, rainfall, humidity, soil_moisture, gdd
    X = np.array([[data.get("temperature",0),
                   data.get("rainfall",0),
                   data.get("humidity",0),
                   data.get("soil_moisture",0),
                   data.get("gdd",0)]])
    Xs = scaler.transform(X)
    pred = yield_model.predict(Xs)
    return jsonify({"predicted_yield": float(pred[0])})
 
@app.route("/predict_pest", methods=["POST"])
def predict_pest():
    data = request.get_json()
    X = np.array([[data.get("temperature",0),
                   data.get("rainfall",0),
                   data.get("humidity",0),
                   data.get("soil_moisture",0),
                   data.get("gdd",0)]])
    Xs = scaler.transform(X)
    pred = pest_model.predict(Xs)
    prob = pest_model.predict_proba(Xs)[:,1][0] if hasattr(pest_model, "predict_proba") else None
    return jsonify({"pest_risk_label": int(pred[0]), "pest_risk_prob": float(prob) if prob is not None else None})
 
@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")
 
if __name__ == "__main__":
    app.run(debug=True)
 
 