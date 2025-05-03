# fertilizer_api/app.py
import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("fertilizer_model.pkl")
le_soil = joblib.load("soil_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")
le_fert = joblib.load("fertilizer_encoder.pkl")

# Utility function to normalize categorical input
def normalize_text(s: str) -> str:
    return s.strip().lower()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload with:
    {
      "Temperature": float,
      "Humidity": float,
      "Moisture": float,
      "Soil_Type": str,
      "Crop_Type": str,
      "Nitrogen": float,
      "Potassium": float,
      "Phosphorus": float
    }
    Returns:
      { "fertilizer": str }
    """
    try:
        data = request.get_json()

        # Parse numeric inputs
        temp = float(data["Temperature"])
        hum  = float(data["Humidity"])
        moist= float(data["Moisture"])
        n    = float(data["Nitrogen"])
        k    = float(data["Potassium"])
        p    = float(data["Phosphorus"])

        # Normalize and encode categorical inputs
        soil_raw = normalize_text(data["Soil_Type"])
        crop_raw = normalize_text(data["Crop_Type"])

        soil_enc = le_soil.transform([soil_raw])[0]
        crop_enc = le_crop.transform([crop_raw])[0]

        # Prepare feature array in the same order used in training
        features = np.array([[
            temp,
            hum,
            moist,
            soil_enc,
            crop_enc,
            n,
            k,
            p
        ]])

        # Make prediction
        pred_enc = model.predict(features)[0]
        pred_fertilizer = le_fert.inverse_transform([pred_enc])[0]

        return jsonify({"fertilizer": pred_fertilizer})

    except KeyError as ke:
        return jsonify({"error": f"Missing key in JSON: {ke}"}), 400
    except ValueError as ve:
        return jsonify({"error": f"Invalid value: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     # Run on 0.0.0.0 to allow external access, port 5007
#     app.run(host="0.0.0.0", port=5007, debug=True)
