from flask import Flask, request, jsonify
import joblib
import pandas as pd
from genai import generate_support_message
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load ML model once
model = joblib.load("ML/risk_model.pkl")

risk_map = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

@app.route("/trend", methods=["POST"])
def trend():
    try:
        data = request.json
        mood_history = data.get("mood_history", [])

        if len(mood_history) < 7:
            return jsonify({"error": "At least 7 days required"}), 400

        avg_last_7 = sum(mood_history[-7:]) / 7
        current_mood = mood_history[-1]

        drop_detected = current_mood < avg_last_7

        return jsonify({
            "average_last_7": round(avg_last_7, 2),
            "current_mood": current_mood,
            "drop_detected": drop_detected
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/")
def home():
    return "MindGuard AI Backend Running"

@app.route("/analyze", methods=["POST"])
def analyze():

    required_fields = ["mood","sleep","stress","study","screen","appetite","social"]

    data = request.json
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    try:
        data = request.json

        mood = data["mood"]
        sleep = data["sleep"]
        stress = data["stress"]
        study = data["study"]
        screen = data["screen"]
        appetite = data["appetite"]
        social = data["social"]

        input_df = pd.DataFrame([{
            "mood": mood,
            "sleep": sleep,
            "stress": stress,
            "study": study,
            "screen": screen,
            "appetite": appetite,
            "social": social
        }])

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        risk_label = risk_map[prediction]
        confidence_high = probabilities[2]

        # Issue summary logic
        issues = []
        if sleep < 5:
            issues.append("Low sleep")
        if stress > 7:
            issues.append("High stress")
        if mood < 4:
            issues.append("Low mood")

        issue_summary = ", ".join(issues)

        # Call GenAI
        message = generate_support_message(
            risk_label,
            issue_summary,
            mood_score=mood,
            stress_level=stress
        )

        return jsonify({
    "risk": risk_label,
    "probabilities": {
        "low": round(float(probabilities[0]), 2),
        "moderate": round(float(probabilities[1]), 2),
        "high": round(float(probabilities[2]), 2)
    },
    "message": message
})
    


    

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


    


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)