from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/student_model.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Student Performance Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    study_hours = data["study_hours"]
    attendance = data["attendance"]
    previous_marks = data["previous_marks"]

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks
    }])

    prediction = model.predict(input_data)

    return jsonify({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "predicted_marks": round(float(prediction[0]), 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)