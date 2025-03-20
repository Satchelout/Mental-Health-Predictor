from flask import Flask, render_template, request,jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load trained model, scaler, and label encoder
model = joblib.load("mental_health_knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Landing Page


@app.route('/')
def landing():
    return render_template("landing.html")  # First page (landing)

# Assessment Page
@app.route('/assessment')
def assessment():
    return render_template("index.html")  # When 'Take Assessment' is clicked

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form inputs to float
        input_data = [float(request.form[key]) for key in request.form.keys()]

        # Normalize input using the scaler
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Decode prediction label
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return render_template("result.html", prediction=predicted_label)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
