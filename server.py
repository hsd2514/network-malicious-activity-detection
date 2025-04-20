import flask
import joblib
import pandas as pd
import numpy as np
import os
import logging

app = flask.Flask(__name__)

COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app.logger.setLevel(logging.ERROR)

MODEL_FILENAME = 'mlp_model.joblib'
SCALER_FILENAME = 'scaler.joblib'
COLUMNS_FILENAME = 'model_columns.joblib'

if not all(os.path.exists(f) for f in [MODEL_FILENAME, SCALER_FILENAME, COLUMNS_FILENAME]):
    print(f"{COLOR_RED}FATAL ERROR: Required model/scaler/columns files not found.{COLOR_RESET}")
    print(f"Ensure '{MODEL_FILENAME}', '{SCALER_FILENAME}', and '{COLUMNS_FILENAME}' exist.")
    exit()
try:
    print("Loading predictive components...")
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    model_columns = joblib.load(COLUMNS_FILENAME)
    print("Components loaded successfully.")
    print(f"Model Classes known by loaded model: {model.classes_}")
except Exception as e:
    print(f"{COLOR_RED}FATAL ERROR: Failed to load components: {e}{COLOR_RESET}")
    exit()


@app.route('/')
def home():
    return "MLP Network Attack Detection API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    color_map = {
        'Denial of Service': COLOR_RED,
        'Malware': COLOR_RED,
        'Port Scanning': COLOR_YELLOW,
        'None': COLOR_GREEN
    }

    if not flask.request.is_json:
        print(f"{COLOR_RED}>>> Server Bad Request (400): Request content type must be application/json{COLOR_RESET}")
        return flask.jsonify({"error": "Request content type must be application/json"}), 400
    data = flask.request.get_json()
    if data is None:
         print(f"{COLOR_RED}>>> Server Bad Request (400): No JSON data received in request body{COLOR_RESET}")
         return flask.jsonify({"error": "No JSON data received in request body"}), 400
    missing_cols = [col for col in model_columns if col not in data]
    if missing_cols:
        error_message = f"Missing required features in JSON data: {missing_cols}"
        print(f"{COLOR_RED}>>> Server Bad Request (400): {error_message}{COLOR_RESET}")
        return flask.jsonify({"error": error_message}), 400
    try:
        input_df = pd.DataFrame([data])[model_columns]
        input_df = input_df.apply(pd.to_numeric)
    except (ValueError, TypeError) as e:
         error_message = f"Invalid data type for one or more features (must be numeric): {e}"
         print(f"{COLOR_RED}>>> Server Bad Request (400): {error_message}{COLOR_RESET}")
         return flask.jsonify({"error": error_message}), 400
    except Exception as e:
        print(f"{COLOR_RED}Server Error (500): Error processing input data - {e}{COLOR_RESET}")
        return flask.jsonify({"error": "Internal server error during data processing"}), 500
    try:
        scaled_features = scaler.transform(input_df)
    except Exception as e:
        print(f"{COLOR_RED}Server Error (500): Scaling failed - {e}{COLOR_RESET}")
        return flask.jsonify({"error": "Internal server error during data scaling"}), 500
    predicted_class = "Error"
    prob_dict = {}
    try:
        probabilities = model.predict_proba(scaled_features)
        probs_for_sample = probabilities[0]
        prob_dict = dict(zip(model.classes_, probs_for_sample))
        predicted_class = model.classes_[np.argmax(probs_for_sample)]

        color_code = color_map.get(predicted_class, COLOR_RESET)
        if predicted_class == 'None':
            print(f"--- No Attack Detected: {color_code}{predicted_class}{COLOR_RESET} ---")
        elif predicted_class in color_map:
             print(f"--- Attack Detected: {color_code}{predicted_class}{COLOR_RESET} ---")
        else:
            print(f"--- Prediction (Unknown Type): {predicted_class} ---")

    except AttributeError:
        print(f"{COLOR_YELLOW}Server Warning: Model does not support predict_proba. Falling back to predict.{COLOR_RESET}")
        try:
            prediction = model.predict(scaled_features)
            predicted_class = prediction[0]
            prob_dict = {"error": "Probabilities not available"}
            color_code = color_map.get(predicted_class, COLOR_RESET)
            if predicted_class == 'None':
                print(f"--- No Attack Detected (Fallback): {color_code}{predicted_class}{COLOR_RESET} ---")
            elif predicted_class in color_map:
                 print(f"--- Attack Detected (Fallback): {color_code}{predicted_class}{COLOR_RESET} ---")
            else:
                print(f"--- Prediction (Fallback/Unknown Type): {predicted_class} ---")
        except Exception as e:
             print(f"{COLOR_RED}Server Error (500): Fallback Prediction failed - {e}{COLOR_RESET}")
             return flask.jsonify({"error": "Internal server error during prediction"}), 500
    except Exception as e:
        print(f"{COLOR_RED}Server Error (500): Prediction failed - {e}{COLOR_RESET}")
        return flask.jsonify({"error": "Internal server error during prediction"}), 500

    return flask.jsonify({'prediction': predicted_class, 'probabilities': prob_dict})


if __name__ == '__main__':
    print("Starting Flask web server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
