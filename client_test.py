import requests
import json
import pandas as pd
import joblib
import time
import random
import sys
import math

TRAIN_FILE_PATH = 'train_net.csv'
COLUMNS_FILENAME = 'model_columns.joblib'
API_URL = 'http://127.0.0.1:5000/predict'
TARGET_COLUMN = 'ALERT'
ALL_CLASSES = ['None', 'Port Scanning', 'Denial of Service', 'Malware']
LABEL_FOR_NAN = 'None'
DELAY_BETWEEN_REQUESTS = 0.1

def load_components():
    print("--- Loading Components ---")
    try:
        model_columns = joblib.load(COLUMNS_FILENAME)
        print(f"Loaded expected model columns ({len(model_columns)}) from {COLUMNS_FILENAME}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find '{COLUMNS_FILENAME}'.")
        return None, None
    except Exception as e:
        print(f"FATAL ERROR: Failed to load '{COLUMNS_FILENAME}': {e}")
        return None, None

    columns_to_read = list(set(model_columns + [TARGET_COLUMN]))
    try:
        df_train_full = pd.read_csv(TRAIN_FILE_PATH, usecols=columns_to_read)
        print(f"Loaded training data columns from {TRAIN_FILE_PATH} ({len(df_train_full)} rows)")
        if TARGET_COLUMN not in df_train_full.columns:
            print(f"FATAL ERROR: Target column '{TARGET_COLUMN}' not found in the training data.")
            return model_columns, None
        return model_columns, df_train_full
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find training data file '{TRAIN_FILE_PATH}'.")
        return model_columns, None
    except ValueError as e:
        print(f"FATAL ERROR: Column mismatch reading training CSV '{TRAIN_FILE_PATH}'. Error: {e}")
        return model_columns, None
    except Exception as e:
        print(f"FATAL ERROR: Failed to load training data '{TRAIN_FILE_PATH}': {e}")
        return model_columns, None

def get_samples(df_train_full, target_class_label, num_samples):
    if target_class_label == LABEL_FOR_NAN:
        df_class_specific = df_train_full[pd.isna(df_train_full[TARGET_COLUMN])]
        label_desc = f"where '{TARGET_COLUMN}' is originally NaN"
    else:
        df_class_specific = df_train_full[df_train_full[TARGET_COLUMN] == target_class_label]
        label_desc = f"for class label '{target_class_label}'"

    num_available = len(df_class_specific)
    if num_available == 0:
        print(f"    No rows found {label_desc}. Skipping.")
        return None

    num_to_sample = min(num_samples, num_available)
    if num_to_sample < num_samples:
        print(f"    WARNING: Only {num_available} rows available {label_desc}, sampling {num_to_sample}.")

    sampled_df = df_class_specific.sample(n=num_to_sample, random_state=int(time.time()))
    print(f"    Selected {len(sampled_df)} samples.")
    return sampled_df

def send_request_and_analyze(index, row, model_columns):
    global error_count, success_count, mismatch_count

    original_label_in_csv = row[TARGET_COLUMN]
    expected_prediction_label = LABEL_FOR_NAN if pd.isna(original_label_in_csv) else original_label_in_csv
    print(f"\n--> Processing Sample Index: {index} (Expected: '{expected_prediction_label}')")

    row_features = row[model_columns].copy()
    if 'ANOMALY' in row_features.index and pd.isna(row_features['ANOMALY']):
         row_features['ANOMALY'] = 0

    has_nan_after_fill = False
    nan_cols = []
    for key, value in row_features.items():
        if isinstance(value, float) and math.isnan(value):
            has_nan_after_fill = True; nan_cols.append(key)
    if has_nan_after_fill:
        print(f"  ERROR: NaN present in '{', '.join(nan_cols)}'. Skipping.")
        error_count += 1; return False

    data_dict = row_features.to_dict()
    try:
        json_data = json.dumps(data_dict)
    except TypeError as e:
         print(f"  ERROR: JSON serialization failed: {e}. Skipping."); error_count += 1; return False

    try:
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json_data, timeout=10)
        response.raise_for_status()
        prediction_result = response.json()
        predicted_label = prediction_result.get('prediction', 'ERROR')
        probabilities = prediction_result.get('probabilities', None)

        print(f"  Status Code: {response.status_code}")
        if predicted_label == LABEL_FOR_NAN: print(f"  Analysis Result: No Attack Detected (Predicted '{predicted_label}')")
        elif predicted_label in ['Port Scanning', 'Denial of Service', 'Malware']: print(f"  Analysis Result: Attack Detected ({predicted_label})")
        else: print(f"  Analysis Result: Potential Issue or Unknown Type ({predicted_label})")
        if probabilities and isinstance(probabilities, dict):
             probs_str = ", ".join([f"{k}: {v:.4f}" for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)])
             print(f"  Probabilities: {{{probs_str}}}")
        match = (predicted_label == expected_prediction_label)
        print(f"  Match (Predicted == Expected): {match}")
        success_count += 1
        if not match: mismatch_count += 1
        return True

    except requests.exceptions.HTTPError as http_err:
        print(f"  ERROR: HTTP error: {http_err}")
        error_count += 1
    except requests.exceptions.ConnectionError as conn_err:
        print(f"  FATAL ERROR: Connection error: {conn_err}")
        return 'STOP'
    except requests.exceptions.Timeout as timeout_err:
        print(f"  ERROR: Timeout error: {timeout_err}")
        error_count += 1
    except requests.exceptions.RequestException as req_err:
        print(f"  ERROR: Request error: {req_err}")
        error_count += 1
    except json.JSONDecodeError:
        print(f"  ERROR: Failed to decode JSON response")
        error_count += 1
    return False
model_columns, df_train_full = load_components()

if model_columns is None or df_train_full is None:
    sys.exit(1)

success_count = 0
error_count = 0
mismatch_count = 0

while True:
    print("\n--- Client Menu ---")
    print("1. Test Random Samples for ALL Classes")
    print("2. Test Random Samples for a SPECIFIC Class")
    print("Q. Quit")
    choice = input("Enter your choice: ").strip().upper()

    num_samples_input = 0
    if choice in ['1', '2']:
        while True:
            num_samples_input_str = input(f"  How many samples per class? (e.g., 3): ").strip()
            if num_samples_input_str.isdigit():
                num_samples_input = int(num_samples_input_str)
                if num_samples_input > 0:
                    break
                else:
                    print("  Please enter a positive number.")
            else:
                print("  Invalid input. Please enter a number.")

    if choice == '1':
        print("\n--- Testing ALL Classes ---")
        total_tested_all = 0
        for class_label in ALL_CLASSES:
            df_to_test = get_samples(df_train_full, class_label, num_samples_input)
            if df_to_test is not None:
                total_tested_all += len(df_to_test)
                for index, row in df_to_test.iterrows():
                    result = send_request_and_analyze(index, row, model_columns)
                    if result == 'STOP': break
                    if DELAY_BETWEEN_REQUESTS > 0: time.sleep(DELAY_BETWEEN_REQUESTS)
                if result == 'STOP': break
            print("-" * 20)
        print(f"\nFinished testing all classes. Total samples processed in this run: {total_tested_all}")

    elif choice == '2':
        print("\n--- Testing SPECIFIC Class ---")
        print("Available classes (use exact name or 'None' for original NaN):")
        for i, cls_name in enumerate(ALL_CLASSES): print(f"  {i+1}. {cls_name}")

        while True:
            class_choice_str = input("  Enter the number or exact name of the class to test: ").strip()
            if class_choice_str.isdigit():
                class_idx = int(class_choice_str) - 1
                if 0 <= class_idx < len(ALL_CLASSES):
                    selected_class = ALL_CLASSES[class_idx]
                    break
                else:
                    print("  Invalid number.")
            elif class_choice_str in ALL_CLASSES:
                selected_class = class_choice_str
                break
            elif class_choice_str.lower() == 'none' and LABEL_FOR_NAN in ALL_CLASSES:
                selected_class = LABEL_FOR_NAN
                break
            else:
                print(f"  Invalid class name. Choose from: {ALL_CLASSES}")

        print(f"Testing class: '{selected_class}'")
        df_to_test = get_samples(df_train_full, selected_class, num_samples_input)
        if df_to_test is not None:
            for index, row in df_to_test.iterrows():
                result = send_request_and_analyze(index, row, model_columns)
                if result == 'STOP': break
                if DELAY_BETWEEN_REQUESTS > 0: time.sleep(DELAY_BETWEEN_REQUESTS)
        print(f"\nFinished testing specific class '{selected_class}'.")

    elif choice == 'Q':
        print("\nExiting.")
        break

    else:
        print("\nInvalid choice. Please try again.")

print("\n--- Overall Session Summary ---")
total_attempted = success_count + error_count
print(f"Total API calls attempted in session: {total_attempted}")
print(f"Successful API calls: {success_count}")
print(f"Samples skipped / Errored calls: {error_count}")
if success_count > 0:
    correct_predictions = success_count - mismatch_count
    print(f"Correct predictions (Predicted == Expected): {correct_predictions}")
    print(f"Incorrect predictions (Predicted != Expected): {mismatch_count}")
    accuracy = correct_predictions / success_count
    print(f"Overall accuracy on successfully tested samples: {accuracy:.2%}")
else:
    print("No samples were successfully tested.")
